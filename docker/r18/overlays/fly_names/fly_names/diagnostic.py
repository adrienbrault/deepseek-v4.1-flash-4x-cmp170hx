# SPDX-License-Identifier: MIT
"""Pure response and executed-state analysis, also usable for offline replay."""
import argparse
import json
from pathlib import Path

from .common import need
from .policy import identity


def token_identity(choice):
    ids = choice.get('token_ids')
    need(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids), 'missing token IDs')
    return ids


def repeated(groups, logprobs):
    need(len(groups) >= 2, 'diagnostic repeats missing')
    key = identity if logprobs else token_identity
    signatures = [[key(r['choice']) for r in g['rows']] for g in groups]
    need(signatures[0] and all(len(s) == len(signatures[0]) for s in signatures), 'diagnostic population changed')
    return all(s == signatures[0] for s in signatures[1:])


def greedy_audit(choice, stop_ids, min_tokens):
    """Raw logprobs precede min_tokens masking; never mistake masked EOS for RNG.

    Stop IDs come from effective server SamplingParams, or a cited model
    fixture for the historical offline replay. No baked-in EOS token value.
    """
    identity(choice)  # validate lengths, finite numbers and real API format
    result = []
    lp = choice['logprobs']
    for pos, (tid, value, top) in enumerate(zip(choice['token_ids'], lp['token_logprobs'], lp['top_logprobs'])):
        parsed = {}
        for key, v in top.items():
            need(key.startswith('token_id:') and key[9:].isdigit(), 'diagnostic requires token-ID logprobs')
            parsed[int(key[9:])] = v
        better = [k for k, v in parsed.items() if v > value]
        if better:
            masked = [k for k in better if k in stop_ids and pos < min_tokens]
            result.append(dict(position=pos, chosen=tid, chosen_logprob=value,
                               better_ids=better, masked_stop_ids=masked,
                               explained_by_min_tokens=len(masked) == len(better)))
    return result


def historical(gate, config):
    need(len(gate['a']['rows']) == len(gate['b']['rows']) == 4, 'historical c4 population missing')
    eos = config['eos_token_id']
    stop_ids = eos if isinstance(eos, list) else [eos]
    output = {}
    for arm in ('a', 'b'):
        output[arm] = [dict(row=i, exceptions=greedy_audit(r['choice'], stop_ids, len(r['choice']['token_ids'])))
                       for i, r in enumerate(gate[arm]['rows'])]
    output['first_token_logprob_delta'] = [b['choice']['logprobs']['token_logprobs'][0] -
        a['choice']['logprobs']['token_logprobs'][0] for a, b in zip(gate['a']['rows'], gate['b']['rows'])]
    output['tokens_equal'] = [token_identity(a['choice']) == token_identity(b['choice'])
                            for a, b in zip(gate['a']['rows'], gate['b']['rows'])]
    output['first_divergence'] = [next((i for i,(x,y) in enumerate(zip(a['choice']['token_ids'], b['choice']['token_ids'])) if x != y), None)
                                 for a,b in zip(gate['a']['rows'], gate['b']['rows'])]
    return output


def classify(captures, observations, raw):
    """Return hypotheses supported by this instrumented boot, not a serving fix."""
    findings, faults, greedy = [], [], []
    for label, groups in captures.items():
        for group in groups:
            for row in group['rows']:
                seen = [r for r in observations if r['cell'] == group['cell'] and r['request_id'] == row['internal_id']]
                need(seen, 'diagnostic worker evidence missing: ' + group['cell'])
                need(len({r['rank'] for r in seen}) == 4, 'diagnostic evidence missing a TP rank')
                need(any(r['path'] in ('gathered', 'target-argmax') for r in seen), 'diagnostic verifier path missing')
                if any(r['params'] != seen[0]['params'] for r in seen):
                    faults.append('REQUEST_PARAMS_CHANGED')
                for r in seen:
                    p = r['params']
                    if r['name'] != raw or r['lora_id'] != r['no_lora_id'] or r['adapter'] is not None:
                        faults.append('RAW_LORA_MAPPING_MISMATCH')
                    if p['temperature'] != 0 or r['temperature_cpu'] != 0 or r['temperature_gpu'] != 0:
                        faults.append('EFFECTIVE_SAMPLING_MISMATCH')
                    if any(seed != row['seed'] for seed in (p['seed'], r['seed_cpu'], r['seed_gpu'], r['verifier_seed_gpu'])):
                        faults.append('SLOT_SEED_MISMATCH')
                p = seen[0]['params']
                if row['choice'].get('logprobs') is not None:
                    exceptions = greedy_audit(row['choice'], p['all_stop_token_ids'], p['min_tokens'])
                    greedy.append(dict(cell=group['cell'], request_id=row['id'], exceptions=exceptions))
                    if any(not e['explained_by_min_tokens'] for e in exceptions):
                        faults.append('UNEXPLAINED_GREEDY_LOGPROBS')
    stability = {label: repeated(groups, label.endswith('-lp')) for label, groups in captures.items()}
    for label, groups in captures.items():
        if len(groups) >= 3 and not stability[label] and repeated(groups[1:], label.endswith('-lp')):
            findings.append('FIRST_REPEAT_DIFFERS:' + label)
    for label in ('barrier-tokens', 'barrier-lp'):
        for group in captures[label]:
            need(any(len(r['batch_request_ids']) == 4 and r['num_draft_tokens'] > 0
                     for r in observations if r['cell'] == group['cell']), 'barrier lacks c4 speculative batch evidence')
    for mode in ('burst', 'burst-k3', 'ordered', 'barrier'):
        if stability[mode + '-tokens'] and not stability[mode + '-lp']:
            findings.append('LOGPROB_PATH_INSTABILITY:' + mode)
    if not all(stability['burst-' + p] for p in ('tokens', 'lp')) and all(stability['barrier-' + p] for p in ('tokens', 'lp')):
        findings.append('ADMISSION_SHAPE_DEPENDENCE')
    if not all(stability['burst-' + p] for p in ('tokens', 'lp')) and all(stability['burst-k3-' + p] for p in ('tokens', 'lp')):
        findings.append('ADAPTIVE_VERIFY_SHAPE_DEPENDENCE')
    if not all(stability['burst-k3-' + p] for p in ('tokens', 'lp')) and all(stability['barrier-' + p] for p in ('tokens', 'lp')):
        findings.append('ADMISSION_DEPENDENCE_AT_K3')
    if not all(stability['barrier-' + p] for p in ('tokens', 'lp')):
        faults.append('MATCHED_BATCH_BASELINE_UNSTABLE')
    for i in range(4):
        if not stability[f'c1-{i}-lp'] or not stability[f'c1-{i}-tokens']:
            faults.append('C1_BASELINE_UNSTABLE')
    return dict(status='PASS' if not faults else 'BLOCKED', findings=sorted(set(findings)),
                faults=sorted(set(faults)), stability=stability, greedy_audit=greedy,
                logprobs_preserve_tokens={mode: [token_identity(r['choice']) for r in captures[mode+'-tokens'][0]['rows']] ==
                    [token_identity(r['choice']) for r in captures[mode+'-lp'][0]['rows']]
                    for mode in ('burst', 'burst-k3', 'ordered', 'barrier')},
                serving_bug_suspected=bool(set(faults) & {'RAW_LORA_MAPPING_MISMATCH', 'EFFECTIVE_SAMPLING_MISMATCH',
                    'SLOT_SEED_MISMATCH', 'C1_BASELINE_UNSTABLE'}),
                attribution='Instrumented current-r7a OFF; no uninstrumented serving control in this boot.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('gate', type=Path); p.add_argument('--model-config', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(historical(json.loads(a.gate.read_text()), json.loads(a.model_config.read_text())), indent=2))
