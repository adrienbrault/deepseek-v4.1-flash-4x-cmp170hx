# SPDX-License-Identifier: MIT
import argparse, itertools, json, math, random, sys, threading, time, urllib.request
PROSE = 'The maintainers agreed that the loader should verify every shard before the model is placed, because a silent mismatch costs more than a refused boot. The change touches the manifest reader, the placement planner and the two tests that cover them. '
CODE = 'def resolve(self, name, *, strict=True):\n    entry = self._index.get(name)\n    if entry is None:\n        if strict:\n            raise KeyError(name)\n        return None\n    return entry.load()\n\n'
TOK_PER_WORD = {'prose': 1.125, 'code': 1.8}

def filler(tokens, kind, rng):
    unit = (CODE if kind == 'code' else PROSE).split()
    if tokens <= 0:
        return ''
    return ' '.join((rng.choice(unit) for _ in range(int(tokens / TOK_PER_WORD[kind])))) + ' '

def post(url, model, prompt, ntok, timeout, temp):
    body = {'model': model, 'prompt': prompt, 'max_tokens': ntok, 'min_tokens': ntok, 'temperature': temp, 'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(url + '/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    tf = None
    n = 0
    text = []
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                d = line[5:].strip()
                if d == '[DONE]':
                    break
                try:
                    c = json.loads(d)
                except Exception:
                    continue
                if c.get('usage'):
                    usage = c['usage']
                for ch in c.get('choices') or []:
                    piece = ch.get('text') or (ch.get('delta') or {}).get('content') or ''
                    if piece:
                        if tf is None:
                            tf = time.time()
                        n += 1
                        text.append(piece)
    except Exception as e:
        return (False, None, time.time() - t0, 0, None, f'ERROR {e}', 0)
    pt = (usage or {}).get('prompt_tokens')
    ct = (usage or {}).get('completion_tokens')
    return (True, tf - t0 if tf else None, time.time() - t0, ct if ct else n, pt, ''.join(text), n)

def draw(sid, a, rng):
    if a.ladder:
        return a.ladder[sid % len(a.ladder)]
    spread = max(0.0, a.spread)
    if a.dist == 'lognormal':
        return (max(4000, int(a.base * math.exp(rng.gauss(0.0, spread)))), max(2, int(round(a.steps * math.exp(rng.gauss(0.0, spread * 0.6))))))
    return (max(4000, int(a.base * rng.uniform(1 - spread, 1 + spread))), max(2, int(round(a.steps * rng.uniform(1 - spread, 1 + spread)))))

def session(sid, a, rng, sink, lock, stop, inst=0):
    base_t, nsteps = draw(sid, a, rng)
    base = filler(base_t, 'code' if sid % 2 else 'prose', rng)
    convo = base + '\n\nTask: find why the loader refuses the shard and propose a patch.\n'
    print(f'[s{sid}#{inst}] base ~{base_t} tokens, {nsteps} steps', flush=True)
    for step in range(nsteps):
        if stop.is_set():
            return
        gen = max(64, int(rng.gauss(a.gen, a.gen * 0.5)))
        t_sub = time.time()
        ok, ttft, wall, n, pt, text, frames = post(a.url, a.model, convo, gen, a.timeout, a.temp)
        rec = {'session': sid, 'inst': inst, 'step': step, 'steps_asked': nsteps, 'base_asked': base_t, 'ok': ok, 't_submit': t_sub, 'ttft_s': ttft, 'wall_s': wall, 'gen_tokens': n, 'asked_tokens': gen, 'prompt_tokens': pt, 'sse_frames': frames, 'prompt_chars': len(convo)}
        with lock:
            sink.write(json.dumps(rec) + '\n')
            sink.flush()
            if ok:
                a.done_steps[sid] = a.done_steps.get(sid, 0) + 1
        if not ok:
            print(f'[s{sid}#{inst} step{step}] {text[:120]}', flush=True)
            return
        print(f"[s{sid}#{inst} step{step}/{nsteps}] prompt~{pt or '?'} gen {n} ttft {(ttft if ttft is None else round(ttft, 2))} wall {wall:.1f}s", flush=True)
        convo += text + '\n\n[tool output]\n' + filler(a.tool, 'code', rng) + '\n'
        if stop.wait(max(0.0, rng.gauss(a.think, a.think * 0.4))):
            return
LADDER_PROD = '3500:9,4000:9,5000:28,6000:23,7000:16,9000:28,9500:7,10500:23,12500:30,20500:19'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sessions', type=int, default=6)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--base', type=int, default=22000, help='starting context per session, tokens')
    p.add_argument('--gen', type=int, default=700, help='mean forced generation length')
    p.add_argument('--tool', type=int, default=900, help='synthetic tool output appended per step, tokens')
    p.add_argument('--think', type=float, default=6.0, help="mean seconds between a session's steps")
    p.add_argument('--dist', choices=('lognormal', 'uniform'), default='lognormal', help="shape of the per-session draw. lognormal is the default because production's prompt distribution is skewed (median 26k, p90 65k) and a symmetric draw cannot hit both.")
    p.add_argument('--spread', type=float, default=0.5, help="per-session variation in base size and step count, as a fraction (0 = every session identical, which is what made R593's prompt p90 far too low)")
    p.add_argument('--ladder', default='', help="fixed session shapes, 'base:steps,base:steps,...'. Overrides --base/--steps/--dist/--spread and makes the prompt distribution identical in every arm. The default in LADDER_PROD matches production's median and p90 to better than 0.1 %%.")
    p.add_argument('--prod-ladder', action='store_true', help='use the solved production ladder')
    p.add_argument('--max-fail-streak', type=int, default=3, help='consecutive no-progress sessions on one slot before the whole run aborts. Guards against respawning into a dead server, which cost R600 an arm and 14,245 requests.')
    p.add_argument('--drain', action='store_true', help='at --max-seconds, stop STARTING sessions and let in-flight ones finish, instead of cutting them. Without it the last instance of every slot is truncated by definition, which removes exactly the deep-session tail the ladder was solved for: R599 lost 10 of 32 instances that way and its realized p90 landed 7 %% under the designed 64,560.')
    p.add_argument('--drain-grace', type=float, default=420, help='hard cap on the drain, so a stuck session cannot hold the GPU past the round')
    p.add_argument('--respawn', action='store_true', help='restart a session slot when it finishes, so concurrency does not decay as the shallow ladder entries retire')
    p.add_argument('--stagger', type=float, default=4.0, help='seconds between session starts')
    p.add_argument('--temp', type=float, default=0.6, help='real clients sample; 0 would be the probe regime')
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--max-seconds', type=float, default=900)
    a = p.parse_args()
    if a.prod_ladder and (not a.ladder):
        a.ladder = LADDER_PROD
    if a.ladder:
        a.ladder = [tuple((int(x) for x in e.split(':'))) for e in a.ladder.split(',') if e.strip()]
        a.sessions = len(a.ladder)
    seed = a.seed if a.seed is not None else int(time.time())
    nonce = int(time.time() * 1000) % 100000 if a.seed is None else seed
    shape = 'ladder ' + ','.join((f'{b}:{s}' for b, s in a.ladder)) if a.ladder else f'{a.sessions} sessions x {a.steps} steps, base {a.base}, {a.dist} spread {a.spread}'
    print(f'agent_replay: {shape}, gen ~{a.gen}, tool {a.tool}, think ~{a.think}s, stagger {a.stagger}s, temp {a.temp}, seed {seed}, content nonce {nonce}', flush=True)
    sink = open(a.out, 'a')
    lock = threading.Lock()
    stop = threading.Event()
    abort = threading.Event()
    done_steps = {}
    a.done_steps = done_steps
    t0 = time.time()
    deadline = t0 + a.max_seconds
    fail_streak = {}

    def worker(sid):
        for inst in itertools.count():
            if stop.is_set() or time.time() > deadline:
                return
            t_inst = time.time()
            before = done_steps.get(sid, 0)
            session(sid, a, random.Random(seed + sid * 7919 + inst * 104729 + nonce), sink, lock, stop, inst)
            if not a.respawn:
                return
            progressed = done_steps.get(sid, 0) > before
            fail_streak[sid] = 0 if progressed else fail_streak.get(sid, 0) + 1
            if fail_streak[sid] >= a.max_fail_streak:
                print(f'[s{sid}] {fail_streak[sid]} consecutive sessions made no progress -- stopping the run', flush=True)
                abort.set()
                stop.set()
                return
            if not progressed:
                if stop.wait(min(30.0, 2.0 * fail_streak[sid])):
                    return
            elif time.time() - t_inst < 1.0:
                if stop.wait(1.0):
                    return
    threads = []
    for s in range(a.sessions):
        th = threading.Thread(target=worker, args=(s,), daemon=True)
        threads.append(th)
        th.start()
        if stop.wait(a.stagger):
            break
    while any((t.is_alive() for t in threads)):
        if time.time() > deadline:
            if a.drain:
                print(f'max-seconds reached, draining in-flight sessions (grace {a.drain_grace:.0f}s)', flush=True)
                hard = time.time() + a.drain_grace
                while any((t.is_alive() for t in threads)) and time.time() < hard:
                    time.sleep(1)
                if any((t.is_alive() for t in threads)):
                    print('drain grace expired, cutting what is left', flush=True)
                    stop.set()
            else:
                print('max-seconds reached, stopping sessions', flush=True)
                stop.set()
            break
        time.sleep(1)
    for t in threads:
        t.join(timeout=120)
    stop.set()
    print(f'agent_replay: {time.time() - t0:.0f} s wall. Score this run with agent_score.py.', flush=True)
    if abort.is_set():
        print('agent_replay: ABORTED -- the server stopped answering; this arm did not measure anything', flush=True)
        return 2
    return 0
if __name__ == '__main__':
    sys.exit(main() or 0)
