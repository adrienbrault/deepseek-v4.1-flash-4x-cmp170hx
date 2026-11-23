# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions Copyright 2026 Morrowmake (replicated wkv design, see NOTICE.md)
"""Engram ``wkv`` replicated for decode steps: no output all-gather.

Stock: ``wkv`` is column-parallel with ``gather_output``. Each TP rank runs the MXFP8
Marlin GEMM for its 6400 of the 25600 output columns, then an NCCL all-gather moves
``rows x 6400`` BF16 per rank to every other rank. On opera's PCIe links that gather
is most of the Engram scope (2026-11-09 light timing: 0.17 ms per Engram layer at c1,
0.64 ms at c8, the same on layers 1 and 14 and on all four ranks).

Here, after loading, every rank also holds the other ranks' processed shard bytes
(``weight``, ``weight_scale``: the only tensors MarlinMxfp8LinearKernel.apply_weights
reads). A step of ``min_rows..max_rows`` rows runs each shard's GEMM on this rank (the
call that shard's owner makes: same input, same bytes, same shapes) and concatenates
the outputs in rank order, which is what the all-gather assembles; reviewed.py has
the argument. Other row counts run the stock forward. The choice is a pure function
of the row count, so a CUDA graph captured for a size always replays its own path,
and all ranks take the same path (no collective is skipped on one rank only).

Adapted from Morrowmake's ``sm80_engram_wkv.py`` (Apache-2.0), which replicates the
same projection for eager prefill steps only. Not copied: their prefill-only bound,
their quant-method wrapper and capture integration. Here the replica is built after
``Worker.load_model`` (hooks.py), the own shard is re-pointed at its slot only after
a byte check, every slot's bytes are cross-checked between ranks, and verification
boots compare both paths bitwise inside the served graphs.

Shadow (verification boots): every replicated call also runs the stock forward,
counts on the device (recorded into captured graphs) calls and BF16 elements that
differ bitwise, and returns the STOCK output, so the verify boot writes stock text.
A new nonce in the drain file makes the next eager call synchronize and log FINAL.
"""
import hashlib
import json
import re
from types import SimpleNamespace

from vllm.logger import init_logger

from . import reviewed
from .config import qualification_rows

logger = init_logger('vllm.' + __name__)
TAG = 'DSV41_EG_WKV_REPLICA'
STATE = '_dsv41_wkv_replica'
TENSORS = ('weight', 'weight_scale')
MODELOPT = 'ModelOptLinearMethod'
TOKEN_RE = re.compile(r'[A-Za-z0-9]{8,64}')
# Every operation adoption and the forward take from ``ops`` (vllm_ops() and the CPU fakes).
OPS = frozenset(('tp_info', 'engram_cls', 'linear_cls', 'review', 'device_facts', 'act_dtype', 'route', 'status',
                 'gather_into', 'empty', 'zeros', 'sample_into', 'no_grad', 'byte_dtype', 'supports', 'as_bytes', 'same_bytes', 'checksum',
                 'capturing', 'sync', 'cat', 'count_diff', 'row'))


class Refused(RuntimeError):
    pass


class Runtime:
    """Per process: switches, rank, shadow counters and the drain cursor (shared by both layers)."""

    def __init__(self, cfg, rank, tp, ops, counters=None):
        self.min_rows = cfg['min_rows']
        self.max_rows = cfg['max_rows']
        self.drain_file = cfg.get('drain_file')
        self.rank = rank
        self.tp = tp
        self.ops = ops
        # Shadow: int64 [2, layers, max_rows + 1, 3] = per (eager | CUDA-graph replay, Engram
        # layer slot, row count): calls compared, calls with any difference, differing elements.
        # Allocated at load, before any capture.
        self.counters = counters
        self.drained = None
        self.announced = set()
        self.layers = []

    def selected(self, rows):
        return self.min_rows <= rows <= self.max_rows


class LayerState:
    def __init__(self, runtime, shards, layer, slot):
        self.runtime = runtime
        self.shards = shards
        self.layer = layer
        self.slot = slot


# ------------------------------------------------------------------ eligibility

def _modelopt_api():
    """The image classes the ModelOpt review compares by identity."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic, kMxfp8Static
    return SimpleNamespace(method=modelopt.ModelOptLinearMethod, fmt=modelopt.FormatScheme,
                           weight=kMxfp8Static, activation=kMxfp8Dynamic, kernel=MarlinMxfp8LinearKernel)


def review_quant(linear, api, source_problem):
    """None if ``linear.quant_method`` is exactly the reviewed MXFP8 + Marlin chain."""
    method = getattr(linear, 'quant_method', None)
    if type(method) is not api.method:
        return f'wkv quant method {type(method).__name__} is not the reviewed ModelOptLinearMethod'
    spec = getattr(method, 'spec', None)
    if getattr(spec, 'weight', None) != api.weight or getattr(spec, 'activation', None) != api.activation:
        return f'ModelOpt wkv scheme {spec} is not MXFP8 static weight + MXFP8 dynamic activation'
    if type(getattr(method, 'fmt', None)) is not api.fmt:
        return f'ModelOpt wkv format scheme {type(getattr(method, "fmt", None)).__name__} not reviewed'
    if type(getattr(method, 'kernel', None)) is not api.kernel:
        return f'MXFP8 wkv kernel {type(getattr(method, "kernel", None)).__name__} not reviewed'
    if getattr(linear, 'is_bmm', False):
        return 'batched (BMM) ModelOpt wkv'
    return source_problem()


def layer_problem(engram, linear_cls, tp, review):
    """Why this constructed, loaded Engram layer cannot take the replica, or None."""
    wkv = getattr(engram, 'wkv', None)
    if type(wkv) is not linear_cls:
        return f'wkv is {type(wkv).__name__}, not exactly ColumnParallelLinear'
    if getattr(engram, '_batch', None) is not None:
        return 'batched Engram projection (EngramBatch)'
    if getattr(engram, 'use_sequence_parallel', False):
        return 'sequence parallelism'
    if not getattr(wkv, 'gather_output', False):
        return 'wkv does not gather its output'
    if tp < 2 or getattr(wkv, 'tp_size', None) != tp:
        return f'wkv tp_size {getattr(wkv, "tp_size", None)} is not the TP group size {tp} (>= 2)'
    if getattr(wkv, 'bias', None) is not None or getattr(wkv, 'skip_bias_add', True) or getattr(wkv, 'return_bias', True):
        return 'wkv has a bias or returns one'
    if getattr(wkv, 'output_size_per_partition', 0) * tp != getattr(wkv, 'output_size', -1):
        return 'wkv output is not split evenly over the TP ranks'
    for name in TENSORS:
        t = getattr(wkv, name, None)
        if t is None or not hasattr(t, 'is_contiguous') or not t.is_contiguous():
            return f'wkv.{name} missing or not contiguous'
    return review(wkv)


def describe(layers, cfg, tp):
    """Everything every rank must share before any payload moves: the dispatch contract
    (row bounds, shadow, TP size) and, per Engram layer in module order, its identity,
    logical sizes and the processed tensors' shape / dtype / byte count."""
    out = dict(min_rows=cfg['min_rows'], max_rows=cfg['max_rows'], shadow=bool(cfg['shadow']), tp=tp, layers=[])
    for name, engram in layers:
        wkv = engram.wkv
        out['layers'].append(dict(
            name=name, index=int(engram.layer_hash_index),
            quant=[type(wkv.quant_method).__qualname__, type(getattr(wkv.quant_method, 'kernel', None)).__qualname__],
            sizes=[int(wkv.input_size), int(wkv.output_size),
                   int(wkv.input_size_per_partition), int(wkv.output_size_per_partition)],
            tensors=[[t, list(getattr(wkv, t).shape), str(getattr(wkv, t).dtype),
                      int(getattr(wkv, t).numel() * getattr(wkv, t).element_size())] for t in TENSORS]))
    return out


def digest_ints(obj):
    """Two non-negative 60-bit integers from a JSON digest: fixed-size control payload."""
    h = hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return [int(h[:15], 16), int(h[15:30], 16)]


CONTROL = ('ok', 'layers', 'sm_count', 'cc_major', 'cc_minor', 'route', 'min_rows', 'max_rows', 'shadow',
           'descriptor_a', 'descriptor_b')


def control_row(ok, n_layers, facts, route, cfg, digest):
    row = [int(ok), n_layers, *facts, route, cfg['min_rows'], cfg['max_rows'], int(bool(cfg['shadow'])), *digest]
    assert len(row) == len(CONTROL)
    return row


def agree(local_reason, rows):
    """One verdict from every rank's fixed-size control row, or the reason all refuse."""
    bad = [r for r, row in enumerate(rows) if row[0] != 1]
    if bad:
        return f'rank(s) {bad} ineligible' + (f' (this rank: {local_reason})' if local_reason else '')
    differ = [CONTROL[i] for i in range(1, len(CONTROL)) if len({row[i] for row in rows}) != 1]
    if differ:
        return f'ranks differ in {differ}: {[dict(zip(CONTROL[1:], row[1:])) for row in rows]}'
    return None


# ------------------------------------------------------------------ adoption

class Layer:
    """One Engram wkv and the device buffers adoption will use for it (allocated in prepare())."""

    def __init__(self, slot, name, engram):
        self.slot, self.name, self.engram, self.wkv = slot, name, engram, engram.wkv
        self.local = {}      # tensor name -> this rank's processed tensor (gather input)
        self.stacked = {}    # tensor name -> [tp, *shape] gather output: every rank's slot
        self.wire = {}       # tensor name -> (input bytes, output bytes): uint8 views of the two above
        self.shards = None


def prepare(model, cfg, ops):
    """Phase 1, local only, never raises: checks, descriptor, and EVERY device buffer adoption
    needs later (payload outputs, shadow counters, qualification input, gather output and its
    placeholder). Any failure becomes ``p.reason``, which the control gather makes common."""
    p = SimpleNamespace(rank=-1, tp=0, reason=None, layers=[], facts=[0, 0, 0], route=0, digest=[0, 0],
                        counters=None, sample=None, gathered=None, placeholder=None, n=0)
    try:
        p.rank, p.tp = ops.tp_info()
        if getattr(model, STATE, None) is not None:
            p.reason = 'second load_model in one process (weight reload is not supported)'
        found = [(name, m) for name, m in model.named_modules() if isinstance(m, ops.engram_cls)]
        p.layers = [Layer(slot, name, engram) for slot, (name, engram) in enumerate(found)]
        if not p.layers:
            p.reason = p.reason or 'no Engram layer in the loaded model'
        for lay in p.layers:
            p.reason = p.reason or layer_problem(lay.engram, ops.linear_cls, p.tp, ops.review)
        p.facts = [int(v) for v in ops.device_facts()]
        p.route = int(ops.route())
        # Every collective input dtype must be one the route can carry: payloads travel as bytes,
        # qualification outputs in the activation dtype (PyNCCL's map has no float8_e8m0fnu).
        for dtype in (ops.byte_dtype, ops.act_dtype):
            if not ops.supports(dtype):
                p.reason = p.reason or f'gather route {p.route} cannot carry {dtype}'
        if p.reason is None:
            p.digest = digest_ints(describe(found, cfg, p.tp))
            for lay in p.layers:
                for tname in TENSORS:
                    t = getattr(lay.wkv, tname).data
                    lay.local[tname] = t
                    lay.stacked[tname] = ops.empty((p.tp, *t.shape), t.dtype)
                    # The processed tensors (Marlin int32 weight, float8_e8m0fnu scales) move as
                    # their raw bytes: no conversion, and Marlin keeps the typed views of the same storage.
                    lay.wire[tname] = (ops.as_bytes(t), ops.as_bytes(lay.stacked[tname]))
            if cfg['shadow']:
                p.counters = ops.zeros(2, len(p.layers), cfg['max_rows'] + 1, 3)
            k = {lay.wkv.input_size_per_partition for lay in p.layers}
            p.n = max(lay.wkv.output_size_per_partition for lay in p.layers)
            p.sample = ops.empty((cfg['max_rows'], max(k)), ops.act_dtype)
            ops.sample_into(p.sample)                       # fixed seed: the same bytes on every rank
            p.gathered = ops.empty((p.tp * cfg['max_rows'] * p.n,), ops.act_dtype)
            p.placeholder = ops.zeros(cfg['max_rows'], p.n, dtype=ops.act_dtype)
    except Exception as exc:  # noqa: BLE001  (becomes a common refusal at the control gather)
        p.reason = p.reason or f'local preparation raised {type(exc).__name__}: {exc}'
    return p


def qualify_case(lay, p, rows, ops, problems):
    """Phase 3, one case. Local work fenced, then exactly one gather whatever happened, then the
    fenced comparison. The reference is the stock forward split in its two halves: the same local
    call ColumnParallelLinear.forward makes (quant_method.apply(self, x, None)), and the rank-order
    gather of those outputs laid out as the image's gather does (reshape -> movedim -> reshape,
    checked against the real code by check_image.py)."""
    x = p.sample[:rows, :lay.wkv.input_size_per_partition]
    n = lay.wkv.output_size_per_partition
    local = rep = None
    try:
        local = lay.wkv.quant_method.apply(lay.wkv, x, None)
        if tuple(local.shape) != (rows, n) or local.dtype != ops.act_dtype or not local.is_contiguous():
            raise ValueError(f'local output {tuple(local.shape)} {local.dtype}')
    except Exception as exc:  # noqa: BLE001
        problems.append(f'layer {lay.name} rows {rows} stock local GEMM raised {type(exc).__name__}: {exc}')
        local = None
    if lay.shards is not None:
        try:
            rep = replicated(lay.wkv.quant_method, lay.shards, x, ops.cat)
        except Exception as exc:  # noqa: BLE001
            problems.append(f'layer {lay.name} rows {rows} replica raised {type(exc).__name__}: {exc}')
    out = p.gathered[:p.tp * rows * n].view(p.tp, rows, n)
    ops.gather_into(out, local if local is not None else p.placeholder[:rows, :n])
    try:
        ref = out.movedim(0, 1).reshape(rows, p.tp * n)
        return rep is not None and tuple(rep.shape) == tuple(ref.shape) and int(ops.count_diff(rep, ref)) == 0
    except Exception:  # noqa: BLE001
        return False


def adopt_model(model, cfg, ops):
    """After weight processing, before profiling and capture: build the replicas or refuse.

    Every rank runs the same collectives in the same order whatever fails locally:
    1. prepare(): every local check and every device allocation, fenced -> ONE control gather
       on the CPU (gloo) status channel, fixed size -> common verdict. Payload sizes, buffers and
       the collective route are agreed before any payload moves.
    2. per Engram layer (agreed order), one gather per tensor in TENSORS into its preallocated
       output; then, fenced, the own-slot byte check, checksums and shard views.
    3. self-qualification: for every row count the replica serves, on every layer, the local
       GEMMs (fenced), exactly one gather into a preallocated buffer (a placeholder when the
       local GEMM failed), and the fenced bitwise comparison with the replica.
    4. one status gather (ok, cases, bad cases, slot checksums) -> common verdict.
    Only after (4) passes does any rank re-point its tensors or switch its forward.

    No collective allocates: payload, gather and status buffers all exist before (1)'s verdict,
    so a local exception (an allocation, a GEMM, a checksum) cannot skip a collective. What
    remains is the collectives themselves failing, a communicator fault as in stock serving."""
    p = prepare(model, cfg, ops)
    verdict = agree(p.reason, ops.status(control_row(p.reason is None, len(p.layers), p.facts, p.route,
                                                     cfg, p.digest)))
    if verdict is not None:
        raise Refused(f'{TAG}_REFUSED rank={p.rank} {verdict}')

    rank, tp, problems, sums, extra = p.rank, p.tp, [], [], 0
    for lay in p.layers:
        for tname in TENSORS:
            wire_in, wire_out = lay.wire[tname]
            ops.gather_into(wire_out, wire_in)
    for lay in p.layers:
        ok = True
        for tname in TENSORS:
            try:
                full, local = lay.stacked[tname], lay.local[tname]
                if not ops.same_bytes(full[rank], local):
                    raise ValueError(f'own {tname} slot differs from the loaded bytes')
                sums.append([[int(v) for v in ops.checksum(full[r])] for r in range(tp)])
                extra += full[0].numel() * full.element_size() * (tp - 1)
            except Exception as exc:  # noqa: BLE001
                problems.append(f'layer {lay.name} {tname}: {exc}')
                sums.append([[0, 0]] * tp)
                ok = False
        if ok:
            wkv = lay.wkv
            lay.shards = [wkv if r == rank else SimpleNamespace(
                weight=lay.stacked['weight'][r], weight_scale=lay.stacked['weight_scale'][r], bias=None,
                output_size_per_partition=wkv.output_size_per_partition,
                input_size_per_partition=wkv.input_size_per_partition) for r in range(tp)]

    cases = bad_cases = 0
    with ops.no_grad():
        for lay in p.layers:
            for rows in qualification_rows(cfg['min_rows'], cfg['max_rows']):
                cases += 1
                bad_cases += 0 if qualify_case(lay, p, rows, ops, problems) else 1
    table = ops.status([0 if problems else 1, cases, bad_cases] + [v for row in sums for pair in row for v in pair])
    failed = [r for r, row in enumerate(table) if row[0] != 1]
    if failed:
        raise Refused(f'{TAG}_REFUSED rank={rank} rank(s) {failed} could not build the replica'
                      + (f' (this rank: {problems[0]})' if problems else ''))
    unqualified = {r: row[2] for r, row in enumerate(table) if row[2] or row[1] != table[0][1]}
    if unqualified:
        raise Refused(f'{TAG}_REFUSED rank={rank} self-qualification: replica differs from the gathered wkv '
                      f'in {unqualified} of {table[0][1]} cases per rank')
    for r, row in enumerate(table):
        for t in range(len(sums)):
            for j in range(tp):
                k = 3 + 2 * (t * tp + j)
                if row[k:k + 2] != table[j][k:k + 2]:
                    raise Refused(f'{TAG}_REFUSED rank={rank} rank {r} holds slot {j} of tensor {t} '
                                  'with bytes that differ from its owner')
    runtime = Runtime(cfg, rank, tp, ops, p.counters)
    for lay in p.layers:
        for tname in TENSORS:   # own tensors become views of their checked slot: same bytes, no copy
            getattr(lay.wkv, tname).data = lay.stacked[tname][rank]
        setattr(lay.wkv, STATE, LayerState(runtime, lay.shards, lay.engram.layer_hash_index, lay.slot))
        lay.wkv.__class__ = replica_class(type(lay.wkv))
        runtime.layers.append(lay.engram.layer_hash_index)
    object.__setattr__(model, STATE, runtime)
    layer_list = ','.join(map(str, runtime.layers))
    logger.info('%s_READY rank=%d layers=%s min_rows=%d max_rows=%d shadow=%d extra_mib=%.1f facts=%s',
                TAG, rank, layer_list, runtime.min_rows, runtime.max_rows, int(p.counters is not None),
                extra / 2**20, ','.join(map(str, p.facts)))
    logger.info('%s_QUALIFIED rank=%d layers=%s rows=%d-%d cases=%d bad_cases=%d',
                TAG, rank, layer_list, runtime.min_rows, runtime.max_rows, cases, bad_cases)
    return runtime


_CLASSES = {}


def replica_class(base):
    """``base`` with this forward; one subclass per base class."""
    cls = _CLASSES.get(base)
    if cls is None:
        stock = base.forward

        def forward(self, input_):
            return replica_forward(self, input_, stock)
        cls = _CLASSES[base] = type('ReplicatedDecodeWkv', (base,), dict(
            forward=forward, __module__=__name__, __doc__='Engram wkv: shard GEMMs on every rank for selected row counts.'))
    return cls


# ------------------------------------------------------------------ forward

def replicated(method, shards, x, cat):
    """Every rank's shard GEMM on this rank, outputs in rank order (the gathered row)."""
    return cat([method.apply(shard, x, None) for shard in shards])


def replica_forward(layer, x, stock):
    st = layer.__dict__.get(STATE)
    if st is None:
        return stock(layer, x)
    rt = st.runtime
    capturing = bool(rt.ops.capturing())
    if rt.drain_file and rt.counters is not None and not capturing:
        drain(rt)
    rows = x.shape[0]
    if not rt.selected(rows):
        return stock(layer, x)
    key = (rows, capturing)
    if key not in rt.announced:  # once per row count, separately for eager/warmup and graph capture
        rt.announced.add(key)
        logger.info('%s_ACTIVE rank=%d rows=%d capturing=%d', TAG, rt.rank, rows, int(capturing))
    out = replicated(layer.quant_method, st.shards, x, rt.ops.cat)
    if rt.counters is None:
        return out
    ref = stock(layer, x)
    bad = rt.ops.count_diff(out, ref)
    # Recorded into the graph when capturing, so [1] grows only on replay.
    rt.counters[int(capturing), st.slot, rows].add_(rt.ops.row(bad))
    return ref


def read_token(path):
    try:
        with open(path) as f:
            token = f.read(80).strip()
    except OSError:
        return None
    return token if TOKEN_RE.fullmatch(token) else None


def summarize(counters, layers):
    """(graph totals, eager totals, 'layer:rows:calls:bad,...' of graph replays) from the counters."""
    totals = [[0, 0, 0], [0, 0, 0]]
    sizes = []
    for mode, per_layer in enumerate(counters):
        for slot, per_rows in enumerate(per_layer):
            for rows, (calls, bad_calls, bad_elems) in enumerate(per_rows):
                totals[mode] = [totals[mode][0] + calls, totals[mode][1] + bad_calls, totals[mode][2] + bad_elems]
                if mode == 1 and calls:
                    sizes.append(f'{layers[slot]}:{rows}:{calls}:{bad_calls}')
    return totals[1], totals[0], ','.join(sizes) or '-'


def drain(rt):
    """Terminal receipt: once per new nonce, after ALL queued device work, log the counters."""
    token = read_token(rt.drain_file)
    if token is None or token == rt.drained:
        return
    rt.drained = token
    rt.ops.sync()
    graph, eager, sizes = summarize(rt.counters.cpu().tolist(), rt.layers)
    logger.info('%s_FINAL rank=%d token=%s calls=%d bad_calls=%d bad_elems=%d eager=%d:%d:%d sizes=%s',
                TAG, rt.rank, token, *graph, *eager, sizes)


# ------------------------------------------------------------------ runtime bindings

def vllm_ops():
    """The real torch/vLLM operations (CPU tests substitute fakes)."""
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.models.deepseek_v41.common.engram import Engram

    def device():
        return torch.device('cuda', torch.cuda.current_device())

    def status(values):
        # Control and verdict rows travel on the TP group's CPU (gloo) group: no device memory,
        # so the channel still works when a device allocation or kernel failed on one rank.
        mine = torch.tensor(values, dtype=torch.int64)
        rows = [torch.empty_like(mine) for _ in range(get_tp_group().world_size)]
        dist.all_gather(rows, mine, group=get_tp_group().cpu_group)
        return [r.tolist() for r in rows]

    routes = []

    def route():
        """1: the TP group's PyNCCL communicator (what CudaCommunicator.all_gather uses),
        2: torch.distributed on the device group (DeviceCommunicatorBase.all_gather's call).
        Both fill a caller-provided rank-major output and allocate nothing."""
        if not routes:
            comm = get_tp_group().device_communicator
            pynccl = getattr(comm, 'pynccl_comm', None)
            if pynccl is not None and not pynccl.disabled:
                if pynccl.device != device():
                    raise RuntimeError(f'PyNCCL device {pynccl.device} is not {device()}')
                routes.append((1, pynccl.all_gather))
            else:
                group = get_tp_group().device_group
                routes.append((2, lambda out, inp: dist.all_gather_into_tensor(out, inp, group=group)))
        return routes[0][0]

    def gather_into(out, inp):
        # out: contiguous, tp x inp.numel() elements of inp's dtype, allocated before the control verdict.
        routes[0][1](out, inp)

    def supports(dtype):
        if route() != 1:
            return dtype in (torch.uint8, torch.bfloat16)   # both in torch's ProcessGroupNCCL map
        from vllm.distributed.device_communicators.pynccl_wrapper import ncclDataTypeEnum
        return bool(ncclDataTypeEnum.supports_torch_dtype(dtype))

    def as_bytes(t):
        if not t.is_contiguous():
            raise ValueError(f'{tuple(t.shape)} {t.dtype} tensor is not contiguous')
        return t.view(-1).view(torch.uint8)

    def checksum(t, chunk=1 << 22):
        # Byte sum and position-weighted byte sum, in 4 MiB chunks (small transient memory).
        data = t.contiguous().view(-1).view(torch.uint8)
        total = weighted = 0
        for start in range(0, data.numel(), chunk):
            part = data[start:start + chunk].to(torch.int64)
            pos = torch.arange(start, start + part.numel(), device=part.device, dtype=torch.int64) % 65521 + 1
            total += int(part.sum().item())
            weighted += int((part * pos).sum().item())
        return [total, weighted]

    def device_facts():
        p = torch.cuda.get_device_properties(device())
        return [int(p.multi_processor_count), int(p.major), int(p.minor)]

    def bits(t):
        return t.contiguous().view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])

    def count_diff(a, b):
        return (bits(a) != bits(b)).sum()

    def row(bad):
        bad = bad.to(torch.int64)
        return torch.stack((torch.ones_like(bad), (bad > 0).to(torch.int64), bad))

    def sample_into(buf):
        # Built on the CPU from a private fixed-seed generator: the same bytes on every rank,
        # and the serving RNG streams are neither read nor reseeded.
        gen = torch.Generator().manual_seed(20261109)
        buf.copy_(torch.randn(tuple(buf.shape), generator=gen).to(buf.dtype))

    api = []

    def review(wkv):
        if not api:   # resolved lazily: an import failure is a local reason, not a pre-collective raise
            api.append(_modelopt_api())
        return review_quant(wkv, api[0], reviewed.source_problem)

    return SimpleNamespace(
        tp_info=lambda: (int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())),
        engram_cls=Engram, linear_cls=ColumnParallelLinear, review=review, device_facts=device_facts,
        act_dtype=torch.bfloat16,   # Engram feeds wkv BF16 rows; wkv returns BF16
        route=route, status=status, gather_into=gather_into,
        empty=lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device()),
        zeros=lambda *shape, dtype=torch.int64: torch.zeros(*shape, dtype=dtype, device=device()),
        sample_into=sample_into, no_grad=torch.inference_mode,
        byte_dtype=torch.uint8, supports=supports, as_bytes=as_bytes,
        same_bytes=lambda a, b: bool(torch.equal(a.contiguous().view(-1).view(torch.uint8),
                                                 b.contiguous().view(-1).view(torch.uint8))),
        checksum=checksum, capturing=torch.cuda.is_current_stream_capturing, sync=torch.cuda.synchronize,
        cat=lambda parts: torch.cat(parts, dim=-1), count_diff=count_diff, row=row)
