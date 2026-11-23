# SPDX-License-Identifier: Apache-2.0
"""Reject different torch selection semantics before replacing its glue.

The retained model export omitted indexer_topk.py. The operator must qualify
the installed implementation, rather than assuming a recent upstream file
is the served revision. Comments, annotations and docstrings are immaterial.
"""
import ast
import inspect
import textwrap
import hashlib
import importlib
import json
from pathlib import Path

TORCH_BRANCH = '''
row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
cols = torch.arange(logits.shape[1], device=logits.device)
masked = logits.masked_fill(
    cols.unsqueeze(0) >= row_ends.unsqueeze(1), float("-inf"))
indices = masked.topk(topk_tokens, dim=-1).indices
in_range = torch.arange(topk_tokens, device=logits.device).unsqueeze(0) < row_ends.unsqueeze(1)
indices = torch.where(in_range, indices, -1)
topk_indices.copy_(indices)
'''
ROW_ENDS = '''
if seq_lens.numel() == num_rows:
    row_ends = seq_lens.reshape(-1)
else:
    next_n_offsets = torch.arange(next_n, dtype=torch.int32, device=seq_lens.device)
    row_ends = ((seq_lens.reshape(-1, 1) - next_n + 1 + next_n_offsets).clamp_(min=0).reshape(-1))
assert row_ends.dtype == torch.int32
return row_ends
'''


def dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def body(source):
    node = ast.parse(textwrap.dedent(source)).body[0]
    nodes = node.body
    if nodes and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant) and isinstance(nodes[0].value.value, str):
        nodes = nodes[1:]
    return nodes


def validate_sources(forward_source, ends_source):
    tree = ast.parse(textwrap.dedent(forward_source))
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            test = node.test
            if (isinstance(test.left, ast.Name) and test.left.id == "backend"
                and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "torch"):
                matches.append(node.body)
    if len(matches) != 1 or dump(matches[0]) != dump(ast.parse(TORCH_BRANCH).body):
        raise RuntimeError("unqualified indexer torch branch; retain baseline and audit installed source")
    if dump(body(ends_source)) != dump(ast.parse(ROW_ENDS).body):
        raise RuntimeError("unqualified indexer row-end semantics; retain baseline")


def validate_topk(cls):
    if hasattr(cls.forward, '__wrapped__'):
        raise RuntimeError('indexer selector already wrapped; audit plugin composition')
    validate_sources(inspect.getsource(cls.forward), inspect.getsource(cls._row_ends))


def validate_metadata():
    expected = json.loads(Path(__file__).with_name('metadata-contract.json').read_text())
    for target, digest in expected.items():
        module, member = target.split(':')
        cls_name, method = member.split('.')
        cls = getattr(importlib.import_module(module), cls_name)
        fn = getattr(cls, method)
        if hasattr(fn, '__wrapped__'):
            raise RuntimeError(f'metadata method already wrapped: {target}; audit plugin composition')
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        node.decorator_list = []
        if (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        if actual != digest:
            raise RuntimeError(f'unqualified metadata implementation {target}; audit installed source')
