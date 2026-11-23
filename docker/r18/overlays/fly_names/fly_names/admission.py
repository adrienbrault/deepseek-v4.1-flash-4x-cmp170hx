# SPDX-License-Identifier: MIT
"""Finite test admission, derived from bitwise-capture-r1/profile_r2.py.

QV installs this on the constructed scheduler. No waiting inside schedule:
EngineCore must be free to receive the remaining AddRequests.
"""
import re


class Admission:
    def __init__(self, add, control):
        self.add, self.control = add, control
        self.pending, self.cell = [], None

    def check(self):
        c = self.control()
        n = c['admission']
        if type(n) is not int or not 1 <= n <= 8:
            raise ValueError('FLy admission size')
        if self.pending and c != self.cell:
            raise ValueError('FLy cell changed with pending admission')
        return c

    def __call__(self, request, *args, **kwargs):
        c = self.check()
        self.cell = dict(c)
        self.pending.append((request, args, kwargs))
        if len(self.pending) > c['admission']:
            raise ValueError('FLy admission overflow')
        if len(self.pending) == c['admission']:
            batch, self.pending = self.pending, []
            if c['admission'] > 1:
                def index(row):
                    match = re.search(r'(?:^|-)flycell-([0-9a-f]{16})-row([0-7])(?:-|$)', row[0].request_id)
                    if match is None:
                        raise ValueError('FLy barrier request has no cell row ID')
                    if match[1] != c['request_group']:
                        raise ValueError('FLy barrier stale/wrong cell request')
                    return int(match[2])
                batch.sort(key=index)
                if [index(row) for row in batch] != list(range(c['admission'])):
                    raise ValueError('FLy barrier duplicate/missing row ID')
            for req, a, kw in batch:
                self.add(req, *a, **kw)


def install(scheduler, control):
    if not hasattr(scheduler, '_fly_admission'):
        scheduler._fly_admission = Admission(scheduler.add_request, control)
        scheduler.add_request = scheduler._fly_admission
    scheduler._fly_admission.check()
