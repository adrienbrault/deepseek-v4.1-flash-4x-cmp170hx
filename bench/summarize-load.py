"""Align one-second load counters with timestamps in the actual loader log."""
import argparse
import collections
import datetime as dt
import json
import gzip
from pathlib import Path
import re
import statistics as st


def analyze(root):
    def read(name):
        path = root / name
        return path.read_text() if path.exists() else gzip.open(str(path) + '.gz', 'rt').read()
    source = read('serve.log')
    started = dt.datetime.fromisoformat((root / 'boot-telemetry-times.txt').read_text().splitlines()[0])
    def timeof(s):
        value = dt.datetime.combine(started.date(), dt.time.fromisoformat(s), tzinfo=dt.timezone.utc).timestamp()
        return value + 86400 if value < started.timestamp() - 43200 else value
    start = timeof(re.search(r'INFO \d\d-\d\d (\d\d:\d\d:\d\d).*Loading model from scratch', source)[1])
    finishes = [(timeof(m[1]), float(m[2])) for m in re.finditer(
        r'INFO \d\d-\d\d (\d\d:\d\d:\d\d).*Model loading took .*? and ([\d.]+) seconds', source)]
    end = max(x[0] for x in finishes)
    passes = [(timeof(m[1]), float(m[2])) for m in re.finditer(
        r'INFO \d\d-\d\d (\d\d:\d\d:\d\d).*Loading weights took ([\d.]+) seconds', source)]
    disk_start = dt.datetime.fromisoformat((root / 'boot-telemetry-times.txt').read_text().splitlines()[0]).timestamp()
    disk = []
    for line in (root / 'boot-iostat.txt').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'nvme0n1':
            disk.append((disk_start + len(disk) + 1, list(map(float, fields[1:]))))
    pcie = []
    for line in (root / 'boot-pcie.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0].isdigit():
            try:
                pcie.append((timeof(fields[1]), int(fields[2]), float(fields[3]), float(fields[4])))
            except ValueError:
                pass
    cpu = []
    for line in read('boot-pidstat.txt').splitlines():
        fields = line.split()
        if fields and fields[-1].startswith('VLLM::Worker') and len(fields) == 20:
            clock = dt.datetime.strptime(fields[0] + ' ' + fields[1], '%I:%M:%S %p').strftime('%H:%M:%S')
            cpu.append((timeof(clock), int(fields[3]), float(fields[4]), float(fields[5]), float(fields[8]),
                        float(fields[10]), float(fields[11])))
    result = {'date_utc': started.date().isoformat(), 'measurement': 'one-second counters; disk sample times inferred from collector start (approximately one-second alignment)',
              'model_load_reported_s': max(x[1] for x in finishes), 'phases': {}}
    phases = [('whole_model_load', start, end)]
    for i, (t, duration) in enumerate(passes):
        phases.append((f'weight_iterator_{i+1}', t-duration, t))
    for name, lo, hi in phases:
        d = [v for t,v in disk if lo <= t <= hi]
        c = [v for v in cpu if lo <= v[0] <= hi]
        g = [v for v in pcie if lo <= v[0] <= hi]
        row = {'duration_s':hi-lo,'utc_start':dt.datetime.fromtimestamp(lo,dt.timezone.utc).isoformat(),
               'disk_samples':len(d), 'disk_read_GB': sum(v[1]*1024 for v in d)/1e9,
               'disk_read_mean_MB_s':st.mean(v[1]*1024/1e6 for v in d),
               'disk_read_max_MB_s':max(v[1]*1024/1e6 for v in d),
               'disk_r_await_mean_ms':st.mean(v[4] for v in d),
               'disk_util_mean_pct':st.mean(v[-1] for v in d),
               'disk_queue_mean':st.mean(v[-2] for v in d),
               'worker_cpu_mean_pct':st.mean(v[4] for v in c),
               'worker_usr_mean_pct':st.mean(v[2] for v in c),
               'worker_system_mean_pct':st.mean(v[3] for v in c),
               'worker_minor_fault_mean_s':st.mean(v[5] for v in c),
               'worker_major_fault_mean_s':st.mean(v[6] for v in c),
               'pcie_by_gpu':{str(gpu):{'rx_mean_MB_s':st.mean(v[2] for v in g if v[1]==gpu),
                                     'rx_max_MB_s':max(v[2] for v in g if v[1]==gpu),
                                     'tx_mean_MB_s':st.mean(v[3] for v in g if v[1]==gpu)} for gpu in range(4)}}
        result['phases'][name] = row
    dsos = collections.defaultdict(float)
    for line in read('loader-perf-report.txt').splitlines():
        m = re.match(r'\s*([\d.]+)%\s+\S+\s+(\S+)\s+\[.\]',line)
        if m: dsos[m[2]] += float(m[1])
    result['native_sample_cpu_percent_by_dso'] = dict(sorted(dsos.items(),key=lambda x:-x[1]))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    data=analyze(a.directory)
    a.out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
