#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Join phase-local visible-kernel/explicit callback evidence and host counter deltas."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import re

CALLBACK=re.compile(r'(^|::)row_store_(lookup|callback)(\(|$)|ssd_lookup_callback',re.I)
ENQUEUE=re.compile(r'cu(?:da)?LaunchHostFunc',re.I)
RANK=re.compile(r'(?:^|_)rank([0-9]+)(?:[_.]|$)')


def overlap(left,right):
    # Union first, so nested/multistream events cannot multiply overlap time.
    def merge(items):
        result=[]
        for lo,hi in sorted(items):
            if result and lo<=result[-1][1]:result[-1]=(result[-1][0],max(hi,result[-1][1]))
            else:result.append((lo,hi))
        return result
    left,right=merge(left),merge(right);i=j=0;total=0
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j];total+=max(0,min(b,d)-max(a,c))
        if b<d:i+=1
        else:j+=1
    return total


def explicit_spans(path,core):
    opener=gzip.open if path.name.endswith('.gz') else open
    with opener(path,'rb') as inp:raw=inp.read(core.MAX_TRACE_BYTES+1)
    if len(raw)>core.MAX_TRACE_BYTES:raise ValueError('trace too large')
    payload=json.loads(raw);events=payload.get('traceEvents',[]) if isinstance(payload,dict) else payload
    callback=[];comm=[];enqueues=[];names={}
    for event in events:
        if not isinstance(event,dict) or event.get('ph')!='X':continue
        ts,dur=event.get('ts'),event.get('dur');name=str(event.get('name',''))
        if not core.number(ts) or not core.number(dur) or dur<0:continue
        cats={x.strip() for x in str(event.get('cat','')).split(',')}
        if 'kernel' in cats:
            if core.COMMUNICATION.search(name):comm.append((ts,ts+dur))
        elif ENQUEUE.search(name):enqueues.append((ts,ts+dur))
        elif CALLBACK.search(name):
            callback.append((ts,ts+dur));names[name]=names.get(name,0)+1
    return {'explicit_rowstore_cpu_spans_present':bool(callback),'explicit_rowstore_cpu_span_names':names,
            'explicit_rowstore_cpu_span_union_us':core.merged_length(callback) if callback else None,
            'host_callback_enqueue_count':len(enqueues),'host_callback_enqueue_summed_us':sum(b-a for a,b in enqueues),
            'rowstore_cpu_span_communication_temporal_overlap_us':overlap(callback,comm) if callback else None,
            'interpretation':'cudaLaunchHostFunc is enqueue time, never callback execution. Named rowstore CPU spans are inclusive. Temporal overlap with communication is not proof of causation, rank waiting, or exclusive stall time. No cross-rank timestamp alignment is assumed.'}


def counters(text):
    out={}
    for line in text.splitlines():
        parts=line.replace(':','').split()
        if len(parts)==2:
            try:out[parts[0]]=int(parts[1])
            except ValueError:pass
    return out


def host_delta(path):
    first=last=None
    with path.open() as inp:
        for line in inp:
            value=json.loads(line)
            if first is None:first=value
            last=value
    if first is None:return {'error':'no host samples'}
    vm0,vm1=counters(first.get('vmstat','')),counters(last.get('vmstat',''))
    vm={key:vm1[key]-vm0[key] for key in ('pgfault','pgmajfault','pgpgin','pswpin','pswpout') if key in vm0 and key in vm1}
    def disks(text):
        return {p[2]:[int(v) for v in p[3:]] for line in text.splitlines() if len(p:=line.split())>=14}
    d0,d1=disks(first.get('diskstats','')),disks(last.get('diskstats',''))
    disk={name:{'reads_completed':row[0]-d0[name][0],'read_bytes':(row[2]-d0[name][2])*512,
                'read_time_ms':row[3]-d0[name][3],'io_time_ms':row[9]-d0[name][9]}
          for name,row in d1.items() if name in d0}
    pids={}
    for pid,row in last.get('pids',{}).items():
        old=first.get('pids',{}).get(pid,{})
        try:
            a=old['stat'][old['stat'].rfind(')')+2:].split();b=row['stat'][row['stat'].rfind(')')+2:].split()
            if a[19]!=b[19]:pids[pid]={'error':'PID starttime changed'};continue
            pids[pid]={key:int(b[index])-int(a[index]) for key,index in [('minor_faults',7),('major_faults',9),('user_ticks',11),('system_ticks',12)]}
            ia,ib=counters(old.get('io','')),counters(row.get('io',''))
            pids[pid]['io']={key:ib[key]-ia[key] for key in ('rchar','syscr','read_bytes') if key in ia and key in ib}
        except (KeyError,IndexError,ValueError):pids[pid]={'error':'process samples unavailable'}
    return {'elapsed_s':last['monotonic']-first['monotonic'],'system_vm_deltas':vm,'disk_deltas':disk,'process_deltas':pids,
            'interpretation':'System/disk counters include all host activity. Diskstats are block-device accounting; md and member NVMe bytes overlap and must not be summed. CPU ticks are CPU time, not off-CPU wait.'}


def perf_evidence(path):
    if not path.exists() or path.stat().st_size==0:return {'available':False,'reason':'perf report missing or empty'}
    text=path.read_text()[:4*1024*1024];selected=[]
    for line in text.splitlines():
        if re.search(r'row_store|pread|nccl|futex',line,re.I):selected.append(line)
    return {'available':True,'selected_report_lines':selected[:100],
            'interpretation':'perf report --no-children shows sampled on-CPU self-cycle shares across worker/engine/API processes. A sleeping callback is absent from cycles samples; these are not callback wall times or disk-wait durations.'}


def summarize(capture,trace_dir,core):
    meta=json.loads((capture/'diagnostics.json').read_text());result={'status':'complete','phases':[],'errors':[],
        'safe_for_scored_work':meta.get('safe_for_scored_work',False),'capture_status':meta.get('status'),
        'limitations':'12 engine iterations with configured warmup/active schedule cover a bounded portion of each phase. Missing explicit CPU callback spans prevents exclusive Engram-vs-NCCL stall attribution. Communication kernels include work and possible waits; their duration alone does not measure rank waiting.'}
    if meta.get('status')!='completed':result['errors'].append('capture was not completed: '+str(meta.get('status')))
    if not meta.get('phases'):result['errors'].append('capture contains no phases')
    for phase in meta['phases']:
        item={'label':phase['label'],'profile_prefix':phase['profile_prefix'],'traces':[],
              'capture_errors':phase.get('errors',[]),
              'request_capture_complete':(len(phase.get('requests',[]))==phase['concurrency'] and not phase.get('request_errors')) if 'concurrency' in phase else None}
        files=sorted(p for p in trace_dir.rglob('*') if p.is_file() and core.trace_name(p) and phase['profile_prefix'] in p.name)
        if len(files)>core.MAX_FILES:raise ValueError('too many phase traces')
        ranks=set()
        for path in files:
            trace=core.summarize_file(path);trace.update(explicit_spans(path,core));item['traces'].append(trace)
            if trace['kernel_count']:
                match=RANK.search(path.name)
                if match:ranks.add(int(match.group(1)))
        item['observed_gpu_global_ranks']=sorted(ranks);item['all_four_rank_traces_present']=ranks=={0,1,2,3}
        if not item['all_four_rank_traces_present']:result['errors'].append(phase['label']+': missing/unknown GPU rank traces')
        folder=capture/phase['label']
        item['host_counters']=host_delta(folder/'host-counters.jsonl')
        item['cpu_perf']=perf_evidence(folder/'cpu-perf-report.txt');result['phases'].append(item)
    result['gpu_rank_coverage_complete_for_present_phases']=bool(result['phases']) and all(p['all_four_rank_traces_present'] for p in result['phases'])
    if result['errors']:result['status']='partial'
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('capture',type=Path);ap.add_argument('--trace-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--helpers-dir',type=Path,default=Path(__file__).parent);args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('trace_core',args.helpers_dir/'summarize-profiler.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    result=summarize(args.capture,args.trace_dir,core);args.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
