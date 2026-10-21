#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unscored, loaded-model c8-decode / cold-8k-c1 / cold-8k-c4 diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time
import urllib.request


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def proc_sample(pids):
    result={'monotonic':time.monotonic(),'pids':{}}
    for name in ('diskstats','vmstat'):
        try:result[name]=Path('/proc',name).read_text()
        except OSError as error:result[name+'_error']=str(error)
    for pid in pids:
        row={};base=Path('/proc',str(pid))
        for name in ('stat','io','status'):
            try:row[name]=(base/name).read_text()
            except OSError as error:row[name+'_error']=str(error)
        result['pids'][str(pid)]=row
    return result


class Sampler:
    def __init__(self,path,pids):self.path=path;self.pids=pids;self.done=threading.Event();self.thread=None
    def start(self):
        def run():
            with self.path.open('w') as out:
                while True:
                    out.write(json.dumps(proc_sample(self.pids))+'\n');out.flush()
                    if self.done.wait(1):break
                out.write(json.dumps(proc_sample(self.pids))+'\n')
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
    def stop(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=3)


STOP_ACK_TIMEOUT=60
CLEANUP_RESERVE=80  # stop ACK + perf/sampler/report/metrics bounds


def stop_perf(child):
    status={'sigint_requested':False,'recorder_force_killed':False}
    if child is None:return status
    if child.poll() is None:
        try:
            os.killpg(child.pid,signal.SIGINT);status['sigint_requested']=True
        except ProcessLookupError:pass
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        status['recorder_force_killed']=True
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=3)
    # Remove only remaining descendants in our owned group (e.g. sleep100).
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    return status


def prepare_perf_report(child,termination,folder,phase,observers_stopped):
    """Validate recorder lifecycle/data before scheduling CPU-only parsing."""
    if child is None:return False
    phase['perf_returncode']=child.returncode
    phase['perf_stop']=termination
    phase['perf_record_accepted']=False
    expected_interrupt=(child.returncode==-signal.SIGINT and
                        termination.get('sigint_requested',False) and
                        not termination.get('recorder_force_killed',False))
    phase['perf_expected_interrupt']=expected_interrupt
    if child.returncode!=0 and not expected_interrupt:
        phase['errors'].append('perf record exit '+str(child.returncode));return False
    if not observers_stopped:return False
    data=folder/'cpu-perf.data'
    if not data.is_file() or data.stat().st_size==0:
        phase['errors'].append('perf data missing or empty');return False
    phase['perf_report_status']='pending'
    return True


def report_perf(child,termination,folder,phase,observers_stopped,timeout=20):
    """Accept requested SIGINT only after actual bounded perf parsing succeeds."""
    if not prepare_perf_report(child,termination,folder,phase,observers_stopped):return
    try:
        with (folder/'cpu-perf-report.txt').open('w') as out:
            # Only self-cycle rows are consumed; resolving every recorded call
            # chain and demangling huge Torch symbols wastes capture time.
            report=subprocess.run(['perf','report','--stdio','--no-children',
                '--call-graph','none','--no-demangle','--sort','dso,symbol',
                '-i',str(folder/'cpu-perf.data')],stdout=out,stderr=subprocess.STDOUT,
                timeout=timeout,env=dict(os.environ,DEBUGINFOD_URLS=''))
        phase['perf_report_returncode']=report.returncode
        if report.returncode:phase['errors'].append('perf report exit '+str(report.returncode))
        else:phase['perf_record_accepted']=True
    except Exception as error:phase['errors'].append('perf report: '+str(error))
    phase['perf_report_status']='complete' if phase['perf_record_accepted'] else 'failed'


def decode(base,model,text,timeout,barrier):
    payload={'model':model,'messages':[{'role':'user','content':text}],
             'chat_template_kwargs':{'thinking':False},'max_tokens':128,'ignore_eos':True,
             'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    barrier.wait(timeout=5);start=time.monotonic();first=None;usage=None
    with urllib.request.urlopen(req,timeout=timeout) as response:
        for line in response:
            line=line.decode().strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':break
            event=json.loads(data)
            if event.get('error'):raise RuntimeError(str(event['error']))
            usage=event.get('usage') or usage
            for choice in event.get('choices',[]):
                delta=choice.get('delta') or {}
                if first is None and (delta.get('content') or delta.get('reasoning_content')):first=time.monotonic()
    if not usage or usage.get('completion_tokens')!=128:raise ValueError('incomplete decode diagnostic request: '+str(usage))
    return {'usage':usage,'ttft_s':first-start if first else None,'elapsed_s':time.monotonic()-start}


def capture(output,base,container,helpers,budget=240):
    output.mkdir(parents=True,exist_ok=False)
    support=module('diagnostic_support',helpers/'experiment-diagnostics.py')
    cold=module('prefill_client',helpers/'prefill-screen.py')
    result={'classification':'unscored diagnostic; never a benchmark score','status':'running','phases':[],
            'errors':[],'capture_iterations':12,'safe_for_scored_work':False}
    start=time.monotonic();deadline=start+budget;active=False;requests_uncertain=False;observers_uncertain=False
    pending_reports=[]
    def save():
        temp=output/'diagnostics.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output/'diagnostics.json')
    def remaining(cap=60):
        value=min(cap,deadline-time.monotonic()-CLEANUP_RESERVE)
        if value<=0:raise TimeoutError('diagnostic work budget exhausted; preserving cleanup time')
        return value
    save()
    try:
        model=support.request(base,'/v1/models',timeout=remaining(5))['data'][0]['id']
        paths=support.request(base,'/openapi.json',timeout=remaining(5)).get('paths',{})
        if not {'/start_profile','/stop_profile'}.issubset(paths):raise RuntimeError('profiler unavailable at this model boot')
        pids=support.workers(container)
        if len(pids)!=4:raise RuntimeError('expected exactly four TP worker PIDs: '+str(pids))
        controls=support.workers(container,controls=True)
        if not controls:result['errors'].append('engine/API CPU control PIDs were not found; worker-only perf sample')
        pids=sorted(set(pids+controls))
        result['cpu_sample_pids']=pids
        for label,count,long in [('decode-c8',8,False),('prefill8k-c1',1,True),('prefill8k-c4',4,True)]:
            phase={'label':label,'concurrency':count,'requests':[],'errors':[],'start_monotonic':time.monotonic()}
            folder=output/label;folder.mkdir();sampler=Sampler(folder/'host-counters.jsonl',pids)
            perf=None;perf_log=None;perf_stop={}
            # Calibration is deliberately before capture; actual completion
            # tokenization remains part of measured application work.
            prompts=[cold.fresh_prompt(8192,base,model,remaining(5),deadline-CLEANUP_RESERVE)[0] for _ in range(count)] if long else [
                secrets.token_hex(128)+'\nExplain the design of a robust job queue in detail.' for _ in range(count)]
            prefix='pf'+secrets.token_hex(4)+'-'+label;phase['profile_prefix']=prefix
            before,err=cold.snapshot(base,folder/'metrics-session-before.prom',remaining(5))
            if err:phase['errors'].append(err)
            try:
                observers_uncertain=True
                sampler.start();perf_log=(folder/'cpu-perf-record.log').open('w')
                try:
                    perf=subprocess.Popen(['perf','record','-F','99','-e','cycles','--call-graph','fp',
                        '-p',','.join(map(str,pids)),'-o',str(folder/'cpu-perf.data'),'--','sleep','100'],
                        stdout=perf_log,stderr=subprocess.STDOUT,start_new_session=True)
                    phase['owned_perf_pid']=perf.pid;phase['owned_perf_pgid']=perf.pid
                except OSError as error:phase['errors'].append('perf: '+str(error))
                active=True  # A failed/timeout start may still have started workers.
                support.request(base,'/start_profile',{'profile_prefix':prefix,'delay_iterations':0,'max_iterations':12},timeout=remaining(10))
                phase['profiler_start_ack']=True
                if long:
                    # Boot policy has two warmup model iterations. Tiny fresh
                    # one-token requests advance them before the target input.
                    phase['profiler_warmup_requests']=[]
                    for _ in range(2):
                        requests_uncertain=True
                        warm=cold.stream_one(base,model,secrets.token_hex(128)+'\nReply yes.',remaining(),deadline=deadline-CLEANUP_RESERVE)
                        requests_uncertain=False
                        phase['profiler_warmup_requests'].append(warm)
                        if warm['cached_tokens']!=0:raise ValueError('cached profiler warmup unexpectedly reported')
                before,err=cold.snapshot(base,folder/'metrics-before.prom',remaining(5))
                if err:phase['errors'].append(err)
                barrier=threading.Barrier(count);requests_uncertain=True
                phase['request_timeout_s']=remaining()
                phase['work_deadline_monotonic']=deadline-CLEANUP_RESERVE
                phase['request_errors']=[]
                with ThreadPoolExecutor(max_workers=count) as pool:
                    if long:
                        futures=[pool.submit(cold.stream_one,base,model,text,phase['request_timeout_s'],barrier,deadline-CLEANUP_RESERVE) for text in prompts]
                    else:futures=[pool.submit(decode,base,model,text,phase['request_timeout_s'],barrier) for text in prompts]
                    for index,future in enumerate(futures):
                        try:phase['requests'].append(future.result())
                        except Exception as error:
                            message=f'request {index}: {error}'
                            phase['request_errors'].append(message);phase['errors'].append(message)
                requests_uncertain=bool(phase['request_errors'])
                if long and any(row['cached_tokens']!=0 for row in phase['requests']):
                    raise ValueError('profile input was cached; diagnostic rejected, no expensive implicit retry')
            except InterruptedError:raise
            except Exception as error:phase['errors'].append(str(error))
            finally:
                if active:
                    try:support.request(base,'/stop_profile',{},timeout=STOP_ACK_TIMEOUT);active=False;phase['profiler_stop_ack']=True
                    except Exception as error:phase['errors'].append('stop_profile: '+str(error))
                observer_error=False
                try:perf_stop=stop_perf(perf)
                except Exception as error:
                    observer_error=True;phase['errors'].append('perf cleanup: '+str(error))
                finally:
                    sampler.stop()
                    if perf_log:perf_log.close()
                observers_uncertain=(observer_error or sampler.thread.is_alive() or (perf is not None and perf.poll() is None))
                phase['observers_stopped']=not observers_uncertain
                if prepare_perf_report(perf,perf_stop,folder,phase,not observers_uncertain):
                    pending_reports.append((perf,perf_stop,folder,phase,True))
                after,err=cold.snapshot(base,folder/'metrics-after.prom',min(5,max(.1,deadline-time.monotonic())))
                phase['metrics']=cold.metric_delta(before,after)
                if err:phase['errors'].append(err)
                phase['end_monotonic']=time.monotonic();result['phases'].append(phase);save()
            if active or requests_uncertain or observers_uncertain:break
    except Exception as error:result['errors'].append(str(error))
    finally:
        if active:
            try:support.request(base,'/stop_profile',{},timeout=STOP_ACK_TIMEOUT);active=False
            except Exception as error:result['errors'].append('cleanup stop_profile: '+str(error))
        result['safe_for_scored_work']=not active and not requests_uncertain and not observers_uncertain
        result['capture_elapsed_s']=time.monotonic()-start
        save()
        # Parsing is CPU-only and can be slow. It must not consume deadlines
        # belonging to later prefill requests or run while GPU stop is uncertain.
        if result['safe_for_scored_work']:
            for report in pending_reports:
                available=deadline-time.monotonic()
                if available<=0:
                    report[3]['perf_report_status']='deferred_budget'
                    report[3]['errors'].append('perf report deferred: total diagnostic budget exhausted')
                else:report_perf(*report,timeout=min(20,available))
        else:
            for _,_,_,phase,_ in pending_reports:phase['perf_report_status']='deferred_unsafe_capture'
        result['elapsed_s']=time.monotonic()-start
        result['status']='completed' if len(result['phases'])==3 and not result['errors'] and not any(p['errors'] for p in result['phases']) else 'partial'
        save()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('--url',default='http://127.0.0.1:8001')
    ap.add_argument('--container',default='dsv41-1017');ap.add_argument('--helpers-dir',type=Path,default=Path(__file__).parent)
    ap.add_argument('--budget-seconds',type=float,default=240)
    args=ap.parse_args()
    if not 90<=args.budget_seconds<=240:ap.error('budget must be90..240 seconds')
    def interrupted(signum,frame):raise InterruptedError('diagnostic interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    result=capture(args.output,args.url.rstrip('/'),args.container,args.helpers_dir,args.budget_seconds)
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='completed' else (1 if result['safe_for_scored_work'] else 3)


if __name__=='__main__':raise SystemExit(main())
