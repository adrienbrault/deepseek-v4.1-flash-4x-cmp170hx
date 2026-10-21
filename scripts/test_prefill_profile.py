import importlib.util
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
import signal
import sys
import time
from pathlib import Path
import tempfile
import subprocess
import threading
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).parent
if not (ROOT/'prefill-profile.py').exists():
    ROOT=ROOT.parents[1]
if not (ROOT/'prefill-profile.py').exists():
    ROOT=ROOT.parents[1]

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

capture=load('capture','prefill-profile.py');summary=load('summary','summarize-prefill-profile.py')
cold=load('cold','prefill-screen.py');support=load('support','experiment-diagnostics.py');core=load('core','summarize-profiler.py')

class CaptureTests(unittest.TestCase):
    def run_case(self,fail_stop=False,fail_request=False,cached=False,fail_reap=False,fail_stop_once=False,expected_sigint=False,late_final_cleanup=False,slow_reports=False,fail_request_once=False,report_budget_exhausted=False):
        events=[]
        failure_lock=threading.Lock();failures={'left':1 if fail_request_once else 0}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                body=({'data':[{'id':'model'}]} if self.path=='/v1/models' else
                      {'paths':{'/start_profile':{},'/stop_profile':{}}})
                self.send_response(200);self.end_headers()
                self.wfile.write((b'vllm:request_prefill_time_seconds_sum 1\nvllm:request_prefill_time_seconds_count 1\n') if self.path=='/metrics' else json.dumps(body).encode())
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));events.append((self.path,body))
                if self.path=='/tokenize':result={'count':len(body['prompt'].split())+160}
                elif self.path=='/v1/chat/completions':
                    with failure_lock:
                        reject=fail_request or failures['left']>0
                        if failures['left']>0:failures['left']-=1
                    if reject:self.send_response(500);self.end_headers();return
                    self.send_response(200);self.end_headers()
                    usage={'prompt_tokens':len(body['messages'][0]['content'].split())+180,'completion_tokens':body['max_tokens'],'prompt_tokens_details':{'cached_tokens':16 if cached and body['max_tokens']==1 and len(body['messages'][0]['content'])>10000 else 0}}
                    for item in ({'choices':[{'delta':{'content':'yes'}}]},{'choices':[],'usage':usage}):
                        self.wfile.write(('data: '+json.dumps(item)+'\n\n').encode())
                    self.wfile.write(b'data: [DONE]\n\n');return
                else:result={}
                self.send_response(500 if self.path=='/stop_profile' and (fail_stop or (fail_stop_once and sum(path=='/stop_profile' for path,_ in events)==1)) else 200);self.end_headers();self.wfile.write(json.dumps(result).encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'capture';child=Mock(pid=999999,returncode=-signal.SIGINT if expected_sigint else 0)
                perf_state={'running':True}
                if expected_sigint:
                    child.poll.side_effect=lambda:None if perf_state['running'] else -signal.SIGINT
                    def reap(timeout):perf_state['running']=False;return -signal.SIGINT
                    child.wait.side_effect=reap
                if fail_reap:child.wait.side_effect=subprocess.TimeoutExpired('perf',3)
                def modules(name,path):return support if name=='diagnostic_support' else cold
                def start_perf(command,**kwargs):
                    perf_state['running']=True
                    Path(command[command.index('-o')+1]).write_bytes(b'fixture perf data')
                    return child
                original_request=support.request
                original_snapshot=cold.snapshot;original_clock=time.monotonic
                cleanup_clock={'after':0,'offset':0}
                def checked_snapshot(base,path,timeout):
                    value=original_snapshot(base,path,timeout)
                    if Path(path).name=='metrics-after.prom':
                        cleanup_clock['after']+=1
                        if late_final_cleanup and cleanup_clock['after']==3:cleanup_clock['offset']=100
                        if report_budget_exhausted and cleanup_clock['after']==3:cleanup_clock['offset']=300
                    return value
                report_calls=[]
                def parsed_report(*args,**kwargs):
                    self.assertEqual(kwargs['timeout'],20)
                    self.assertEqual(kwargs['env']['DEBUGINFOD_URLS'],'')
                    self.assertIn('--no-demangle',args[0])
                    self.assertEqual(args[0][args[0].index('--call-graph')+1],'none')
                    report_calls.append(sum(path=='/stop_profile' for path,_ in events))
                    if slow_reports:cleanup_clock['offset']+=50
                    return Mock(returncode=0)
                def checked_request(base,path,payload=None,timeout=45):
                    if path=='/stop_profile':self.assertEqual(timeout,60)
                    return original_request(base,path,payload,timeout)
                with patch.object(capture,'module',side_effect=modules),patch.object(support,'workers',return_value=[11,12,13,14]), \
                     patch.object(capture.subprocess,'Popen',side_effect=start_perf),patch.object(support,'request',side_effect=checked_request),patch.object(capture.os,'killpg'), \
                     patch.object(capture.subprocess,'run',side_effect=parsed_report), \
                     patch.object(capture,'proc_sample',return_value={'monotonic':1,'pids':{}}), \
                     patch.object(cold,'snapshot',side_effect=checked_snapshot), \
                     patch.object(capture.time,'monotonic',side_effect=lambda:original_clock()+cleanup_clock['offset']):
                    result=capture.capture(out,f'http://127.0.0.1:{server.server_port}','container',ROOT)
                saved=json.loads((out/'diagnostics.json').read_text());self.assertEqual(result,saved)
                child.wait.assert_called()
                if slow_reports:self.assertEqual(report_calls,[3,3,3])
                return events,result
        finally:server.shutdown();server.server_close();worker.join()
    def test_three_distinct_phases_and_confirmed_stops(self):
        events,result=self.run_case()
        self.assertEqual(result['status'],'completed');self.assertTrue(result['safe_for_scored_work'])
        starts=[v for path,v in events if path=='/start_profile']
        self.assertEqual(len(starts),3);self.assertTrue(all(v['max_iterations']==12 for v in starts))
        self.assertEqual(len({v['profile_prefix'] for v in starts}),3)
        calls=[v for path,v in events if path=='/v1/chat/completions']
        self.assertEqual([v['max_tokens'] for v in calls].count(128),8);self.assertEqual([v['max_tokens'] for v in calls].count(1),9)
        self.assertEqual([len(p.get('profiler_warmup_requests',[])) for p in result['phases']],[0,2,2])
        self.assertEqual(sum(path=='/stop_profile' for path,_ in events),3)
        self.assertTrue(all(p['profiler_stop_ack'] for p in result['phases']))
    def test_slow_cpu_reporting_does_not_consume_later_prefill_deadlines(self):
        _,result=self.run_case(slow_reports=True)
        self.assertEqual(len(result['phases']),3)
        self.assertEqual(result['status'],'completed')
        self.assertTrue(result['safe_for_scored_work'])

    def test_exhausted_total_budget_defers_cpu_parsing_after_safe_cleanup(self):
        _,result=self.run_case(report_budget_exhausted=True)
        self.assertEqual(len(result['phases']),3)
        self.assertTrue(result['safe_for_scored_work'])
        self.assertEqual(result['status'],'partial')
        self.assertTrue(all(p['perf_report_status']=='deferred_budget' for p in result['phases']))

    def test_expected_sigint_is_accepted_in_all_three_capture_phases(self):
        _,result=self.run_case(expected_sigint=True)
        self.assertEqual(result['status'],'completed');self.assertTrue(result['safe_for_scored_work'])
        self.assertTrue(all(p['perf_returncode']==-signal.SIGINT for p in result['phases']))
        self.assertTrue(all(p['perf_record_accepted'] and p['perf_report_returncode']==0 for p in result['phases']))
    def test_successful_final_cleanup_may_consume_reserved_time(self):
        _,result=self.run_case(late_final_cleanup=True)
        self.assertEqual(result['status'],'completed');self.assertTrue(result['safe_for_scored_work'])
        self.assertGreater(result['elapsed_s'],100)
    def test_uncertain_stop_prevents_next_phase_and_scored_work(self):
        events,result=self.run_case(fail_stop=True)
        self.assertFalse(result['safe_for_scored_work']);self.assertEqual(len(result['phases']),1)
        self.assertEqual(sum(path=='/stop_profile' for path,_ in events),2)
    def test_request_failure_still_stops_profiler_and_declines_safe_claim(self):
        events,result=self.run_case(fail_request=True)
        self.assertFalse(result['safe_for_scored_work']);self.assertEqual(len(result['phases']),1)
        self.assertEqual(sum(path=='/stop_profile' for path,_ in events),1)
    def test_failed_request_retains_other_successful_stream_receipts(self):
        _,result=self.run_case(fail_request_once=True)
        self.assertFalse(result['safe_for_scored_work'])
        self.assertEqual(len(result['phases']),1)
        self.assertEqual(len(result['phases'][0]['requests']),7)
        self.assertEqual(len(result['phases'][0]['request_errors']),1)

    def test_unreaped_perf_declines_score_overlap(self):
        _,result=self.run_case(fail_reap=True)
        self.assertFalse(result['safe_for_scored_work']);self.assertEqual(result['status'],'partial')
    def test_cached_prefill_is_rejected(self):
        _,result=self.run_case(cached=True)
        self.assertEqual(result['status'],'partial');self.assertTrue(result['safe_for_scored_work'])
        self.assertIn('cached',result['phases'][1]['errors'][0])
    def test_failed_stop_then_cleanup_ack_still_prevents_next_phase(self):
        events,result=self.run_case(fail_stop_once=True)
        self.assertEqual(len(result['phases']),1)
        self.assertEqual(sum(path=='/start_profile' for path,_ in events),1)
        self.assertEqual(sum(path=='/stop_profile' for path,_ in events),2)
        self.assertTrue(result['safe_for_scored_work']);self.assertEqual(result['status'],'partial')

class PerfTerminationTests(unittest.TestCase):
    def fixture(self,root):
        script=root/'perf'
        script.write_text('#!'+sys.executable+'\n'+"""import os,pathlib,signal,sys,time
if sys.argv[1]=='report':
 print('sampled CPU fixture report')
 raise SystemExit(int(os.environ.get('FAKE_PERF_REPORT_EXIT','0')))
def interrupted(signum,frame):
 signal.signal(signal.SIGINT,signal.SIG_DFL)
 os.kill(os.getpid(),signal.SIGINT)
signal.signal(signal.SIGINT,interrupted)
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'nonempty simulated perf.data')
pathlib.Path(os.environ['FAKE_PERF_READY']).touch()
while True:signal.pause()
""");script.chmod(0o755)
        return script
    def test_actual_requested_sigint_with_data_and_successful_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=self.fixture(root);ready=root/'ready'
            env=dict(os.environ,FAKE_PERF_READY=str(ready),PATH=str(root)+os.pathsep+os.environ['PATH'])
            child=subprocess.Popen([str(script),'record','-o',str(root/'cpu-perf.data')],env=env,start_new_session=True)
            try:
                deadline=time.monotonic()+5
                while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(ready.exists())
                stopped=capture.stop_perf(child)
                self.assertEqual(child.returncode,-signal.SIGINT)
                self.assertTrue(stopped['sigint_requested']);self.assertFalse(stopped['recorder_force_killed'])
                phase={'errors':[]}
                with patch.dict(os.environ,env):capture.report_perf(child,stopped,root,phase,True)
                self.assertEqual(phase['errors'],[]);self.assertTrue(phase['perf_record_accepted'])
                self.assertEqual(phase['perf_report_returncode'],0)
                self.assertTrue((root/'cpu-perf-report.txt').read_text().strip())
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=5)
    def test_unexpected_or_forced_exit_remains_error(self):
        for returncode,stop in [(-signal.SIGINT,{'sigint_requested':False}),
                               (-signal.SIGKILL,{'sigint_requested':True,'recorder_force_killed':True}),
                               (7,{'sigint_requested':True})]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'cpu-perf.data').write_bytes(b'data');phase={'errors':[]}
                with patch.object(capture.subprocess,'run') as report:
                    capture.report_perf(Mock(returncode=returncode),stop,root,phase,True)
                self.assertFalse(phase['perf_record_accepted']);self.assertTrue(phase['errors']);report.assert_not_called()
    def test_expected_sigint_requires_nonempty_data(self):
        for contents in (None,b''):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);phase={'errors':[]}
                if contents is not None:(root/'cpu-perf.data').write_bytes(contents)
                with patch.object(capture.subprocess,'run') as report:
                    capture.report_perf(Mock(returncode=-signal.SIGINT),{'sigint_requested':True},root,phase,True)
                self.assertFalse(phase['perf_record_accepted']);self.assertTrue(phase['errors']);report.assert_not_called()
    def test_expected_sigint_requires_successful_actual_report_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'cpu-perf.data').write_bytes(b'data');phase={'errors':[]}
            with patch.dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],FAKE_PERF_REPORT_EXIT='7'):
                capture.report_perf(Mock(returncode=-signal.SIGINT),{'sigint_requested':True},root,phase,True)
            self.assertFalse(phase['perf_record_accepted']);self.assertEqual(phase['perf_report_returncode'],7)
            self.assertTrue(phase['errors'])

class SummaryTests(unittest.TestCase):
    def test_enqueue_is_not_callback_and_overlap_is_union(self):
        events=[{'ph':'X','cat':'cuda_runtime','name':'cudaLaunchHostFunc','ts':0,'dur':10},
                {'ph':'X','cat':'kernel','name':'ncclAllReduce','ts':10,'dur':20}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.json';path.write_text(json.dumps({'traceEvents':events}))
            result=summary.explicit_spans(path,core)
            self.assertFalse(result['explicit_rowstore_cpu_spans_present']);self.assertEqual(result['host_callback_enqueue_count'],1)
            self.assertIsNone(result['explicit_rowstore_cpu_span_union_us'])
            events += [{'ph':'X','cat':'user_annotation','name':'row_store_lookup','ts':0,'dur':20},
                       {'ph':'X','cat':'user_annotation','name':'row_store_lookup','ts':5,'dur':20}]
            path.write_text(json.dumps({'traceEvents':events}));result=summary.explicit_spans(path,core)
            self.assertEqual(result['explicit_rowstore_cpu_span_union_us'],25)
            self.assertEqual(result['rowstore_cpu_span_communication_temporal_overlap_us'],15)
    def test_rank_completeness_checks_actual_gpu_kernel_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);capture_dir=root/'capture';capture_dir.mkdir();phase=capture_dir/'prefill8k-c4';phase.mkdir();traces=root/'traces';traces.mkdir()
            (capture_dir/'diagnostics.json').write_text(json.dumps({'status':'completed','safe_for_scored_work':True,'phases':[{'label':'prefill8k-c4','profile_prefix':'pf123'}]}))
            (phase/'host-counters.jsonl').write_text(json.dumps({'monotonic':0,'pids':{}})+'\n')
            for rank in range(4):
                path=traces/f'pf123_tp{rank}_rank{rank}.pt.trace.json'
                path.write_text(json.dumps({'traceEvents':[{'ph':'X','cat':'kernel','name':'matmul','ts':0,'dur':1}]}))
            self.assertEqual(summary.summarize(capture_dir,traces,core)['status'],'complete')
            (traces/'pf123_tp3_rank3.pt.trace.json').write_text(json.dumps({'traceEvents':[]}))
            self.assertEqual(summary.summarize(capture_dir,traces,core)['status'],'partial')
    def test_incomplete_capture_never_claims_complete_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'diagnostics.json').write_text(json.dumps({'status':'partial','phases':[]}))
            self.assertEqual(summary.summarize(root,root,core)['status'],'partial')
    def test_empty_report_is_not_cpu_perf_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.txt';path.write_text('')
            self.assertFalse(summary.perf_evidence(path)['available'])

    def test_host_disk_and_fault_delta(self):
        def row(t,reads,sectors,faults):return {'monotonic':t,'pids':{},'vmstat':f'pgfault {faults}\npgmajfault 1\n','diskstats':f'259 0 nvme0n1 {reads} 0 {sectors} 2 0 0 0 0 0 3 4\n'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'host.jsonl';path.write_text('\n'.join(json.dumps(r) for r in [row(1,10,20,100),row(3,15,40,130)]))
            result=summary.host_delta(path)
            self.assertEqual(result['system_vm_deltas']['pgfault'],30);self.assertEqual(result['disk_deltas']['nvme0n1']['read_bytes'],20*512)
            self.assertEqual(result['elapsed_s'],2)

if __name__=='__main__':unittest.main()
