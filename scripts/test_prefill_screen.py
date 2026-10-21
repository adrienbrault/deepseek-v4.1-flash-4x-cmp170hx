import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import unittest

spec=importlib.util.spec_from_file_location('prefill',Path(__file__).with_name('prefill-screen.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class Handler(BaseHTTPRequestHandler):
    active=0;peak=0;calls=[];cached_remaining=0;omit_cache=False;lock=threading.Lock()
    def log_message(self,*args):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path=='/tokenize':
            value={'count':len(body['prompt'].split())+160}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(value).encode());return
        assert self.path=='/v1/chat/completions'
        assert body['max_tokens']==1 and body['chat_template_kwargs']=={'thinking':False}
        with self.lock:
            type(self).active+=1;type(self).peak=max(self.peak,self.active)
            type(self).calls.append(body)
            cached=16 if self.cached_remaining else 0
            if self.cached_remaining:type(self).cached_remaining-=1
        try:
            time.sleep(.025)
            usage={'prompt_tokens':len(body['messages'][0]['content'].split())+180,'completion_tokens':1}
            if not self.omit_cache:usage['prompt_tokens_details']={'cached_tokens':cached}
            events=[{'choices':[{'delta':{'role':'assistant','content':''}}]},
                    {'choices':[{'delta':{'content':'Yes'},'finish_reason':None}]},
                    {'choices':[{'delta':{},'finish_reason':'length'}]}, {'choices':[],'usage':usage}]
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            for event in events:self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
        finally:
            with self.lock:type(self).active-=1

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.worker=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.worker.start()
        cls.base='http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.worker.join()
    def setUp(self):Handler.calls=[];Handler.peak=0;Handler.cached_remaining=0;Handler.omit_cache=False
    def test_concurrency_exact_usage_and_fresh_prefixes(self):
        result=p.run_batch(self.base,'model',2048,4,3)
        self.assertEqual(result['status'],'COLD');self.assertEqual(len(Handler.calls),4)
        self.assertGreaterEqual(Handler.peak,3)
        rows=result['attempts'][0]['requests']
        self.assertEqual(len({r['prompt_sha256'] for r in rows}),4)
        self.assertTrue(all(r['cached_tokens']==0 and r['completion_tokens']==1 and not r['first_event_terminal_only'] for r in rows))
        self.assertEqual(result['prompt_tokens_total'],sum(r['usage']['prompt_tokens'] for r in rows))
        self.assertTrue(1950<result['actual_prompt_tokens_min']<2150)
    def test_cached_batch_retried_with_new_content(self):
        Handler.cached_remaining=1
        result=p.run_batch(self.base,'model',2048,1,3)
        self.assertEqual([a['cold'] for a in result['attempts']],[False,True])
        self.assertNotEqual(Handler.calls[0]['messages'],Handler.calls[1]['messages'])
    def test_missing_cache_reporting_rejected(self):
        Handler.omit_cache=True
        with self.assertRaisesRegex(ValueError,'cached token count missing'):p.run_batch(self.base,'model',2048,1,3)
    def test_cached_retry_bound(self):
        Handler.cached_remaining=100
        result=p.run_batch(self.base,'model',2048,1,3,retries=1)
        self.assertEqual(result['status'],'REJECTED_CACHED');self.assertEqual(len(Handler.calls),2)
    def test_metrics_real_prefill_and_histogram_deltas(self):
        before=p.parse_metrics('vllm:request_prefill_time_seconds_sum{model="a"} 3\nvllm:request_prefill_time_seconds_count{model="a"} 2\nvllm:request_prefill_time_seconds_bucket{le="5"} 2\n')
        after=p.parse_metrics('vllm:request_prefill_time_seconds_sum{model="a"} 7\nvllm:request_prefill_time_seconds_count{model="a"} 4\nvllm:request_prefill_time_seconds_bucket{le="5"} 4\n')
        delta=p.metric_delta(before,after)
        self.assertEqual(delta['server_request_prefill_mean_s'],2)
        self.assertEqual(delta['server_request_prefill_observation_count'],2)
        self.assertEqual(len(delta['series']),3)
        self.assertIsNone(p.metric_delta({}, {})['server_request_prefill_mean_s'])
    def test_expired_budget_before_request(self):
        with self.assertRaises(TimeoutError):p.run_batch(self.base,'model',2048,1,3,deadline=time.monotonic()-1)
        self.assertEqual(Handler.calls,[])

if __name__=='__main__':unittest.main()
