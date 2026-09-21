"""Mechanism checks use scripted responses; these are not model-quality results."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
import sandbox
import workload


class RunnerTests(unittest.TestCase):
    def test_plan_freezes_identical_tasks_across_arms(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'trial'; manifest=run.plan(path,19,'hard',1)
            self.assertEqual(len({r['task_sha256'] for r in manifest['trials']}),1)
            self.assertEqual(set(r['arm'] for r in manifest['trials']),{'opus-solo','opus-team','astra-solo'})

    def test_unknown_paid_call_is_not_retried(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root); (path/'calls/1').mkdir(parents=True)
            with patch('models.complete') as call:
                with self.assertRaises(RuntimeError):run.ask(path,1,'opus','prompt',{},None)
                call.assert_not_called()

    def test_investigators_cannot_edit(self):
        response=dict(action='edit',hypothesis='',prediction='',source='pass',probes_json='[]')
        with self.assertRaises(ValueError):run.validate(response,investigator=True)

    def test_initial_prompts_hide_final_and_reference(self):
        task=workload.task(81,'hard'); task['final']=['SECRET_FINAL_SENTINEL']
        text=run.prompt(task,task['starter'],{'passed':False},'investigator-1',[])
        self.assertNotIn('SECRET_FINAL_SENTINEL',text)
        self.assertNotIn('correct_source',text)

    def test_held_out_failure_cannot_become_success(self):
        task=workload.task(42,'hard')
        with patch('sandbox.evaluate',return_value={'outputs':[{'point':['0','0']}]*48,'error':None}):
            result=run.grade('irrelevant',task['final'],sandbox.IMAGE,None)
        self.assertFalse(result['passed'])

    def test_missing_or_truncated_outputs_fail_closed(self):
        task=workload.task(42,'hard')
        for outputs in (None, [], [{}], 'invalid'):
            with self.subTest(outputs=outputs), patch('sandbox.evaluate',return_value={'outputs':outputs,'error':None}):
                result=run.grade('irrelevant',task['final'],sandbox.IMAGE,None)
                self.assertFalse(result['passed'])
                self.assertEqual(result['failed'],len(task['final']))

    def test_bad_optional_probe_does_not_prevent_seal(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'trial'; manifest=run.plan(path,19,'hard',1)
            entry=next(e for e in manifest['trials'] if e['arm']=='opus-solo')
            response={'error':None,'response':dict(action='edit',hypothesis='',prediction='',source=workload.correct_source(),probes_json='[17]')}
            def checked(source, requests, *args):
                if requests==[17]: raise TypeError('invalid request')
                return {'passed':source==workload.correct_source(),'failures':[]}
            with patch('run.ask',return_value=response) as ask, patch('run.grade',side_effect=checked):
                result=run.trial(path,entry,manifest)
            self.assertEqual(ask.call_count,1)
            self.assertTrue(result['accepted'])

    def test_failed_adviser_does_not_cancel_integrator(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'trial'; manifest=run.plan(path,19,'hard',1)
            entry=next(e for e in manifest['trials'] if e['arm']=='opus-team')
            def scripted(directory,number,*args):
                if number==1:return {'error':'timeout_usage_unknown','response':None}
                if number==2:return {'error':None,'response':dict(action='hypothesis',hypothesis='test',prediction='',source='',probes_json='[]')}
                return {'error':None,'response':dict(action='edit',hypothesis='repair',prediction='',source=workload.correct_source(),probes_json='[]')}
            def checked(source,*args):return {'passed':source==workload.correct_source(),'failures':[]}
            with patch('run.ask',side_effect=scripted),patch('run.grade',side_effect=checked):
                result=run.trial(path,entry,manifest)
            self.assertEqual(len(result['calls']),3)
            self.assertTrue(result['accepted'])


@unittest.skipUnless(os.environ.get('BRIDGE_DOCKER_TESTS')=='1','explicit disposable Docker checks')
class DockerTests(unittest.TestCase):
    def test_known_good_and_mutants(self):
        task=workload.task(23,'hard')
        self.assertTrue(run.grade(workload.correct_source(),task['final'],sandbox.IMAGE,None)['passed'])
        for name,source in workload.mutants().items():
            with self.subTest(mutant=name):
                self.assertFalse(run.grade(source,task['development'],sandbox.IMAGE,None)['passed'])

    def test_no_host_mounts_or_external_network_and_read_only_root(self):
        source='''import os,socket
def evaluate(request):
    result={"uid":os.getuid(),"credential":bool(os.environ.get("OPENAI_API_KEY"))}
    try:
        socket.create_connection(("1.1.1.1",443),timeout=1).close(); result["network"]=True
    except OSError: result["network"]=False
    try:
        open("/escape-proof","w").write("no"); result["root_write"]=True
    except OSError: result["root_write"]=False
    result["host_path"]=os.path.exists(request["sentinel"])
    return result
'''
        with tempfile.NamedTemporaryFile() as sentinel:
            result=sandbox.evaluate(source,[{'sentinel':sentinel.name}])
        self.assertEqual(result['outputs'],[{'uid':65534,'credential':False,'network':False,'root_write':False,'host_path':False}])

    def test_infinite_loop_is_stopped(self):
        result=sandbox.evaluate('def evaluate(r):\n while True: pass',[{}],timeout=1)
        self.assertEqual(result['error'],'timeout')
        self.assertTrue(result['cleanup_confirmed'])


if __name__=='__main__':unittest.main()
