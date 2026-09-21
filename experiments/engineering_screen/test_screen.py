import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import screen

class ScreenTests(unittest.TestCase):
    def test_missing_and_wrong_output_fail_closed(self):
        qs=[{'rings':[],'rule':'evenodd'}]
        for outputs in (None,[],[{'area':0}],[{'area':True}]):
            with patch.object(screen.sandbox,'evaluate',return_value={'outputs':outputs,'error':None}):
                self.assertFalse(screen.grade('geometry','',qs,None)['passed'])

    def test_prompt_excludes_holdout_and_oracle_source(self):
        p={'contract':'test','development':[],'final':['SECRET']}
        prompt=screen.prompt(p,{'task':'geometry','attempts':[]})
        self.assertNotIn('SECRET',prompt)
        self.assertNotIn('def geometry',prompt)

    def test_frozen_final_inputs_do_not_repeat_development(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'screen';screen.freeze(root)
            for task in json.loads((root/'tasks.json').read_text()).values():
                development={json.dumps(q,sort_keys=True) for q in task['development']}
                self.assertTrue(task['final'])
                self.assertFalse(development & {json.dumps(q,sort_keys=True) for q in task['final']})

    def test_unknown_call_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'geometry-opus'/'call-1').mkdir(parents=True)
            with patch.object(screen.models,'complete') as model:
                with self.assertRaises(RuntimeError):screen.trial(root,{'task':'geometry','provider':'opus'},{'contract':'','development':[],'final':[]},{'calls_per_arm':2,'timeout':150})
                model.assert_not_called()

    def test_final_failure_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            response={'error':None,'response':{'action':'edit','source':'def evaluate(q): return {}'}}
            grades=[{'passed':True,'execution':{'error':None}},{'passed':False,'execution':{'error':None}}]
            with patch.object(screen.models,'complete',return_value=response) as model,patch.object(screen,'grade',side_effect=grades) as grade:
                result=screen.trial(Path(d),{'task':'geometry','provider':'opus'},{'contract':'','development':[],'final':[]},{'calls_per_arm':2,'timeout':150})
            self.assertFalse(result['accepted']);self.assertEqual(model.call_count,1);self.assertEqual(grade.call_count,2)

if __name__=='__main__':unittest.main()
