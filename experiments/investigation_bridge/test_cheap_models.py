"""Transport failures cannot become apparent small-model successes."""
import json
from pathlib import Path
import tempfile
import unittest

import models


class CheapDecodeTests(unittest.TestCase):
    def decode(self, provider, raw, code=0):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'stdout.log').write_text(json.dumps(raw))
            receipt = {'provider':provider,'exit_code':code,'error':None,'response':None}
            models.decode(receipt,out)
            return receipt

    def test_hosted_aliases_preserve_resolved_identity_and_cost(self):
        for provider in ('sonnet','haiku'):
            result = self.decode(provider, {'subtype':'success','is_error':False,
                'modelUsage':{'resolved-model':{'outputTokens':12}},
                'total_cost_usd':0.01,'structured_output':{'source':'code'}})
            self.assertEqual(result['response'],{'source':'code'})
            self.assertEqual(result['usage'],{'resolved-model':{'outputTokens':12}})
            self.assertEqual(result['estimated_cost_usd'],0.01)

    def test_local_truncation_is_not_a_valid_candidate(self):
        result = self.decode('llama3.2:1b', {'done':True,'done_reason':'length',
            'message':{'content':'{"source":"truncated"}'}})
        self.assertEqual(result['error'],'local_incomplete_response')
        self.assertIsNone(result['response'])

    def test_local_complete_response_retains_token_counts(self):
        result = self.decode('llama3.2:1b', {'done':True,'done_reason':'stop',
            'prompt_eval_count':123,'eval_count':45,
            'message':{'content':'{"source":"code"}'}})
        self.assertEqual(result['response'],{'source':'code'})
        self.assertEqual(result['usage']['prompt_eval_count'],123)

    def test_local_nonzero_exit_cannot_pass(self):
        result = self.decode('llama3.2:1b', {'done':True,'message':{'content':'{}'}},code=28)
        self.assertEqual(result['error'],'local_incomplete_response')
