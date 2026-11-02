import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from fastapi import HTTPException
from backend.features import extract_features, STRUCTURAL_FEATURES
from backend.train import load_dataset, split_groups
from backend.main import AnalyzeRequest, analyze_code

class PipelineChecks(unittest.TestCase):
    def test_comment_ratio(self):
        f=extract_features('# heading\nx = 1  # inline\nprint(x)\n')
        self.assertEqual(f['comment_to_code_ratio'], 1.0)
    def test_ast_and_feature_schema(self):
        f=extract_features('def f(xs):\n    for x in xs:\n        if x: return x\n')
        self.assertEqual(set(f),set(STRUCTURAL_FEATURES))
        self.assertEqual((f['function_count'],f['loop_count'],f['conditional_count'],f['max_nesting_depth']),(1,1,1,3))
    def test_group_separation(self):
        a,b,c=split_groups(load_dataset())
        self.assertFalse(set(a.group)&set(b.group) or set(a.group)&set(c.group) or set(b.group)&set(c.group))
        self.assertEqual((len(a),len(b),len(c)),(96,32,32))
    def test_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.csv'
            pd.DataFrame([{'code':'x=1','label':'human'},{'code':'x=1','label':'ai'}]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'conflicting'):
                load_dataset(p)
    def test_syntax_validation(self):
        with self.assertRaises(HTTPException) as ctx:
            analyze_code(AnalyzeRequest(code='def broken(:\n    return 100'))
        self.assertEqual(ctx.exception.status_code,422)
    def test_language_validation(self):
        with self.assertRaises(HTTPException):
            analyze_code(AnalyzeRequest(code='def f():\n    return 100',language='java'))
    def test_saved_pipeline(self):
        r=analyze_code(AnalyzeRequest(code='def f(xs):\n    return [x for x in xs if x > 0]'))
        self.assertIn(r['prediction'],['human','ai'])
        self.assertTrue(0<=r['ai_probability']<=1)
    def test_submitted_source_not_executed(self):
        r=analyze_code(AnalyzeRequest(code='raise RuntimeError("This must never execute")'))
        self.assertIn(r['prediction'],['human','ai'])

if __name__=='__main__': unittest.main()
