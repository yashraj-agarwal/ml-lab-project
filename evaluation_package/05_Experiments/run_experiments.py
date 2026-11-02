"""Reproduce controlled experiments without replacing the deployed model."""
import sys, json, hashlib, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
import sklearn, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from backend.features import extract_features, STRUCTURAL_FEATURES
from backend.train import load_dataset, split_groups, metric_row
from backend.predict import analyze

OUT = Path(__file__).resolve().parent
base = 'def select(values):\n    result = []\n    for value in values:\n        if value % 2 == 0:\n            result.append(value)\n    return result\n'
variants = {
 'original': base,
 'variable naming': base.replace('select','filterEvenNumbers').replace('values','inputNumbers').replace('result','filteredNumbers').replace('value','currentNumber'),
 'comment density': '# Select even values\n' + base.replace('    result', '    # Collect matching elements\n    result', 1),
 'formatting': base.replace('    ', '  ').replace(' = ', '='),
 'structure': 'def select(values):\n    return [value for value in values if value % 2 == 0]\n',
 'unseen problem': 'def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n',
}
rows=[]
for name,code in variants.items():
 r=analyze(code)
 rows.append({'experiment':name,'configuration':code,'prediction':r['prediction'],'ai_probability':r['ai_probability'],'features':json.dumps(r['features'])})
df=load_dataset()
dataset_summary={'source_rows':df.attrs['source_rows'],'removed_rows':df.attrs['removed_rows'],'unique_rows':len(df),'class_counts':df.label.value_counts().to_dict(),'group_count':int(df.group.nunique())}
df=pd.concat([df,pd.DataFrame([extract_features(c) for c in df.code])],axis=1)
tr,va,te=split_groups(df)
assert not (set(tr.group)&set(va.group) or set(tr.group)&set(te.group) or set(va.group)&set(te.group))
stylo=['comment_to_code_ratio','average_line_length','average_identifier_length','snake_case_ratio','camel_case_ratio','blank_line_ratio']
struct=['function_count','loop_count','conditional_count','class_count','exception_count','ast_node_count','max_nesting_depth','docstring_count']
for group in ['all','without lexical','without stylometric','without structural']:
 for c in ([0.01,1,100] if group=='all' else [1]):
  numeric=[f for f in STRUCTURAL_FEATURES if not (group=='without stylometric' and f in stylo) and not(group=='without structural' and f in struct)]
  transformers=[('numeric',StandardScaler(),numeric)]
  if group!='without lexical': transformers.insert(0,('lexical',TfidfVectorizer(max_features=1200,ngram_range=(1,2),sublinear_tf=True,token_pattern=r'(?u)\b\w+\b'),'code'))
  pipe=Pipeline([('features',ColumnTransformer(transformers)),('classifier',LogisticRegression(C=c,max_iter=1500,class_weight='balanced',random_state=42))])
  pipe.fit(tr,tr.label)
  for threshold in ([.4,.5,.6,.7] if group=='all' and c==1 else [.5]):
   p=pipe.predict_proba(va)[:,list(pipe.classes_).index('ai')]
   pred=['ai' if x>=threshold else 'human' for x in p]
   demo=pipe.predict_proba(pd.DataFrame([{'code':base,**extract_features(base)}]))[0,list(pipe.classes_).index('ai')]
   rows.append({'experiment':'configuration','configuration':f'{group}; C={c}; threshold={threshold}','prediction':'ai' if demo>=threshold else 'human','ai_probability':float(demo),**metric_row(va.label,pred),'roc_auc':roc_auc_score(va.label=='ai',p)})
pd.DataFrame(rows).to_csv(OUT/'experiment_results.csv',index=False)
artifact=joblib.load(ROOT/'models/codetrace.joblib')
inventory={'python':platform.python_version(),'sklearn':sklearn.__version__,'joblib':joblib.__version__,'model_sha256':hashlib.sha256((ROOT/'models/codetrace.joblib').read_bytes()).hexdigest(),'dataset':dataset_summary,'selected_estimator_parameters':artifact['pipeline'].named_steps['classifier'].get_params(),'tfidf_parameters':{k:str(v) for k,v in artifact['pipeline'].named_steps['features'].named_transformers_['lexical'].get_params().items()},'lexical_vocabulary_size':len(artifact['pipeline'].named_steps['features'].named_transformers_['lexical'].vocabulary_),'artifact_metadata':artifact.get('metadata'),'split_groups':{'train':sorted(tr.group.unique()),'validation':sorted(va.group.unique()),'test':sorted(te.group.unique())},'split_class_counts':{name:x.label.value_counts().to_dict() for name,x in [('train',tr),('validation',va),('test',te)]},'saved_metrics':json.loads((ROOT/'models/metrics.json').read_text())}
(OUT/'technical_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).drop(columns=['features'],errors='ignore').to_string(index=False))
