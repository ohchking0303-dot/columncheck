"""Evaluate locked hand-authored synthetic holdout; not a production benchmark."""
import json,platform,time
from pathlib import Path
import sklearn
from csv_review import ROOT,train_model,suggest,baseline,normalize
start=time.perf_counter()
train=json.loads((ROOT/'data/train.json').read_text())
holdout=json.loads((ROOT/'data/holdout.json').read_text())
assert not ({normalize(x['header']) for x in train}&{normalize(x['header']) for x in holdout})
model=train_model()
rows=[{**x,**suggest(model,x['header']),'rules_only':baseline(x['header'])} for x in holdout]
summary={}
for method in ['candidate','suggestion','rules_only']:
    covered=[x for x in rows if x[method]!='review_required']
    correct=sum(x[method]==x['label'] for x in rows)
    summary[method]={'correct':correct,'total':len(rows),'coverage':len(covered)/len(rows),'accuracy_counting_abstention_as_incorrect':correct/len(rows),'accuracy_when_answered':sum(x[method]==x['label'] for x in covered)/len(covered) if covered else None}
print(json.dumps({'warning':'Tiny authored synthetic holdout with related vocabulary, not independent real-world evidence. No tuning was performed on this holdout.','train_records':len(train),'holdout_records':len(holdout),'normalized_overlap':0,'summary':summary,'predictions':rows,'runtime_seconds':time.perf_counter()-start,'python':platform.python_version(),'sklearn':sklearn.__version__},indent=2))
