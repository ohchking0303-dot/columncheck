"""ColumnCheck: local learned header suggestions plus explicit format checks."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, re, sys, time
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parent
MAX_BYTES=5_000_000
MAX_ROWS=10_000
MAX_COLUMNS=100

def normalize(value):
    value=re.sub(r'([a-z])([A-Z])',r'\1 \2',value)
    return re.sub(r'[^a-z0-9]+',' ',value.lower()).strip()

def train_model():
    records=json.loads((ROOT/'data/train.json').read_text())
    model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5)),LogisticRegression(C=10,max_iter=1000,random_state=0))
    model.fit([normalize(x['header']) for x in records],[x['label'] for x in records])
    return model

def suggest(model, header):
    normalized=normalize(header)
    probabilities=model.predict_proba([normalized])[0]
    order=probabilities.argsort()[::-1]
    score=float(probabilities[order[0]])
    margin=score-float(probabilities[order[1]])
    known_features=model[0].transform([normalized]).nnz
    accepted=known_features>0 and score>=0.45 and margin>=0.15
    return {'candidate':str(model.classes_[order[0]]),'suggestion':str(model.classes_[order[0]]) if accepted else 'review_required','model_score':round(score,4),'score_margin':round(margin,4),'note':'Model scores are uncalibrated, not correctness probabilities.'}

RULES={'email':{'email','mail'},'date':{'date','deadline'},'money':{'amount','total','price','subtotal','balance','fee','cost'},'identifier':{'id','identifier','sku','code','reference','ref'},'quantity':{'qty','quantity','count','units','pieces'},'description':{'description','notes','memo','comment','details','name','summary','title','message'}}
def baseline(header):
    words=set(normalize(header).split())
    labels=[k for k,v in RULES.items() if words&v]
    return labels[0] if len(labels)==1 else 'review_required'

def valid(value,label):
    """Intentionally narrow locale contracts; never infer currency or repair values."""
    if label=='email': return bool(re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+',value))
    if label=='date':
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value): return False
        try: date.fromisoformat(value); return True
        except ValueError: return False
    if label=='money':
        if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?',value): return False
        try: return Decimal(value).is_finite()
        except InvalidOperation: return False
    if label=='quantity': return bool(re.fullmatch(r'\d+',value))
    return None

def review(path,model,schema=None):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES: raise ValueError('CSV exceeds 5 MB limit')
    raw=path.read_bytes()
    source_hash=hashlib.sha256(raw).hexdigest()
    import io
    reader=csv.reader(io.StringIO(raw.decode('utf-8-sig')),strict=True)
    try: headers=next(reader)
    except StopIteration: raise ValueError('CSV is empty')
    if not headers or any(not h.strip() for h in headers): raise ValueError('Headers must be nonempty')
    if len(headers)>MAX_COLUMNS: raise ValueError('CSV exceeds 100 columns')
    if len(set(headers))!=len(headers): raise ValueError('Duplicate headers are not supported')
    if schema is None: schema={}
    if not isinstance(schema,dict): raise ValueError('Schema must be a JSON object')
    if any(h not in headers for h in schema): raise ValueError('Schema contains unknown column names')
    if any(not isinstance(label,str) or label not in {*RULES,'review_required'} for label in schema.values()):
        raise ValueError('Schema contains unsupported types')
    columns=[[] for _ in headers]
    rows=0
    for index,row in enumerate(reader,2):
        if not row: continue
        if len(row)!=len(headers): raise ValueError(f'CSV record {index} has wrong column count')
        rows+=1
        if rows>MAX_ROWS: raise ValueError('CSV exceeds 10000 rows')
        for column,value in zip(columns,row): column.append((index,value.strip()))
    reports=[]
    for header,values in zip(headers,columns):
        prediction=suggest(model,header)
        label=schema.get(header,prediction['suggestion'])
        prediction['effective_label']=label
        prediction['label_source']='user_schema' if header in schema else 'model_suggestion'
        missing=[i for i,v in values if not v]
        invalid=[i for i,v in values if v and valid(v,label) is False]
        checked=sum(bool(v) and valid(v,label) is not None for _,v in values)
        reports.append({'header':header,**prediction,'rules_only_suggestion':baseline(header),'nonempty_values':len(values)-len(missing),'checked_values':checked,'invalid_record_numbers':invalid,'missing_record_numbers':missing,'validator':label if checked else 'no_format_validation'})
    return {'tool':'ColumnCheck prototype','source_sha256':source_hash,'rows':rows,'columns':reports,'privacy':'No network calls; report contains headers and record numbers, not cell contents.','warning':'Suggestions need human review; validity of meaning, identifiers, currency and business rules is not established.'}

def render_html(report):
    # Escape every user-derived string; no external scripts/resources or raw CSV cells.
    content=html.escape(json.dumps(report,indent=2,ensure_ascii=False))
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>ColumnCheck review</title><style>body{max-width:1000px;margin:3rem auto;font:16px system-ui;background:#f5f7fb;color:#16243a}pre{white-space:pre-wrap;background:white;padding:2rem;border-radius:12px}h1{color:#205ba8}</style><h1>ColumnCheck review</h1><p>Local ML suggestions. Human review required. Source CSV is unchanged.</p><pre>'+content+'</pre></html>'

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path',type=Path)
    parser.add_argument('--json-out',type=Path)
    parser.add_argument('--html-out',type=Path)
    parser.add_argument('--schema',type=Path,help='Optional JSON mapping of exact column names to types; overrides model suggestions')
    args=parser.parse_args(argv)
    paths=[p.resolve() for p in [args.csv_path,args.schema,args.json_out,args.html_out] if p]
    if len(paths)!=len(set(paths)): parser.error('Input and output paths must be distinct')
    for output in [args.json_out,args.html_out]:
        if output and output.exists(): parser.error('Output already exists; choose a new path')
    start=time.perf_counter()
    try:
        schema=None
        if args.schema:
            if args.schema.stat().st_size>100_000: raise ValueError('Schema exceeds 100 KB limit')
            def unique_pairs(pairs):
                result={}
                for key,value in pairs:
                    if key in result: raise ValueError('Duplicate schema keys are not supported')
                    result[key]=value
                return result
            schema=json.loads(args.schema.read_text(encoding='utf-8'),object_pairs_hook=unique_pairs)
            if not isinstance(schema,dict): raise ValueError('Schema must be a JSON object')
        report=review(args.csv_path,train_model(),schema)
        report['runtime']={'seconds':round(time.perf_counter()-start,6),'sklearn':sklearn.__version__}
        encoded=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
        if args.json_out:
            with args.json_out.open('x',encoding='utf-8') as f: f.write(encoded)
        if args.html_out:
            with args.html_out.open('x',encoding='utf-8') as f: f.write(render_html(report))
        print(encoded)
    except (OSError,ValueError,UnicodeError,csv.Error) as exc:
        parser.exit(2,f'ColumnCheck: {exc}\n')

if __name__=='__main__': main()
