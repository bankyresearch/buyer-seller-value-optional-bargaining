"""Existing-label validation, without pretending that a semantic annotation study occurred."""
from pathlib import Path
import sys,json,math
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'multidataset_bargaining'
sys.path.insert(0,str(BASE))
from prepare_data import digest,number
from analyze import FEATURES,aggregate_error,ci
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)

def pipeline(kind,param):
    prep=ColumnTransformer([('numeric',StandardScaler(),FEATURES[:-1]),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['category'])])
    estimator=Ridge(alpha=param) if kind=='Ridge' else HistGradientBoostingRegressor(loss='absolute_error',max_iter=150,max_leaf_nodes=param,min_samples_leaf=25,l2_regularization=1.,random_state=20260925,early_stopping=False)
    return make_pipeline(prep,estimator)

def structured_events():
    retained=pd.read_csv(BASE/'data/processed/craigslist_dialogues.csv');events=[]
    raw=ROOT.parent/'research_paper_bm3il/data/raw/craigslist'
    if not raw.exists():raw=BASE/'data/raw/craigslist'
    for split in ['train','dev','test']:
        allowed=set(retained.loc[retained.split==split,'dialogue'])
        seen=set()
        for d in json.loads((raw/(split+'.json')).read_text(encoding='utf8')):
            uid=digest(d.get('uuid'))
            if uid not in allowed or uid in seen:continue
            seen.add(uid);sid=digest(d['scenario_uuid']);kbs=d['scenario']['kbs'];roles=[k['personal']['Role'] for k in kbs]
            listing=float(kbs[0]['item']['Price']);cat=kbs[0]['item'].get('Category','unknown')
            history=[];own={'buyer':[],'seller':[]}
            for i,e in enumerate(d.get('events',[])):
                if e.get('action')!='offer' or e.get('agent') not in (0,1):continue
                value=number((e.get('data') or {}).get('price'))
                if value is None or not 0<=value/listing<=2:continue
                role=roles[e['agent']];other='seller' if role=='buyer' else 'buyer';a=own[role];b=own[other]
                events.append(dict(dialogue=uid,scenario=sid,split=split,category=cat,listing=listing,role=role,event=i,target_ratio=value/listing,
                    log_listing=math.log(listing),last_own=a[-1] if a else 1.,last_other=b[-1] if b else 1.,last_public=history[-1] if history else 1.,
                    min_public=min(history) if history else 1.,max_public=max(history) if history else 1.,own_seen=int(bool(a)),other_seen=int(bool(b)),
                    log_own_count=math.log1p(len(a)),log_other_count=math.log1p(len(b)),log_event=math.log1p(i),is_buyer=int(role=='buyer')))
                own[role].append(value/listing);history.append(value/listing)
    frame=pd.DataFrame(events);frame.to_csv(OUT/'structured_offer_events.csv',index=False);return frame

def structured_benchmark(from_processed=False):
    data=pd.read_csv(OUT/'structured_offer_events.csv') if from_processed else structured_events()
    train,dev,test=[data[data.split==s] for s in ['train','dev','test']]
    selection=[];models={};y=train.target_ratio.values
    for kind,grid in [('Ridge',[.1,1.,10.,100.]),('Gradient boosting',[7,15,31])]:
        best=(np.inf,None,None)
        for param in grid:
            model=pipeline(kind,param).fit(train[FEATURES],y)
            score=float(aggregate_error(dev,np.abs(np.clip(model.predict(dev[FEATURES]),0,2)-dev.target_ratio)).mean())
            selection.append(dict(model=kind,parameter=param,development_mae=score))
            if score<best[0]:best=(score,model,param)
        models[kind]=best[1];joblib.dump(best[1],OUT/('structured_'+kind.replace(' ','_')+'.joblib'),compress=3)
    outputs={'Training median':np.full(len(test),np.median(y)),'Last own price':test.last_own.values,'Last public price':test.last_public.values}
    outputs.update({k:np.clip(m.predict(test[FEATURES]),0,2) for k,m in models.items()})
    pred=test[['dialogue','scenario','category','role','listing','event','target_ratio']].copy();metrics={};errors={}
    for label,values in outputs.items():
        pred[label]=values;errors[label]=aggregate_error(test,np.abs(values-test.target_ratio.values));metrics[label]=ci(errors[label])
    pred.to_csv(OUT/'structured_predictions.csv',index=False);pd.DataFrame(selection).to_csv(OUT/'structured_selection.csv',index=False)
    result=dict(rows={s:int(sum(data.split==s)) for s in ['train','dev','test']},test_scenarios=int(test.scenario.nunique()),
        metrics=metrics,boosting_minus_ridge=ci(errors['Gradient boosting']-errors['Ridge']),
        scope='Native offer prices and native offer-only histories. Not a semantic parser accuracy estimate.')
    (OUT/'structured_summary.json').write_text(json.dumps(result,indent=2));print('STRUCTURED',json.dumps(result),flush=True)

def transport():
    data=pd.read_csv(BASE/'data/processed/craigslist_price_events.csv')
    train=data[data.split=='train'];test=data[data.split=='test'];original=joblib.load(BASE/'results/Gradient_boosting.joblib')
    records=[];summaries=[]
    for category,g in test.groupby('category'):
        tr=train[train.category!=category]
        assert category not in set(tr.category)
        model=pipeline('Gradient boosting',31).fit(tr[FEATURES],tr.target_ratio)
        p=np.clip(model.predict(g[FEATURES]),0,2);baseline=np.clip(original.predict(g[FEATURES]),0,2)
        err=aggregate_error(g,np.abs(p-g.target_ratio));baseerr=aggregate_error(g,np.abs(baseline-g.target_ratio))
        r=g[['scenario','dialogue','category','event','target_ratio']].copy();r['excluded_category_prediction']=p;r['original_prediction']=baseline;records.append(r)
        summaries.append(dict(category=category,training_events=len(tr),test_events=len(g),test_scenarios=g.scenario.nunique(),
                              excluded_category_mae=ci(err),original_mae=ci(baseerr),paired_difference=ci(err-baseerr)))
    pd.concat(records).to_csv(OUT/'category_transport_predictions.csv',index=False)
    (OUT/'category_transport_summary.json').write_text(json.dumps(summaries,indent=2));print('CATEGORY TRANSPORT',json.dumps(summaries),flush=True)

if __name__=='__main__':structured_benchmark('--from-processed' in sys.argv);transport()
