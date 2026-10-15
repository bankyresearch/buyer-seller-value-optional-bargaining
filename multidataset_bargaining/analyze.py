"""Descriptive, predictive and relational analyses; no causal interpretation."""
from pathlib import Path
import json,hashlib,sys,platform,warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data/processed'; OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SEED=20260925;BOOT=2000

def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def ci(values,seed=SEED):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]; rng=np.random.default_rng(seed)
    z=np.array([a[rng.integers(len(a),size=len(a))].mean() for _ in range(BOOT)])
    return {'mean':float(a.mean()),'lo':float(np.quantile(z,.025)),'hi':float(np.quantile(z,.975)),'clusters':len(a)}

def descriptive():
    c=pd.read_csv(DATA/'craigslist_dialogues.csv');a=pd.read_csv(DATA/'amazon_products.csv');p=pd.read_csv(DATA/'casino_participants.csv');d=pd.read_csv(DATA/'casino_dialogues.csv')
    def summary(df,cols):
        return {col:{'n':int(df[col].count()),'mean':float(df[col].mean()),'median':float(df[col].median()),'q25':float(df[col].quantile(.25)),'q75':float(df[col].quantile(.75))} for col in cols}
    craig={'dialogues':len(c),'scenarios':c.scenario.nunique(),'agreement_dialogue_mean':float(c.agreement.mean()),'agreement_scenario_ci':ci(c.groupby('scenario').agreement.mean()),'summary':summary(c,['listing','messages','buyer_first','seller_first'])}
    c['buyer_concession']=c.buyer_last-c.buyer_first;c['seller_concession']=c.seller_first-c.seller_last
    craig['concessions']=summary(c.loc[(c.buyer_price_count>=2)&(c.seller_price_count>=2)],['buyer_concession','seller_concession'])
    cat=c.groupby('category').agg(n=('dialogue','size'),scenarios=('scenario','nunique'),agreement=('agreement','mean'),listing_median=('listing','median'),messages_median=('messages','median'),buyer_opening=('buyer_first','median'),seller_opening=('seller_first','median'))
    cat.to_csv(OUT/'craigslist_category_summary.csv')
    a['low_current_ratio']=a.lowest/a.current;a['high_current_ratio']=a.highest/a.current
    amazon={'products':len(a),'categories':a.category.nunique(),'splits':a.groupby('split').size().to_dict(),'summary':summary(a,['current','lowest','highest','low_current_ratio','high_current_ratio']),
       'low_above_current':int((a.lowest>a.current).sum()),'high_below_current':int((a.highest<a.current).sum())}
    a.groupby('category').agg(products=('product','size'),current_median=('current','median'),low_current_median=('low_current_ratio','median')).to_csv(OUT/'amazon_category_summary.csv')
    casino={'dialogues':len(d),'participant_records':len(p),'accepted_dialogues':int(d.agreement.sum()),'agreement_ci':ci(d.agreement),'summary':summary(p,['own_points','satisfaction','liking','messages']),
      'satisfaction_distribution':{str(k):int(v) for k,v in p.satisfaction.value_counts().sort_index().items()},'annotated_dialogues':int(d.annotated.sum())}
    dump('descriptive.json',{'craigslist':craig,'amazon':amazon,'casino':casino})
    print('DESCRIPTIVE',json.dumps({'craigslist':craig['agreement_scenario_ci'],'casino':casino['agreement_ci']}),flush=True)

FEATURES=['log_listing','last_own','last_other','last_public','min_public','max_public','own_seen','other_seen','log_own_count','log_other_count','log_event','is_buyer','category']
def aggregate_error(frame,err):
    z=frame[['scenario','dialogue']].copy();z['error']=err
    return z.groupby(['scenario','dialogue']).error.mean().groupby('scenario').mean()
def prediction():
    d=pd.read_csv(DATA/'craigslist_price_events.csv');train=d[d.split=='train'].copy();dev=d[d.split=='dev'].copy();test=d[d.split=='test'].copy()
    for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not set(d[d.split==a].scenario)&set(d[d.split==b].scenario)
    Xtr=train[FEATURES];Xdv=dev[FEATURES];Xte=test[FEATURES];y=train.target_ratio.values
    def preprocessing():return ColumnTransformer([('numeric',StandardScaler(),FEATURES[:-1]),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['category'])])
    models={};selection=[]
    for name,params in [('Ridge',[.1,1.,10.,100.]),('Gradient boosting',[7,15,31])]:
        best=None
        for param in params:
            estimator=Ridge(alpha=param) if name=='Ridge' else HistGradientBoostingRegressor(loss='absolute_error',max_iter=150,max_leaf_nodes=param,min_samples_leaf=25,l2_regularization=1.,random_state=SEED,early_stopping=False)
            model=make_pipeline(preprocessing(),estimator);model.fit(Xtr,y)
            pred=np.clip(model.predict(Xdv),0,2);score=float(aggregate_error(dev,np.abs(pred-dev.target_ratio.values)).mean())
            selection.append({'model':name,'parameter':param,'development_scenario_mae':score})
            if best is None or score<best[0]:best=(score,model,param)
        models[name]=best[1];joblib.dump(best[1],OUT/(name.replace(' ','_')+'.joblib'))
        print('SELECTED',name,best[2],best[0],flush=True)
    outputs={'Training median':np.repeat(np.median(y),len(test)),'Last own price':test.last_own.values,'Last public price':test.last_public.values}
    outputs.update({k:np.clip(v.predict(Xte),0,2) for k,v in models.items()})
    pred_frame=test[['dialogue','scenario','role','category','listing','target_ratio','source_action','event']].copy()
    results={};groups={}
    for name,pred in outputs.items():
        pred_frame[name]=pred
        err=np.abs(pred-test.target_ratio.values);agg=aggregate_error(test,err);groups[name]=agg
        results[name]={'relative_mae':ci(agg),'dollar_mae':ci(aggregate_error(test,err*test.listing.values))}
        for role in ['buyer','seller']:
            m=test.role==role;results[name][role+'_mae']=ci(aggregate_error(test[m],err[m]))
    pred_frame.to_csv(OUT/'craigslist_predictions.csv',index=False)
    baseline=min(['Last own price','Last public price'],key=lambda k:float(aggregate_error(dev,np.abs(dev['last_own' if k=='Last own price' else 'last_public']-dev.target_ratio)).mean()))
    contrast=ci(groups['Gradient boosting']-groups[baseline])
    bycat=[]
    for cat,g in pred_frame.groupby('category'):
        bycat.append({'category':cat,'n':len(g),'scenarios':g.scenario.nunique(),**{name:float(aggregate_error(g,np.abs(g[name]-g.target_ratio)).mean()) for name in outputs}})
    pd.DataFrame(bycat).to_csv(OUT/'craigslist_prediction_by_category.csv',index=False)
    pd.DataFrame(selection).to_csv(OUT/'prediction_model_selection.csv',index=False)
    dump('prediction_summary.json',{'features':FEATURES,'training_rows':len(train),'development_rows':len(dev),'test_rows':len(test),'test_scenarios':test.scenario.nunique(),'metrics':results,'development_selected_persistence':baseline,'boosting_minus_persistence':contrast})
    print('PREDICTION',json.dumps(results),flush=True)

def casino_relations():
    df=pd.read_csv(DATA/'casino_participants.csv');ids=df.dialogue.unique(); groups=[g.index.values for _,g in df.groupby('dialogue',sort=False)];rng=np.random.default_rng(SEED)
    pairs=[('own_points','satisfaction'),('other_points','satisfaction'),('liking','satisfaction'),('messages','satisfaction')]
    res={}
    for a,b in pairs:
        data=df[[a,b]].values; stat=float(spearmanr(data[:,0],data[:,1]).statistic);bs=[]
        for _ in range(BOOT):
            ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]);v=data[ix];bs.append(spearmanr(v[:,0],v[:,1]).statistic)
        res[a+'_with_'+b]={'rho':stat,'lo':float(np.quantile(bs,.025)),'hi':float(np.quantile(bs,.975)),'dialogue_clusters':len(groups),'participant_records':len(df)}
    # One-record-per-dialogue robustness removes the two-observations-per-dyad structure.
    mean=df.groupby('dialogue')[['own_points','satisfaction','liking','messages']].mean()
    res['dialogue_mean_points_satisfaction_rho']=float(spearmanr(mean.own_points,mean.satisfaction).statistic)
    # Own-minus-counterpart differences remove all shared dialogue-level context descriptively.
    wide=df.pivot(index='dialogue',columns='actor',values=['own_points','satisfaction'])
    actors=sorted(df.actor.unique());dp=wide[('own_points',actors[0])]-wide[('own_points',actors[1])];ds=wide[('satisfaction',actors[0])]-wide[('satisfaction',actors[1])]
    res['within_dialogue_difference_rho']=float(spearmanr(dp,ds).statistic)
    dump('casino_associations.json',res);print('CASINO ASSOCIATIONS',json.dumps(res),flush=True)

if __name__=='__main__':
    import sklearn,scipy
    dump('analysis_environment.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'seed':SEED,'bootstrap_replicates':BOOT})
    descriptive();prediction();casino_relations()
