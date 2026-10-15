"""Product-grounded synthetic policy experiment, not a field trial."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
import joblib
from analyze import FEATURES,ci

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/processed';OUT=ROOT/'results'
SEED=20260925
REGIMES={'Reference':{},'No abandonment':{'dropout':0.},'Impatient buyers':{'dropout':.15},'Low valuations':{'vlow':.5,'vhigh':.85},'Higher merchant floors':{'floor_factor':.95},'Short platform horizon':{'horizon':4}}
STRATEGIES=np.array(['linear','late_concession','early_concession','reciprocal'])

def scenarios(products,reps=80,regime='Reference'):
    cfg={'dropout':.05,'floor_factor':.8,'vlow':.6,'vhigh':1.15,'horizon':8};cfg.update(REGIMES[regime])
    opening=pd.read_csv(DATA/'craigslist_dialogues.csv').query("split == 'train'").buyer_first.dropna().clip(.3,.85).values
    rows=[];randoms=[]
    for r in products.itertuples():
        seed=int(hashlib.sha256((r.product+':sim-v1').encode()).hexdigest()[:12],16)
        rng=np.random.default_rng(seed)
        for j in range(reps):
            v=rng.uniform(cfg['vlow'],cfg['vhigh']);o=min(float(rng.choice(opening)),.9*v)
            deadline=int(rng.choice([3,5,8]));kind=j%4;u=rng.random(8)
            floor=float(np.clip(r.lowest/r.current*cfg['floor_factor'],.2,.95))
            rows.append((r.product,j,r.category,r.current,floor,v,o,deadline,kind));randoms.append(u)
    df=pd.DataFrame(rows,columns=['product','rep','category','listing','floor','value','opening','deadline','kind'])
    return df,np.stack(randoms),cfg

def propose(policy,params,t,horizon,floor,bid,previous):
    """Observable-only seller action: no buyer value, deadline, type or random draws."""
    if policy=='Fixed price': ask=np.ones_like(bid)
    elif policy=='Uniform discount': ask=np.full_like(bid,1-params['discount'])
    else:
        clock=1-(1-floor)*params['terminal']*((t+1)/horizon)**params['exponent']
        if policy=='Time concession': ask=clock
        elif policy=='Buyer responsive':
            response=bid+params['markup']*np.maximum(1-bid,0)
            ask=(1-params['weight'])*clock+params['weight']*response
        else:raise ValueError(policy)
    return np.maximum(floor,np.minimum(previous,ask))

def rollout(df,randoms,cfg,policy,params,learned=None):
    n=len(df);listing=df.listing.values;floor_cents=np.ceil(df.floor.values*listing*100-1e-9);floor=floor_cents/(listing*100)
    value=df.value.values;opening=df.opening.values;deadline=df.deadline.values;kind=df.kind.values
    if policy=='Posted-price checkout':
        result=df[['product','rep','category','kind','listing','floor','value','deadline']].copy()
        result['policy']=policy;result['deal']=(value>=1).astype(int);result['rounds']=1
        result['price_ratio']=np.where(value>=1,1.,np.nan)
        result['seller_surplus']=np.where(value>=1,1-floor,0.)
        result['buyer_surplus']=np.where(value>=1,value-1,0.)
        result['total_surplus']=result.seller_surplus+result.buyer_surplus
        return result
    alive=np.ones(n,dtype=bool);deal=np.zeros(n,dtype=bool);p=np.full(n,np.nan);rounds=np.zeros(n,int);prevask=np.ones(n);prevbid=opening.copy();sellerhistory=[];buyerhistory=[]
    for t in range(cfg['horizon']):
        progress=np.minimum(1,t/np.maximum(deadline-1,1));exponent=np.choose(kind,[1.,3.,.5,1.])
        bid=opening+(value-opening)*progress**exponent
        if t:bid=np.where(kind==3,np.minimum(value,np.maximum(bid,prevbid+.3*np.maximum(prevask-prevbid,0))),bid)
        # A customer can buy at the listing; offering more than that is dominated.
        bid=np.floor(np.minimum(bid,1.)*listing*100+1e-9)/(listing*100)
        if policy=='Learned proposal':
            history=buyerhistory+sellerhistory+[bid]; othermin=np.min(np.stack(history),axis=0);othermax=np.max(np.stack(history),axis=0)
            X=pd.DataFrame({'log_listing':np.log(listing),'last_own':prevask if t else np.ones(n),'last_other':bid,'last_public':bid,'min_public':othermin,'max_public':othermax,
              'own_seen':int(t>0),'other_seen':1,'log_own_count':np.log1p(t),'log_other_count':np.log1p(t+1),'log_event':np.log1p(2*t+1),'is_buyer':0,
              'category':['electronics' if c=='electronics' else 'unknown' for c in df.category]})
            ask=np.maximum(floor,np.minimum(prevask,np.clip(learned.predict(X[FEATURES]),0,2)))
        else:ask=propose(policy,params,t,cfg['horizon'],floor,bid,prevask)
        ask=np.ceil(ask*listing*100-1e-9)/(listing*100)
        # Seller accepts an incoming offer that meets its current ask, otherwise quotes ask.
        seller_accept=alive&(bid>=ask-1e-12)
        willingness=bid+.25*np.maximum(value-bid,0)
        buyer_accept=alive&~seller_accept&(ask<=willingness+1e-12)&(ask<=value+1e-12)
        agreed=seller_accept|buyer_accept;p[seller_accept]=bid[seller_accept];p[buyer_accept]=ask[buyer_accept]
        deal|=agreed;rounds[alive]=t+1;alive&=~agreed
        alive&=(t+1<deadline)&(randoms[:,t]>=cfg['dropout'])
        prevask=ask;prevbid=bid;buyerhistory.append(bid);sellerhistory.append(ask)
    result=df[['product','rep','category','kind','listing','floor','value','deadline']].copy()
    result['policy']=policy;result['deal']=deal.astype(int);result['rounds']=rounds;result['price_ratio']=p
    result['seller_surplus']=np.where(deal,p-floor,0);result['buyer_surplus']=np.where(deal,value-p,0);result['total_surplus']=result.seller_surplus+result.buyer_surplus
    assert (result.seller_surplus>=-1e-9).all() and (result.buyer_surplus>=-1e-9).all()
    assert not np.any(deal&(floor>value+1e-9))
    return result

def choose(products):
    df,u,cfg=scenarios(products,reps=40)
    selected={'Fixed price':{}};all_scores=[]
    candidates={
      'Uniform discount':[{'discount':x} for x in [0,.05,.1,.15,.2,.25,.3,.35,.4,.45]],
      'Time concession':[{'terminal':q,'exponent':b} for q,b in itertools.product([.25,.5,.75,1.],[.5,1.,2.,4.])],
      'Buyer responsive':[{'terminal':q,'exponent':b,'weight':w,'markup':m} for q,b,w,m in itertools.product([.5,.75,1.],[1.,2.],[.25,.5,.75],[0.,.1,.2])]
    }
    for policy,grid in candidates.items():
        best=(-np.inf,None)
        for params in grid:
            r=rollout(df,u,cfg,policy,params);score=float(r.groupby('product').seller_surplus.mean().mean())
            all_scores.append({'policy':policy,'params':json.dumps(params,sort_keys=True),'development_seller_surplus':score,'agreement':float(r.deal.mean())})
            if score>best[0]:best=(score,params)
        selected[policy]=best[1];print('POLICY SELECTED',policy,best,flush=True)
    pd.DataFrame(all_scores).to_csv(OUT/'policy_selection.csv',index=False)
    (OUT/'frozen_policies.json').write_text(json.dumps({'selection':'182 development products, 40 scenarios each, reference regime only','selected':selected,'primary_metric':'seller surplus/listing per opportunity','regimes':REGIMES,'test_repetitions_per_product':80},indent=2))
    return selected

def main():
    products=pd.read_csv(DATA/'amazon_products.csv');selected=choose(products[products.split=='dev']);selected['Learned proposal']={};selected['Posted-price checkout']={}
    learned=joblib.load(OUT/'Gradient_boosting.joblib');test=products[products.split=='test'];summary={};contrasts={};group_tables=[]
    all_records=[]
    for regime in REGIMES:
        df,u,cfg=scenarios(test,reps=80,regime=regime);grouped={}
        for policy,params in selected.items():
            r=rollout(df,u,cfg,policy,params,learned);r['regime']=regime;all_records.append(r)
            g=r.groupby('product')[['deal','seller_surplus','buyer_surplus','total_surplus','rounds']].mean();grouped[policy]=g
            summary[regime+'|'+policy]={metric:ci(g[metric]) for metric in g}
            summary[regime+'|'+policy]['discount_conditional']=float(1-r.loc[r.deal==1,'price_ratio'].mean()) if r.deal.sum() else None
            for label,sg in r.groupby('kind'):
                group_tables.append({'regime':regime,'policy':policy,'buyer_strategy':STRATEGIES[label],**sg[['deal','seller_surplus','buyer_surplus','rounds']].mean().to_dict()})
        for comparator in ['Fixed price','Uniform discount','Time concession','Learned proposal']:
            contrasts[regime+'|Buyer responsive minus '+comparator]={metric:ci(grouped['Buyer responsive'][metric]-grouped[comparator][metric]) for metric in grouped[comparator]}
        print('REGIME COMPLETE',regime,{p:round(summary[regime+'|'+p]['seller_surplus']['mean'],4) for p in selected},flush=True)
    pd.concat(all_records,ignore_index=True).to_csv(OUT/'simulation_episodes.csv.gz',index=False,compression='gzip')
    pd.DataFrame(group_tables).to_csv(OUT/'simulation_strategy_sensitivity.csv',index=False)
    (OUT/'simulation_summary.json').write_text(json.dumps({'test_products':len(test),'scenarios_per_policy_regime':len(test)*80,'total_trajectories':len(test)*80*len(selected)*len(REGIMES),'summary':summary,'paired_contrasts':contrasts},indent=2))
    print('SIMULATION COMPLETE',flush=True)

if __name__=='__main__':main()
