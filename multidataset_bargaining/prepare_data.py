"""Auditable transformations of three complete public sources."""
from pathlib import Path
import json,hashlib,math,re,collections
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'; OUT=ROOT/'data'/'processed'; OUT.mkdir(parents=True,exist_ok=True)
WORK=ROOT.parent

def digest(x): return hashlib.sha256(str(x).encode('utf8')).hexdigest()
def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError): return None
def money(x):
    if x is None: return None
    z=re.sub(r'[$,]','',str(x)).strip()
    return number(z)
def price(e):
    if e.get('action')=='offer': return number((e.get('data') or {}).get('price'))
    if e.get('action')!='message' or not isinstance(e.get('data'),str): return None
    pats=(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)',r'(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?\b|bucks?\b|\$)')
    vals={float(m.replace(',','')) for pat in pats for m in re.findall(pat,e['data'],re.I)}
    return vals.pop() if len(vals)==1 else None

def craigslist():
    src=WORK/'research_paper_bm3il/data/raw/craigslist'
    # The source may alternatively be supplied next to this script for reproduction.
    if not src.exists(): src=RAW/'craigslist'
    dialogues=[]; events=[]; exclusions=collections.Counter(); source=[]
    seen_ids=set(); seen_content=set(); past_scenarios=set()
    for split in ('train','dev','test'):
        path=src/(split+'.json'); raw=path.read_bytes(); ds=json.loads(raw); current=set()
        source.append({'file':path.name,'sha256':hashlib.sha256(raw).hexdigest(),'n_raw':len(ds),'bytes':len(raw)})
        for d in ds:
            uid=digest(d.get('uuid')); sid=digest(d.get('scenario_uuid'))
            content=digest(json.dumps([(e.get('agent'),e.get('action'),e.get('data')) for e in d.get('events',[])],sort_keys=True))
            kbs=(d.get('scenario') or {}).get('kbs') or []
            if len(kbs)!=2: exclusions[split+'_context']+=1;continue
            roles=[k.get('personal',{}).get('Role') for k in kbs]
            listing=number(kbs[0].get('item',{}).get('Price'))
            if set(roles)!={'buyer','seller'} or listing is None or listing<=0: exclusions[split+'_context']+=1;continue
            if uid in seen_ids or content in seen_content: exclusions[split+'_duplicate']+=1;continue
            if sid in past_scenarios: exclusions[split+'_scenario_overlap']+=1;continue
            if not d.get('scenario_uuid'): exclusions[split+'_missing_scenario']+=1;continue
            seen_ids.add(uid);seen_content.add(content);current.add(sid)
            cat=kbs[0]['item'].get('Category','unknown'); hist=[]; own={'buyer':[],'seller':[]}
            msgs=0; bounded=0
            for i,e in enumerate(d.get('events',[])):
                if e.get('action')=='message': msgs+=1
                actor=e.get('agent')
                if actor not in (0,1):continue
                val=price(e)
                if val is None: continue
                if not 0<=val/listing<=2:
                    exclusions[split+'_out_of_domain_price']+=1;continue
                role=roles[actor]; other='buyer' if role=='seller' else 'seller'
                prior=own[role]; counterpart=own[other]
                events.append({'dialogue':uid,'scenario':sid,'split':split,'category':cat,'listing':listing,'role':role,'event':i,'source_action':e['action'],
                  'target_ratio':val/listing,'log_listing':math.log(listing),'last_own':prior[-1] if prior else 1.,'last_other':counterpart[-1] if counterpart else 1.,
                  'last_public':hist[-1] if hist else 1.,'min_public':min(hist) if hist else 1.,'max_public':max(hist) if hist else 1.,
                  'own_seen':int(bool(prior)),'other_seen':int(bool(counterpart)),'log_own_count':math.log1p(len(prior)),'log_other_count':math.log1p(len(counterpart)),
                  'log_event':math.log1p(i),'is_buyer':int(role=='buyer')})
                own[role].append(val/listing);hist.append(val/listing); bounded+=1
            outcome=d.get('outcome') or {}; offer=outcome.get('offer') or {}; final=number(offer.get('price'))
            agreed=int(outcome.get('reward')==1)
            finalratio=final/listing if final is not None and 0<=final/listing<=2 else np.nan
            dialogues.append({'dialogue':uid,'scenario':sid,'split':split,'category':cat,'listing':listing,'messages':msgs,'price_events':bounded,'agreement':agreed,
              'final_ratio':finalratio,'buyer_first':own['buyer'][0] if own['buyer'] else np.nan,'buyer_last':own['buyer'][-1] if own['buyer'] else np.nan,
              'seller_first':own['seller'][0] if own['seller'] else np.nan,'seller_last':own['seller'][-1] if own['seller'] else np.nan,
              'buyer_price_count':len(own['buyer']),'seller_price_count':len(own['seller'])})
        past_scenarios.update(current)
    pd.DataFrame(dialogues).to_csv(OUT/'craigslist_dialogues.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'craigslist_price_events.csv',index=False)
    (OUT/'craigslist_audit.json').write_text(json.dumps({'source':source,'exclusions':dict(exclusions),'dialogues':len(dialogues),'price_events':len(events)},indent=2))
    print('Craigslist',len(dialogues),len(events),dict(exclusions),flush=True)

def amazon():
    rows=[]; seen=set(); audit=collections.Counter()
    for p in sorted((RAW/'amazon').glob('*.json')):
        for r in json.loads(p.read_text(encoding='utf8')):
            audit['raw_records']+=1
            link=r.get('amazon_link','') or r.get('link','')
            match=re.search(r'(?:/dp/|/product/)([A-Z0-9]{10})',link)
            key=match.group(1) if match else digest(link or r.get('title',''))
            if key in seen: audit['duplicate_product']+=1; continue
            seen.add(key)
            vals={x:money(r.get(x+'_price')) for x in ['current','lowest','highest','list','average']}
            valid=vals['current'] is not None and vals['current']>0
            audit['invalid_current']+=not valid
            h=int(digest('product-split-20260925:'+key)[:12],16)%10
            split='train' if h<6 else 'dev' if h<8 else 'test'
            rows.append({'product':digest(key),'category':r.get('category',p.stem),'split':split,'valid_current':int(valid),**vals})
    df=pd.DataFrame(rows);df.to_csv(OUT/'amazon_products.csv',index=False)
    audit['unique_products']=len(df);audit['valid_products']=int(df.valid_current.sum())
    (OUT/'amazon_audit.json').write_text(json.dumps(dict(audit),indent=2));print('Amazon',dict(audit),flush=True)

SAT={'Extremely dissatisfied':1,'Slightly dissatisfied':2,'Undecided':3,'Neither satisfied nor dissatisfied':3,'Slightly satisfied':4,'Extremely satisfied':5}
LIKE={'Extremely dislike':1,'Slightly dislike':2,'Undecided':3,'Neither like nor dislike':3,'Slightly like':4,'Extremely like':5}
def casino():
    ds=json.loads((RAW/'casino/casino.json').read_text(encoding='utf8'));rows=[];audit=collections.Counter();strategies=[];dialogue_rows=[]
    for d in ds:
        ident=digest(d['dialogue_id']); infos=d['participant_info'];agents=list(infos)
        accepted=any(x['text']=='Accept-Deal' for x in d['chat_logs'])
        msgs=[x for x in d['chat_logs'] if x['text'] not in ('Submit-Deal','Accept-Deal','Reject-Deal','Walk-Away')]
        annotations=d.get('annotations') or []
        counts=collections.Counter(a.strip() for x in annotations for a in x[1].split(','))
        dialogue_rows.append({'dialogue':ident,'messages':len(msgs),'agreement':int(accepted),'annotated':int(bool(annotations)),**{'strategy_'+k:v for k,v in counts.items()}})
        for aid in agents:
            o=infos[aid].get('outcomes',{});other=agents[1] if aid==agents[0] else agents[0]
            sat=SAT.get(o.get('satisfaction'));like=LIKE.get(o.get('opponent_likeness'))
            if sat is None:audit['unmapped_satisfaction_'+str(o.get('satisfaction'))]+=1
            if like is None:audit['unmapped_liking_'+str(o.get('opponent_likeness'))]+=1
            rows.append({'dialogue':ident,'actor':aid,'own_points':number(o.get('points_scored')),'other_points':number(infos[other].get('outcomes',{}).get('points_scored')),
              'satisfaction':sat,'liking':like,'messages':len(msgs),'agreement':int(accepted)})
    df=pd.DataFrame(rows);df.to_csv(OUT/'casino_participants.csv',index=False)
    pd.DataFrame(dialogue_rows).fillna(0).to_csv(OUT/'casino_dialogues.csv',index=False)
    audit['dialogues']=len(ds);audit['participant_records']=len(df);audit['annotated_dialogues']=sum(bool(d.get('annotations')) for d in ds)
    (OUT/'casino_audit.json').write_text(json.dumps(dict(audit),indent=2));print('CaSiNo',dict(audit),flush=True)

if __name__=='__main__': craigslist();amazon();casino()
