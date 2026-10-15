"""Independent recomputation of extension summaries; output is never validation of behaviour."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(res,existing_only=False):
    p=pd.read_csv(res/'product_outcomes.csv');dist=pd.read_csv(res/'product_distribution.csv')
    seeds=pd.read_csv(res/'seed_means.csv');loco=pd.read_csv(res/'leave_category_out.csv');bounds=pd.read_csv(res/'cost_boundaries.csv')
    checks=0
    for (regime,policy),g in p.groupby(['regime','policy']):
        assert len(g)==193 and g['product'].is_unique
        d=dist[(dist.regime==regime)&(dist.policy==policy)].iloc[0]
        assert d.both_positive==int(((g.seller_increment>0)&(g.buyer_increment>0)).sum())
        assert d.seller_negative==int((g.seller_increment<0).sum())
        assert d.buyer_negative==int((g.buyer_increment<0).sum())
        s=seeds[(seeds.regime==regime)&(seeds.policy==policy)]
        np.testing.assert_allclose(g[['seller_increment','buyer_increment','joint_increment']].mean(),s[['seller_increment','buyer_increment','joint_increment']].mean(),atol=1e-12,rtol=0)
        l=loco[(loco.regime==regime)&(loco.policy==policy)]
        assert len(l)==g.category.nunique()
        for row in l.itertuples():
            sub=g[g.category!=row.excluded_category];assert len(sub)==row.remaining_products
            for metric in ['seller_increment','buyer_increment','joint_increment']:
                assert abs(sub[metric].mean()-getattr(row,metric))<1e-12
        if regime=='Reference' and policy=='Full-state FQI':
            assert (l.seller_increment>0).all() and (l.buyer_increment>0).all()
        for party in ['seller','buyer']:
            b=bounds[(bounds.regime==regime)&(bounds.policy==policy)&(bounds.stakeholder==party)].sort_values('fallback')
            assert len(b)==101
            assert (np.diff(b['value'])>=-1e-12).all()
            centre=b[abs(b.fallback-.5)<1e-12].iloc[0]
            # V(c) = V(.005) + (.005-c)H; its root must match the stored boundary.
            root=.005+g[party+'_increment'].mean()/g.negotiation_rounds.mean()
            assert abs(root-centre['value'])<1e-12
        checks+=1
    report=dict(status='passed',policy_regime_cells=checks,product_counts=True,seed_average_identity=True,
        independent_category_deletions=True,threshold_roots=True,threshold_monotonicity=True,
        scope='Same-team analysis validation, not human behaviour or independent external validation')
    if not existing_only:
        manifest=pd.read_csv(res/'structural_manifest.csv');assert len(manifest)==32
        assert manifest.rows.sum()==1482240
        sp=pd.read_csv(res/'structural_product_outcomes.csv');si=pd.read_csv(res/'structural_intervals.csv')
        for row in manifest.itertuples():
            path=res/'trajectories'/row.file;assert sha(path)==row.sha256
            b=pd.read_csv(path);assert len(b)==row.rows
            assert not b.duplicated(['product','rep','training_seed']).any()
            # Reconstruct from receipts independently of coefficient aggregation.
            deal=b.deal.to_numpy().astype(bool);v=b.value.to_numpy();f=b.floor.to_numpy();eligible=v>=1
            price=np.nan_to_num(b.price_ratio.to_numpy());h=b['rounds'].to_numpy()
            checkout=.5*eligible;negotiated=.5*deal;fallback=.25*(~deal)*eligible
            purchases=checkout+negotiated+fallback
            receipts=checkout+negotiated*price+fallback
            s=receipts-f*purchases-.005*.5*h-eligible*(1-f)
            buyer=v*purchases-receipts-.005*.5*h-eligible*(v-1)
            expected=pd.DataFrame({'product':b['product'],'seller_increment':s,'buyer_increment':buyer}).groupby('product').mean()
            saved=sp[(sp.setting==row.setting)&(sp.policy==row.policy)].set_index('product').sort_index()
            np.testing.assert_allclose(expected,saved[['seller_increment','buyer_increment']],rtol=0,atol=1e-12)
            for metric in ['seller_increment','buyer_increment']:
                summary=si[(si.setting==row.setting)&(si.policy==row.policy)&(si.metric==metric)].iloc[0]
                assert abs(expected[metric].mean()-summary['mean'])<1e-12
        report.update(structural_files=32,structural_executions=1482240,structural_hashes=True,independent_receipt_reconstruction=True)
    (res/'revision_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=ROOT/'results');parser.add_argument('--existing-only',action='store_true');args=parser.parse_args();run(args.results,args.existing_only)
