"""Leakage-aware B0-B10 DAIC-WOZ research experiments.

This script deliberately labels HSED as an algorithmic residual subgroup, not a clinical phenotype.
It does NOT use development labels to construct training weights, thresholds, or synthetic samples.
"""
from __future__ import annotations
import json, logging, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.base import clone
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'reports' / 'final_evaluation_v2'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
PHQ_CUTOFF = 10.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('phase7')
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# Existing repository data-loader contract is intentionally reused.
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.phase6_regression import load_config, load_modality_data
from src.validation import validate_split_separation


def pid_series(pids):
    return pd.Series([str(x).strip() for x in pids], dtype='string')


def assert_same_order(base, other, label):
    a, b = pid_series(base), pid_series(other)
    if len(a) != len(b) or not a.equals(b):
        raise ValueError(f'Participant order mismatch for {label}. Align features by participant_id before continuing.')


def metric_row(model, yb, pb, yc, pc, split, hsed_threshold=None):
    pred = (pb >= 0.5).astype(int)
    out = {'model_id':model, 'split':split, 'n':len(yb), 'positive_n':int(yb.sum()),
           'accuracy':accuracy_score(yb,pred), 'macro_f1':f1_score(yb,pred,average='macro',zero_division=0),
           'balanced_accuracy':balanced_accuracy_score(yb,pred), 'precision':precision_score(yb,pred,zero_division=0),
           'recall':recall_score(yb,pred,zero_division=0),
           'roc_auc':roc_auc_score(yb,pb) if len(np.unique(yb))==2 else np.nan,
           'pr_auc':average_precision_score(yb,pb) if len(np.unique(yb))==2 else np.nan,
           'mae':mean_absolute_error(yc,pc), 'rmse':mean_squared_error(yc,pc)**0.5,
           'r2':r2_score(yc,pc), 'spearman':spearmanr(yc,pc).statistic if np.std(pc)>1e-10 else np.nan,
           'mean_discrepancy_high_phq':float(np.mean(yc[yb==1]-pc[yb==1])) if np.any(yb==1) else np.nan,
           'hsed_candidates_high_phq':int(np.sum((yc[yb==1]-pc[yb==1]) >= hsed_threshold)) if hsed_threshold is not None and np.any(yb==1) else np.nan,
           'hsed_threshold_train_oof':hsed_threshold}
    return out, pred


def make_pipe(kind, alpha=100.0, C=1.0):
    if kind == 'reg':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',Ridge(alpha=alpha))])
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(C=C,max_iter=2000,class_weight=None,random_state=SEED))])


def oof_regression_residuals(X, y, pids, folds=5):
    """OOF residuals aligned explicitly by participant ID; preprocessing refit inside each fold."""
    splitter = KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, va in splitter.split(X):
        model = make_pipe('reg')
        model.fit(X[tr], y[tr])
        oof[va] = y[va] - model.predict(X[va])
    if not np.isfinite(oof).all(): raise RuntimeError('OOF residuals contain missing/non-finite values')
    return pd.DataFrame({'participant_id':pid_series(pids), 'phq8_true':y, 'discrepancy':oof})


def weights_from_residuals(oof_df, pids, severity_control=False, clip=(0.5, 3.0)):
    keyed = oof_df.copy(); keyed['participant_id'] = pid_series(keyed['participant_id'])
    keyed = keyed.set_index('participant_id')
    ids = pid_series(pids)
    missing = ids[~ids.isin(keyed.index)].tolist()
    if missing: raise ValueError(f'Missing OOF residuals for participants: {missing[:8]}')
    aligned = keyed.loc[ids]
    d = aligned['discrepancy'].to_numpy(float)
    if not np.isfinite(d).all(): raise ValueError('Non-finite OOF discrepancy')
    if severity_control:
        # OOF residual vs true PHQ is fit only on training participants. Residualized discrepancy
        # reduces, but cannot guarantee removal of, severity dependence.
        y = aligned['phq8_true'].to_numpy(float)
        slope, intercept = np.polyfit(y, d, 1)
        d = d - (slope*y + intercept)
    scale = max(float(np.std(d, ddof=1)), 1e-6)
    w = 1.0 + np.maximum(0.0, d/scale)
    w = np.clip(w, clip[0], clip[1])
    return w / np.mean(w)


class MultiTaskNet(nn.Module):
    def __init__(self, d, hidden=64, dropout=.25):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,hidden), nn.ReLU())
        self.cls = nn.Linear(hidden,1); self.reg = nn.Linear(hidden,1)
    def forward(self,x):
        z=self.encoder(x); return self.cls(z).squeeze(-1), self.reg(z).squeeze(-1), z

class ContrastiveNet(nn.Module):
    def __init__(self,d,hidden=64,proj=32):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU()); self.projector=nn.Linear(hidden,proj)
    def forward(self,x):
        z=self.encoder(x); return z,F.normalize(self.projector(z),dim=1)


def supcon_loss(z, y, weights=None, temperature=.1):
    n=z.shape[0]
    if n < 3: return z.sum()*0
    sim=(z@z.T)/temperature
    sim=sim-sim.max(dim=1,keepdim=True).values.detach()
    eye=torch.eye(n,device=z.device,dtype=torch.bool)
    pos=(y[:,None]==y[None,:]) & ~eye
    exp=torch.exp(sim).masked_fill(eye,0)
    logp=sim-torch.log(exp.sum(dim=1,keepdim=True).clamp_min(1e-8))
    counts=pos.sum(1); valid=counts>0
    if not valid.any(): return z.sum()*0
    per=-(logp.masked_fill(~pos,0).sum(1)[valid]/counts[valid])
    if weights is not None: per=per*weights[valid]
    return per.mean()


def fit_torch_multitask(X, yb, yc, sample_w=None, epochs=100, supcon=False, triplet=False):
    X_t=torch.tensor(X,dtype=torch.float32,device=DEVICE); yb_t=torch.tensor(yb,dtype=torch.float32,device=DEVICE); yc_t=torch.tensor(yc,dtype=torch.float32,device=DEVICE)
    w_t=torch.ones_like(yc_t) if sample_w is None else torch.tensor(sample_w,dtype=torch.float32,device=DEVICE)
    net=MultiTaskNet(X.shape[1]).to(DEVICE); opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    for _ in range(epochs):
        net.train(); opt.zero_grad(); logits,reg,_=net(X_t)
        cls=F.binary_cross_entropy_with_logits(logits,yb_t,reduction='none')
        regloss=(reg-yc_t)**2
        loss=(w_t*cls).mean()+.05*(w_t*regloss).mean(); loss.backward(); opt.step()
    return net


def fit_contrastive(X,yb,yc,weights=None,hard_negative=False,epochs=80):
    X_t=torch.tensor(X,dtype=torch.float32,device=DEVICE); y_t=torch.tensor(yb,dtype=torch.long,device=DEVICE)
    w_t=None if weights is None else torch.tensor(weights,dtype=torch.float32,device=DEVICE)
    net=ContrastiveNet(X.shape[1]).to(DEVICE); opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    if hard_negative:
        # Fixed participant-level triplets mined from training-only standardized feature cosine similarity.
        xn=F.normalize(X_t,dim=1); sim=xn@xn.T
        pos_idx=np.where(yb==1)[0]; neg_idx=np.where(yb==0)[0]
        if len(pos_idx)<2 or len(neg_idx)<1: raise ValueError('Triplet mining requires both classes and >=2 positives')
        triplets=[]; rng=np.random.default_rng(SEED)
        for a in pos_idx:
            p=int(rng.choice(pos_idx[pos_idx!=a])); ranked=neg_idx[np.argsort(sim[a,neg_idx].detach().cpu().numpy())[-min(5,len(neg_idx)):]]
            n=int(rng.choice(ranked)); triplets.append((a,p,n))
        ai,pi,ni=map(lambda k:torch.tensor([t[k] for t in triplets],device=DEVICE),range(3))
        for _ in range(epochs):
            net.train(); opt.zero_grad(); _,za=net(X_t[ai]); _,zp=net(X_t[pi]); _,zn=net(X_t[ni])
            loss=F.triplet_margin_loss(za,zp,zn,margin=.5); loss.backward(); opt.step()
    else:
        for _ in range(epochs):
            net.train(); opt.zero_grad(); _,z=net(X_t); loss=supcon_loss(z,y_t,w_t); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): ztr=net(X_t)[0].cpu().numpy()
    return net,ztr


def latent_interpolate(Xz, yb, yc, weights, target_positive_ratio=.5, seed=SEED):
    """Weighted latent-space interpolation among positive training participants only.
    Interpolates latent vectors and continuous PHQ labels; never uses dev data.
    Returns augmented latent features and aligned labels/weights.
    """
    rng=np.random.default_rng(seed); pos=np.flatnonzero(yb==1); neg=np.flatnonzero(yb==0)
    if len(pos)<2 or len(neg)==0: raise ValueError('Latent interpolation needs >=2 positives and >=1 negative')
    target_n=int(np.ceil(target_positive_ratio*len(neg)/(1-target_positive_ratio)))
    n_new=max(0,target_n-len(pos))
    if n_new==0: return Xz,yb,yc,weights
    probs=weights[pos]/weights[pos].sum(); rows=[]; labs=[]; regs=[]; ws=[]
    for _ in range(n_new):
        i=int(rng.choice(pos,p=probs)); candidates=pos[pos!=i]
        # nearest positive in learned latent space, then random interpolation coefficient
        dist=np.linalg.norm(Xz[candidates]-Xz[i],axis=1); j=int(candidates[np.argmin(dist)])
        lam=float(rng.uniform(0.15,.85)); rows.append(Xz[i]+lam*(Xz[j]-Xz[i]))
        labs.append(1); regs.append(float(yc[i]+lam*(yc[j]-yc[i]))); ws.append(float((weights[i]+weights[j])/2))
    return np.vstack([Xz,np.asarray(rows)]),np.r_[yb,np.asarray(labs)],np.r_[yc,np.asarray(regs)],np.r_[weights,np.asarray(ws)]


def run():
    cfg=load_config()
    # B3/B10 primary modality: MFCC + whole-dialogue MiniLM embeddings (462 features).
    data=load_modality_data('mm_mfcc_text',cfg)
    pids_tr=list(map(str,data['train_pids'])); pids_dev=list(map(str,data['dev_pids']))
    validate_split_separation(pids_tr,pids_dev)
    Xtr=np.asarray(data['X_train'],float); Xdev=np.asarray(data['X_dev'],float)
    ytr=np.asarray(data['y_train'],float); ydev=np.asarray(data['y_dev'],float)
    if len(pids_tr)!=len(ytr) or len(pids_dev)!=len(ydev): raise ValueError('PID/label length mismatch')
    if len(set(pids_tr))!=len(pids_tr) or len(set(pids_dev))!=len(pids_dev): raise ValueError('Duplicate participant IDs')
    if set(pids_tr)&set(pids_dev): raise ValueError('Train/dev participant overlap')
    yb=(ytr>=PHQ_CUTOFF).astype(int); ybd=(ydev>=PHQ_CUTOFF).astype(int)
    if len(np.unique(yb))<2: raise ValueError('Training labels contain one class only')

    # Preprocessing fit on training only. OOF baseline independently refits preprocessing per fold.
    prep=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    X=prep.fit_transform(Xtr); Xv=prep.transform(Xdev)
    oof=oof_regression_residuals(Xtr,ytr,pids_tr)
    oof.to_csv(OUT/'train_oof_discrepancy_aligned.csv',index=False)
    weights=weights_from_residuals(oof,pids_tr)
    weights_ctrl=weights_from_residuals(oof,pids_tr,severity_control=True)
    # Candidate thresholds are model-specific, computed only from training OOF residuals among PHQ>=10.
    tau_oof=float(np.quantile(oof.loc[oof.phq8_true>=PHQ_CUTOFF,'discrepancy'],.75))

    rows=[]; pred_rows=[]
    def save(model_id, probs, regpred, threshold=tau_oof):
        met,_=metric_row(model_id,ybd,probs,ydev,regpred,'dev',threshold); rows.append(met)
        for i,pid in enumerate(pids_dev):
            pred_rows.append({'participant_id':pid,'model_id':model_id,'split':'dev','phq8_true':ydev[i],
              'phq8_pred':regpred[i],'discrepancy':ydev[i]-regpred[i],'true_binary':int(ybd[i]),
              'pred_prob':float(probs[i]),'pred_binary':int(probs[i]>=.5),'hsed_candidate':bool(ybd[i] and ydev[i]-regpred[i]>=threshold)})

    # B0 dummy
    dummyc=DummyClassifier(strategy='most_frequent').fit(X,yb); dummyr=DummyRegressor(strategy='mean').fit(X,ytr)
    save('B0_Dummy',dummyc.predict_proba(Xv)[:,1],dummyr.predict(Xv))
    # B1/B2 modality baselines are loaded separately; IDs must align exactly.
    for mod,mid in [('egemaps','B1_eGeMAPS'),('mfcc','B1_MFCC'),('text_minilm','B2_Text_MiniLM')]:
        dm=load_modality_data(mod,cfg); assert_same_order(pids_tr,dm['train_pids'],mod+' train'); assert_same_order(pids_dev,dm['dev_pids'],mod+' dev')
        pipec=make_pipe('cls'); piper=make_pipe('reg'); pipec.fit(dm['X_train'],yb); piper.fit(dm['X_train'],ytr)
        save(mid,pipec.predict_proba(dm['X_dev'])[:,1],piper.predict(dm['X_dev']))
    # B3 standard multimodal logistic + Ridge
    cl=make_pipe('cls'); rg=make_pipe('reg'); cl.fit(Xtr,yb); rg.fit(Xtr,ytr); save('B3_MFCC_Text',cl.predict_proba(Xdev)[:,1],rg.predict(Xdev))
    # B4 multitask network
    net4=fit_torch_multitask(X,yb,ytr,epochs=100); net4.eval()
    with torch.no_grad(): lg,rr,_=net4(torch.tensor(Xv,dtype=torch.float32,device=DEVICE)); pr=torch.sigmoid(lg).cpu().numpy(); rp=rr.cpu().numpy()
    save('B4_Multitask',pr,rp)
    # B5 generic supervised contrastive representation
    cn5,z5=fit_contrastive(X,yb,ytr,epochs=80); cn5.eval()
    with torch.no_grad(): zv5=cn5(torch.tensor(Xv,dtype=torch.float32,device=DEVICE))[0].cpu().numpy()
    cc=LogisticRegression(max_iter=2000,random_state=SEED).fit(z5,yb); cr=Ridge(alpha=100).fit(z5,ytr); save('B5_SupCon',cc.predict_proba(zv5)[:,1],cr.predict(zv5))
    # B6 ordinary random minority oversampling on train input space
    pos=np.flatnonzero(yb==1); neg=np.flatnonzero(yb==0); rng=np.random.default_rng(SEED)
    ix=np.r_[neg,rng.choice(pos,size=max(0,len(neg)-len(pos)),replace=True)]; rng.shuffle(ix)
    c6=LogisticRegression(max_iter=2000,random_state=SEED).fit(X[ix],yb[ix]); r6=Ridge(alpha=100).fit(X[ix],ytr[ix]); save('B6_RandomOversampling',c6.predict_proba(Xv)[:,1],r6.predict(Xv))
    # B7 discrepancy-weighted row sampling and severity-controlled variant; residuals are PID-aligned.
    for name,w in [('B7_DiscrepancySampling',weights),('B7_SeverityControlled',weights_ctrl)]:
        ix=rng.choice(np.arange(len(yb)),size=len(yb),replace=True,p=w/w.sum())
        c=LogisticRegression(max_iter=2000,random_state=SEED).fit(X[ix],yb[ix]); r=Ridge(alpha=100).fit(X[ix],ytr[ix]); save(name,c.predict_proba(Xv)[:,1],r.predict(Xv))
    # B8 hard-negative triplet representation
    cn8,z8=fit_contrastive(X,yb,ytr,hard_negative=True,epochs=80); cn8.eval()
    with torch.no_grad(): zv8=cn8(torch.tensor(Xv,dtype=torch.float32,device=DEVICE))[0].cpu().numpy()
    c8=LogisticRegression(max_iter=2000,random_state=SEED).fit(z8,yb); r8=Ridge(alpha=100).fit(z8,ytr); save('B8_HardNegativeTriplet',c8.predict_proba(zv8)[:,1],r8.predict(zv8))
    # B9 discrepancy-aware SupCon
    cn9,z9=fit_contrastive(X,yb,ytr,weights=weights,epochs=80); cn9.eval()
    with torch.no_grad(): zv9=cn9(torch.tensor(Xv,dtype=torch.float32,device=DEVICE))[0].cpu().numpy()
    c9=LogisticRegression(max_iter=2000,random_state=SEED).fit(z9,yb); r9=Ridge(alpha=100).fit(z9,ytr); save('B9_DA_SupCon',c9.predict_proba(zv9)[:,1],r9.predict(zv9))
    # B10 proposed: discrepancy-weighted multitask encoder + latent-space positive interpolation + weighted losses.
    net10=fit_torch_multitask(X,yb,ytr,sample_w=weights,epochs=120); net10.eval()
    with torch.no_grad(): ztr=net10(torch.tensor(X,dtype=torch.float32,device=DEVICE))[2].cpu().numpy(); zv=net10(torch.tensor(Xv,dtype=torch.float32,device=DEVICE))[2].cpu().numpy()
    z_aug,yb_aug,yc_aug,w_aug=latent_interpolate(ztr,yb,ytr,weights)
    # Probe the learned latent space on augmented examples; synthetic points exist only in training space.
    c10=LogisticRegression(max_iter=2000,random_state=SEED).fit(z_aug,yb_aug,sample_weight=w_aug)
    r10=Ridge(alpha=10).fit(z_aug,yc_aug,sample_weight=w_aug)
    prob10=c10.predict_proba(zv)[:,1]; reg10=r10.predict(zv)
    # Blend with multitask regression head; this is a fixed blend, not probability/score calibration.
    with torch.no_grad(): _,raw10,_=net10(torch.tensor(Xv,dtype=torch.float32,device=DEVICE)); raw10=raw10.cpu().numpy()
    save('B10_DA_Multitask_LatentInterpolation',prob10,.5*raw10+.5*reg10)

    res=pd.DataFrame(rows).sort_values('macro_f1',ascending=False)
    res.to_csv(OUT/'results_all_models.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/'participant_predictions.csv',index=False)
    summary={'status':'COMPLETED_NOT_SCIENTIFICALLY_VALIDATED','device':str(DEVICE),'n_train':len(ytr),'n_dev':len(ydev),
      'train_positive':int(yb.sum()),'dev_positive':int(ybd.sum()),'phq_cutoff':PHQ_CUTOFF,
      'oof_hsed_q75_threshold':tau_oof,'weights_pid_aligned':True,
      'best_macro_f1_model':str(res.iloc[0]['model_id']),'best_macro_f1':float(res.iloc[0]['macro_f1']),
      'best_mae_model':str(res.sort_values('mae').iloc[0]['model_id']),'best_mae':float(res['mae'].min()),
      'limitations':['Development cohort is small; metrics are exploratory.','No independent clinical masking labels exist.',
      'HSED candidates are model-defined residual cases, not diagnoses.','B10 interpolation is latent-space interpolation, not proof of a clinically distinct HSED phenotype.',
      'Model selection on the same dev set inflates apparent performance; use nested CV/repeated seeds for confirmatory claims.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=['# B0-B10 Experiment Results','',f"- Train N={len(ytr)}; dev N={len(ydev)}; train PHQ>=10={int(yb.sum())}; dev PHQ>=10={int(ybd.sum())}",f'- OOF q75 discrepancy threshold (training high-PHQ only): {tau_oof:.4f}','',res.to_markdown(index=False),'','## Interpretation guardrails','- HSED is a model-defined residual subgroup, not a clinical diagnosis.','- Do not claim success based only on a smaller mean discrepancy; compare to dummy and severity-controlled baselines.','- Dev set is small; report confidence intervals and repeated-seed/nested-CV results before strong conclusions.','- B10 uses a weighted multitask encoder and positive-class latent interpolation. It does not establish that HSED is a distinct clinical phenotype.']
    (OUT/'RESULTS.md').write_text('\n'.join(report),encoding='utf-8')
    log.info('Finished. Outputs: %s',OUT)

if __name__=='__main__': run()
