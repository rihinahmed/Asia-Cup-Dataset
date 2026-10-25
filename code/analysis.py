"""Asia Cup 2022 + 2025: chase / toss / venue effects and small-sample prediction.
Reproduces every number and figure in the manuscript.  Run: python analysis.py"""
import json, warnings, numpy as np, pandas as pd
from scipy import stats, optimize, special
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
SEED=42; rng=np.random.default_rng(SEED); R={}
SRC='data.xlsx'
m=pd.read_excel(SRC,'Master Dataset').sort_values(['Date','Match ID']).reset_index(drop=True)
m['Date']=pd.to_datetime(m['Date'])
VEN={'Dubai International Cricket Stadium':'Dubai','Sheikh Zayed Stadium':'Abu Dhabi','Sharjah Cricket Stadium':'Sharjah'}
m['V']=m['Venue'].map(VEN); m['Yr']=m['Year']
R['n_matches']=len(m); R['n_super_over']=int((m['Result Type']!='Normal').sum())

def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h,c+h)
def binom_p(k,n): return stats.binomtest(k,n,0.5).pvalue

# ---------- 1. descriptives
nn=m[m['Result Type']=='Normal'].copy()                    # exclude the Super Over tie for chase analysis
nn['y']=(nn['Winner Batted']=='Batting First').astype(int)  # 1 = batting-first side won
rows=[]
def desc(df,label):
    k=int((df['Winner Batted']=='Chasing').sum()); n=len(df); lo,hi=wilson(k,n)
    rows.append(dict(Group=label,Matches=len(df),Decided=n,ChaseWins=k,ChaseWinPct=100*k/n if n else np.nan,CI95_lo=100*lo,CI95_hi=100*hi,
        Mean1stInn=df['1st Innings Score'].mean(),Mean2ndInn=df['2nd Innings Score'].mean(),MeanPP1=df['Powerplay Run Team 1'].mean(),MeanTotal=df['Total Match Runs'].mean(),
        TossWinnerWon=int((df['Toss Winner Won Match']=='Yes').sum())))
desc(m,'All matches'); 
for s in ['Asia Cup 2022','Asia Cup 2025']: desc(m[m.Series==s],s)
for v in ['Dubai','Abu Dhabi','Sharjah']: desc(m[m.V==v],v)
for s in ['Group Stage','Super Four','Final']: desc(m[m.Stage==s],s)
for s,v in [('Asia Cup 2022','Dubai'),('Asia Cup 2025','Dubai')]: desc(m[(m.Series==s)&(m.V==v)],f'{v} {s[-4:]}')
D=pd.DataFrame(rows); R['desc']=D.round(2).to_dict('records')
# fix: desc counts 'Decided' as all rows; recompute decided (excl. Super Over) properly
def decided(df): return int((df['Result Type']=='Normal').sum())
D['Decided']=[decided(m)]+[decided(m[m.Series==s]) for s in ['Asia Cup 2022','Asia Cup 2025']]+[decided(m[m.V==v]) for v in ['Dubai','Abu Dhabi','Sharjah']]+[decided(m[m.Stage==s]) for s in ['Group Stage','Super Four','Final']]+[decided(m[(m.Series==s)&(m.V=='Dubai')]) for s in ['Asia Cup 2022','Asia Cup 2025']]
D['ChaseWinPct']=100*D['ChaseWins']/D['Decided']
D['CI95_lo']=[100*wilson(int(k),int(n))[0] for k,n in zip(D.ChaseWins,D.Decided)]
D['CI95_hi']=[100*wilson(int(k),int(n))[1] for k,n in zip(D.ChaseWins,D.Decided)]
R['desc']=D.round(2).to_dict('records')

# ---------- 2. chase advantage
k=int((nn.y==0).sum()); n=len(nn)
R['chase']=dict(k=k,n=n,pct=100*k/n,ci=[100*x for x in wilson(k,n)],p_exact=binom_p(k,n),
   post_mean=(k+1)/(n+2),post_p_gt_half=float(1-stats.beta.cdf(0.5,k+1,n-k+1)),
   post_ci=[float(stats.beta.ppf(q,k+1,n-k+1)) for q in (0.025,0.975)])
# 2022 vs 2025, Dubai vs other
def fisher(a,b):
    t=[[int((a.y==0).sum()),int((a.y==1).sum())],[int((b.y==0).sum()),int((b.y==1).sum())]]
    o,p=stats.fisher_exact(t); return dict(table=t,odds=float(o),p=float(p))
R['fisher_series']=fisher(nn[nn.Series=='Asia Cup 2022'],nn[nn.Series=='Asia Cup 2025'])
R['fisher_dubai_vs_other']=fisher(nn[nn.V=='Dubai'],nn[nn.V!='Dubai'])
R['fisher_group_vs_later']=fisher(nn[nn.Stage=='Group Stage'],nn[nn.Stage!='Group Stage'])
for key,g in [('2022',nn[nn.Series=='Asia Cup 2022']),('2025',nn[nn.Series=='Asia Cup 2025']),('Dubai',nn[nn.V=='Dubai'])]:
    kk=int((g.y==0).sum()); R['chase_'+key]=dict(k=kk,n=len(g),p=binom_p(kk,len(g)))

# ---------- 3. toss
kt=int((m['Toss Winner Won Match']=='Yes').sum()); R['toss']=dict(k=kt,n=len(m),pct=100*kt/len(m),ci=[100*x for x in wilson(kt,len(m))],p=binom_p(kt,len(m)))
fielded=m[m['Elected First']=='Bowl']; R['toss_field']=dict(share=100*len(fielded)/len(m),won=int((fielded['Toss Winner Won Match']=='Yes').sum()),n=len(fielded))
batted=m[m['Elected First']=='Bat']; R['toss_bat']=dict(won=int((batted['Toss Winner Won Match']=='Yes').sum()),n=len(batted))
# elect-to-field share by series
R['field_share_series']={s:float((m[m.Series==s]['Elected First']=='Bowl').mean()*100) for s in m.Series.unique()}

# ---------- 4. scoring
s1,s2=m['1st Innings Score'],m['2nd Innings Score']
w=stats.wilcoxon(s1,s2); R['innings']=dict(mean1=s1.mean(),mean2=s2.mean(),sd1=s1.std(),sd2=s2.std(),median1=s1.median(),median2=s2.median(),wilcoxon_p=float(w.pvalue),
    mean_diff=float((s1-s2).mean()))
bs=[ (rng.choice((s1-s2).values,len(m))).mean() for _ in range(5000)]; R['innings']['diff_ci']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
R['score_series']=dict(mw_p=float(stats.mannwhitneyu(s1[m.Series=='Asia Cup 2022'],s1[m.Series=='Asia Cup 2025']).pvalue),
   m22=float(s1[m.Series=='Asia Cup 2022'].mean()),m25=float(s1[m.Series=='Asia Cup 2025'].mean()))
dub=s1[m.V=='Dubai']; oth=s1[m.V!='Dubai']
R['score_venue']=dict(dub=float(dub.mean()),oth=float(oth.mean()),mw_p=float(stats.mannwhitneyu(dub,oth).pvalue),kruskal_p=float(stats.kruskal(*[s1[m.V==v] for v in ['Dubai','Abu Dhabi','Sharjah']]).pvalue))
pp=m['Powerplay Run Team 1']; lr=stats.linregress(pp,s1); rs=stats.spearmanr(pp,s1)
bsr=[]; 
for _ in range(5000):
    ix=rng.integers(0,len(m),len(m)); bsr.append(np.corrcoef(pp.values[ix],s1.values[ix])[0,1])
R['pp']=dict(pearson=float(lr.rvalue),pearson_ci=[float(np.nanpercentile(bsr,2.5)),float(np.nanpercentile(bsr,97.5))],spearman=float(rs.correlation),spearman_p=float(rs.pvalue),
   slope=float(lr.slope),intercept=float(lr.intercept),slope_se=float(lr.stderr),r2=float(lr.rvalue**2),p=float(lr.pvalue))
# does powerplay score of the chasing side separate winners?
pp2=m['Powerplay Run Team 2']
R['pp_chaser']=dict(winners_mean=float(pp2[m['Winner Batted']=='Chasing'].mean()),losers_mean=float(pp2[m['Winner Batted']=='Batting First'].mean()),
   mw_p=float(stats.mannwhitneyu(pp2[m['Winner Batted']=='Chasing'],pp2[m['Winner Batted']=='Batting First']).pvalue))
R['run_diff']=dict(chase_median_margin_runs=None)
# ---------- 5. Elo (sequential, no leakage)
FULL={'Afghanistan','Bangladesh','India','Pakistan','Sri Lanka'}
def elo_series(init_assoc,K=20):
    r={t:(1500 if t in FULL else init_assoc) for t in set(m['Team 1'])|set(m['Team 2'])}
    out=[]
    for _,x in m.iterrows():
        a,b=x['Team 1'],x['Team 2']; out.append(r[a]-r[b])
        ea=1/(1+10**((r[b]-r[a])/400)); sa=1.0 if x['Match Winner']==a else 0.0
        r[a]+=K*(sa-ea); r[b]-=K*(sa-ea)
    return np.array(out),r
m['elo_diff'],elo_final=elo_series(1350); m['elo_diff_flat'],_=elo_series(1500)
R['elo_final']={k:round(v,1) for k,v in sorted(elo_final.items(),key=lambda kv:-kv[1])}
nn=m[m['Result Type']=='Normal'].copy(); nn['y']=(nn['Winner Batted']=='Batting First').astype(int)
nn['ko']=(nn.Stage!='Group Stage').astype(int)
for v in ['Dubai','Abu Dhabi','Sharjah']: nn['v_'+v]=(nn.V==v).astype(int)
nn['score1']=nn['1st Innings Score']; nn['pp1']=nn['Powerplay Run Team 1']; nn['e1']=nn['Team 1 Extras']
F_PRE=['elo_diff','v_Dubai','v_Abu Dhabi','v_Sharjah','ko']; F_BRK=F_PRE+['score1','pp1','e1']
y=nn['y'].values
# ---------- 6. models
def models():
    return {'Majority class':DummyClassifier(strategy='prior'),
            'Logistic (L2)':make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=2000)),
            'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=3,min_samples_leaf=2,random_state=SEED),
            'Gradient Boosting':GradientBoostingClassifier(n_estimators=80,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=SEED)}
def metrics(yt,p,pred=None):
    p=np.clip(p,1e-3,1-1e-3); pred=(p>=0.5).astype(int) if pred is None else pred
    return dict(acc=accuracy_score(yt,pred),auc=roc_auc_score(yt,p) if (len(set(yt))>1 and len(set(np.round(p,6)))>1) else np.nan,logloss=log_loss(yt,p,labels=[0,1]),brier=brier_score_loss(yt,p))
def boot_ci(yt,p,fn,B=2000,pred=None):
    vals=[]; n=len(yt)
    for _ in range(B):
        ix=rng.integers(0,n,n)
        if len(set(yt[ix]))<2: continue
        vals.append(fn(yt[ix],p[ix]))
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]
acc_fn=lambda a,b:accuracy_score(a,(b>=0.5).astype(int)); br_fn=lambda a,b:brier_score_loss(a,np.clip(b,1e-3,1-1e-3)); auc_fn=roc_auc_score
cv_rows=[]; oof_store={}
for setname,feats in [('Post-toss',F_PRE),('Innings-break',F_BRK)]:
    X=nn[feats].values
    for name,mod in models().items():
        P=np.zeros((len(nn),0)); acc_all=[]
        rskf=RepeatedStratifiedKFold(n_splits=5,n_repeats=20,random_state=SEED)
        oof=np.zeros(len(nn)); cnt=np.zeros(len(nn))
        for tr,te in rskf.split(X,y):
            mod.fit(X[tr],y[tr]); p=mod.predict_proba(X[te])[:,1]; oof[te]+=p; cnt[te]+=1
        oof/=cnt; oof_store[(setname,name)]=oof; mt=metrics(y,oof)
        cv_rows.append(dict(Setting=setname,Model=name,Accuracy=mt['acc'],Acc_lo=boot_ci(y,oof,acc_fn)[0],Acc_hi=boot_ci(y,oof,acc_fn)[1],AUC=mt['auc'],Brier=mt['brier'],LogLoss=mt['logloss']))
# Elo baseline
elo_pred=(nn['elo_diff'].values>0).astype(int); elo_p=1/(1+10**(-nn['elo_diff'].values/400))
mt=metrics(y,elo_p,elo_pred); corr=(elo_pred==y).astype(float); bb=[corr[rng.integers(0,len(corr),len(corr))].mean() for _ in range(2000)]; lo,hi=float(np.percentile(bb,2.5)),float(np.percentile(bb,97.5))
cv_rows.append(dict(Setting='Post-toss',Model='Higher Elo wins (rule)',Accuracy=mt['acc'],Acc_lo=lo,Acc_hi=hi,AUC=mt['auc'],Brier=mt['brier'],LogLoss=mt['logloss']))
chase_acc=float((y==0).mean())
CV=pd.DataFrame(cv_rows); CV.loc[CV.Model=='Majority class','AUC']=np.nan; R['cv']=CV.round(3).to_dict('records'); R['always_chase_acc']=chase_acc
# leave-one-tournament-out
lt_rows=[]
for setname,feats in [('Post-toss',F_PRE),('Innings-break',F_BRK)]:
    for tr_s,te_s in [('Asia Cup 2022','Asia Cup 2025'),('Asia Cup 2025','Asia Cup 2022')]:
        tr=nn.Series==tr_s; te=nn.Series==te_s
        for name,mod in models().items():
            mod.fit(nn.loc[tr,feats].values,y[tr]); p=mod.predict_proba(nn.loc[te,feats].values)[:,1]
            mt=metrics(y[te],p); kk=int(round(mt['acc']*te.sum())); lo,hi=wilson(kk,int(te.sum()))
            lt_rows.append(dict(Setting=setname,Train=tr_s[-4:],Test=te_s[-4:],Model=name,n=int(te.sum()),Accuracy=mt['acc'],Acc_lo=lo,Acc_hi=hi,Brier=mt['brier'],LogLoss=mt['logloss']))
LT=pd.DataFrame(lt_rows); R['loto']=LT.round(3).to_dict('records')
# permutation test for logistic models
def perm_p(feats,B=300):
    X=nn[feats].values
    def cvacc(yy,seed):
        o=np.zeros(len(yy)); c=np.zeros(len(yy))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=seed).split(X,yy):
            mdl=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=2000)).fit(X[tr],yy[tr]); o[te]+=mdl.predict_proba(X[te])[:,1]; c[te]+=1
        return accuracy_score(yy,((o/c)>=0.5).astype(int))
    obs=cvacc(y,SEED); null=[cvacc(rng.permutation(y),i) for i in range(B)]
    return dict(obs=float(obs),p=float((1+sum(v>=obs for v in null))/(B+1)),null_mean=float(np.mean(null)))
R['perm_pre']=perm_p(F_PRE); R['perm_brk']=perm_p(F_BRK)
# innings-break logistic coefficients (standardised)
Xb=StandardScaler().fit_transform(nn[F_BRK].values); lrm=LogisticRegression(C=0.5,max_iter=2000).fit(Xb,y)
R['brk_coef']={f:float(c) for f,c in zip(F_BRK,lrm.coef_[0])}
Xp=StandardScaler().fit_transform(nn[F_PRE].values); lrp=LogisticRegression(C=0.5,max_iter=2000).fit(Xp,y); R['pre_coef']={f:float(c) for f,c in zip(F_PRE,lrp.coef_[0])}
# Elo sensitivity (flat 1500 start)
elo_pred2=(nn['elo_diff_flat'].values>0).astype(int); R['elo_flat_acc']=float(accuracy_score(y,elo_pred2))

# ---------- 7. Bayesian hierarchical logistic (random-walk Metropolis, numpy)
vcode=nn['V'].map({'Dubai':0,'Abu Dhabi':1,'Sharjah':2}).values; xe=(nn['elo_diff'].values/100.0); V=3
def logpost(th):
    mu,s,b=th[0],th[1],th[-1]; z=th[2:2+V]; tau=np.exp(s)
    a=mu+tau*z; eta=a[vcode]+b*xe
    ll=np.sum(y*eta-np.logaddexp(0,eta))
    lp=-0.5*(mu/1.5)**2-0.5*tau**2+s-0.5*np.sum(z**2)-0.5*b**2
    return ll+lp
d=2+V+1; x0=np.zeros(d)
opt=optimize.minimize(lambda t:-logpost(t),x0,method='BFGS'); H=np.zeros((d,d)); e=1e-4; f0=-logpost(opt.x)
for i in range(d):
    for j in range(d):
        ei=np.eye(d)[i]*e; ej=np.eye(d)[j]*e
        H[i,j]=(-logpost(opt.x+ei+ej)+logpost(opt.x+ei-ej)+logpost(opt.x-ei+ej)-logpost(opt.x-ei-ej))/(-4*e*e)*-1 if False else (-(logpost(opt.x+ei+ej)-logpost(opt.x+ei-ej)-logpost(opt.x-ei+ej)+logpost(opt.x-ei-ej)))/(4*e*e)
H=(H+H.T)/2; cov=np.linalg.inv(H+1e-6*np.eye(d)) if np.all(np.linalg.eigvalsh(H)>0) else np.eye(d)*0.1
L=np.linalg.cholesky(cov+1e-8*np.eye(d)); chains=[]; acc_rates=[]
for c in range(4):
    th=opt.x+0.5*rng.standard_normal(d); lpc=logpost(th); out=[]; acc=0; N=30000
    for it in range(N):
        prop=th+(2.38/np.sqrt(d))*L@rng.standard_normal(d); lpp=logpost(prop)
        if np.log(rng.random())<lpp-lpc: th,lpc=prop,lpp; acc+=1
        if it>=8000 and it%4==0: out.append(th.copy())
    chains.append(np.array(out)); acc_rates.append(acc/N)
Ch=np.array(chains)   # (4, n, d)
def rhat(x): 
    m_,n_=x.shape; W=x.var(axis=1,ddof=1).mean(); B=n_*x.mean(axis=1).var(ddof=1); return float(np.sqrt(((n_-1)/n_*W+B/n_)/W))
def ess(x):
    xc=x.reshape(-1); xc=xc-xc.mean(); n_=len(xc); ac=np.correlate(xc,xc,'full')[n_-1:]/ (xc.var()*n_); s=0
    for k in range(1,min(n_,2000)):
        if ac[k]<0.05: break
        s+=ac[k]
    return float(n_/(1+2*s))
names=['mu','log_tau']+[f'z_{v}' for v in ['Dubai','AbuDhabi','Sharjah']]+['b_elo']
R['mcmc']=dict(acc=[float(a) for a in acc_rates],rhat={n_:rhat(Ch[:,:,i]) for i,n_ in enumerate(names)},ess={n_:ess(Ch[:,:,i]) for i,n_ in enumerate(names)})
S=Ch.reshape(-1,d); tau=np.exp(S[:,1]); a_v=S[:,[0]]+tau[:,None]*S[:,2:2+V]
chase_v=1-special.expit(a_v)               # P(chasing side wins) at equal strength
R['bayes']=dict(mu_chase=[float(np.mean(1-special.expit(S[:,0]))),*[float(np.percentile(1-special.expit(S[:,0]),q)) for q in (2.5,97.5)]],
   tau=[float(tau.mean()),float(np.percentile(tau,2.5)),float(np.percentile(tau,97.5))],
   b=[float(S[:,-1].mean()),float(np.percentile(S[:,-1],2.5)),float(np.percentile(S[:,-1],97.5))],p_b_neg=float((S[:,-1]<0).mean()),
   p_b_pos=float((S[:,-1]>0).mean()),
   venues={v:dict(mean=float(chase_v[:,i].mean()),lo=float(np.percentile(chase_v[:,i],2.5)),hi=float(np.percentile(chase_v[:,i],97.5)),p_gt_half=float((chase_v[:,i]>0.5).mean())) for i,v in enumerate(['Dubai','Abu Dhabi','Sharjah'])},
   overall_p_chase_gt_half=float((1-special.expit(S[:,0])>0.5).mean()))
pd.DataFrame(S,columns=names).to_csv('posterior_samples.csv',index=False)

# ---------- 8. figures
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':200})
C1,C2,C3='#1f4e79','#c55a11','#7f7f7f'
# Fig1: chase-win rate with Wilson CI
G=[('All',nn),('2022',nn[nn.Series=='Asia Cup 2022']),('2025',nn[nn.Series=='Asia Cup 2025']),('Dubai',nn[nn.V=='Dubai']),('Abu Dhabi',nn[nn.V=='Abu Dhabi']),('Sharjah',nn[nn.V=='Sharjah']),('Group',nn[nn.Stage=='Group Stage']),('Super Four\n+ Final',nn[nn.Stage!='Group Stage'])]
fig,ax=plt.subplots(figsize=(3.45,1.95)); 
for i,(lab,g) in enumerate(G):
    kk=int((g.y==0).sum()); nn_=len(g); lo,hi=wilson(kk,nn_); ax.errorbar(i,100*kk/nn_,yerr=[[100*kk/nn_-100*lo],[100*hi-100*kk/nn_]],fmt='o',color=C1,capsize=2,ms=4)
    ax.text(i,102,f'{kk}/{nn_}',ha='center',fontsize=6)
ax.axhline(50,ls='--',color=C3,lw=0.8); ax.set_xticks(range(len(G))); ax.set_xticklabels([g[0] for g in G],rotation=45,ha='right',fontsize=6.5); ax.set_ylim(0,112); ax.set_ylabel('Chasing side won (%)')
fig.tight_layout(); fig.savefig('fig/fig1_chase_rate.png',dpi=300); plt.close()
# Fig2: first/second innings scores by series
fig,ax=plt.subplots(figsize=(3.45,1.95)); pos=0; xt=[]; xl=[]
for s in ['Asia Cup 2022','Asia Cup 2025']:
    g=m[m.Series==s]
    for j,(col,cc) in enumerate([('1st Innings Score',C1),('2nd Innings Score',C2)]):
        bp=ax.boxplot(g[col],positions=[pos],widths=0.6,patch_artist=True,showfliers=False); 
        for b_ in bp['boxes']: b_.set(facecolor=cc,alpha=0.35,edgecolor=cc)
        for k_ in ['whiskers','caps','medians']:
            for l_ in bp[k_]: l_.set(color=cc)
        ax.scatter(pos+rng.uniform(-0.15,0.15,len(g)),g[col],s=6,color=cc,zorder=3); xt.append(pos); xl.append(f"{s[-4:]}\n{'1st' if j==0 else '2nd'} inn."); pos+=1
    pos+=0.6
ax.set_xticks(xt); ax.set_xticklabels(xl,fontsize=6.5); ax.set_ylabel('Runs'); fig.tight_layout(); fig.savefig('fig/fig2_innings_scores.png',dpi=300); plt.close()
# Fig3: powerplay vs first-innings total
fig,ax=plt.subplots(figsize=(3.45,1.95)); 
for s,cc,mk in [('Asia Cup 2022',C1,'o'),('Asia Cup 2025',C2,'s')]:
    g=m[m.Series==s]; ax.scatter(g['Powerplay Run Team 1'],g['1st Innings Score'],s=14,color=cc,marker=mk,label=s[-4:])
xx=np.linspace(pp.min(),pp.max(),50); ax.plot(xx,lr.intercept+lr.slope*xx,color='k',lw=0.9)
ax.set_xlabel('Powerplay runs (batting first)'); ax.set_ylabel('First-innings total'); ax.legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig('fig/fig3_powerplay.png',dpi=300); plt.close()
# Fig4: posterior chase prob by venue
fig,ax=plt.subplots(figsize=(3.45,1.95)); 
for i,(v,cc) in enumerate(zip(['Dubai','Abu Dhabi','Sharjah'],[C1,C2,C3])):
    ax.hist(chase_v[:,i],bins=60,density=True,alpha=0.45,color=cc,label=v,histtype='stepfilled')
ax.axvline(0.5,ls='--',color='k',lw=0.8); ax.set_xlabel('P(chasing side wins), equal team strength'); ax.set_ylabel('Posterior density'); ax.legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig('fig/fig4_posterior.png',dpi=300); plt.close()
# Fig5: model comparison
fig,axs=plt.subplots(2,1,figsize=(3.45,3.15),sharey=True)
for ax,setn in zip(axs,['Post-toss','Innings-break']):
    t=CV[CV.Setting==setn].reset_index(drop=True)
    ax.errorbar(range(len(t)),t.Accuracy,yerr=[t.Accuracy-t.Acc_lo,t.Acc_hi-t.Accuracy],fmt='o',color=C1,capsize=2,ms=4)
    ax.axhline(chase_acc,ls='--',color=C3,lw=0.8); ax.axhline(0.5,ls=':',color=C3,lw=0.8)
    ax.set_xticks(range(len(t))); ax.set_xticklabels([{'Majority class':'Majority\nclass','Logistic (L2)':'Logistic\n(L2)','Random Forest':'Random\nForest','Gradient Boosting':'Gradient\nBoosting','Higher Elo wins (rule)':'Higher Elo\nwins (rule)'}[x] for x in t.Model],fontsize=6,rotation=0); ax.set_title(setn.replace('-',' '),fontsize=8); ax.set_ylim(0.2,1.05)
axs[0].set_ylabel('Accuracy (95% CI)'); axs[1].set_ylabel('Accuracy (95% CI)'); fig.tight_layout(); fig.savefig('fig/fig5_models.png',dpi=300); plt.close()
# ---------- save
def conv(o):
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,np.ndarray): return o.tolist()
    return str(o)
import math
def clean(o):
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    if isinstance(o,float) and math.isnan(o): return None
    return o
json.dump(clean(R),open('results.json','w'),indent=1,default=conv)
with pd.ExcelWriter('results_tables.xlsx') as xw:
    D.round(3).to_excel(xw,sheet_name='Descriptives',index=False); CV.round(3).to_excel(xw,sheet_name='CV models',index=False); LT.round(3).to_excel(xw,sheet_name='Cross-tournament',index=False)
    pd.DataFrame([{'Parameter':k,'Value':v} for k,v in R['elo_final'].items()]).to_excel(xw,sheet_name='Final Elo',index=False)
nn[['Match ID','Series','V','Stage','Team 1','Team 2','elo_diff','y']+F_BRK[5:]].to_csv('model_input.csv',index=False)
print('done'); 
