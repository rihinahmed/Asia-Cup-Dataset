"""Builds the two dataset-description figures (pipeline diagram and completeness chart)."""
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':7.5})
NAVY,ORG,GRN='#1f4e79','#c55a11','#2e7d32'
fig,ax=plt.subplots(figsize=(7.2,1.9)); ax.set_xlim(0,100); ax.set_ylim(0,36); ax.axis('off')
def box(x,y,w,h,txt,c,fc,fs=7,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.2',lw=1.0,edgecolor=c,facecolor=fc))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.2)
def arrow(a,b,c_,d): ax.add_patch(FancyArrowPatch((a,b),(c_,d),arrowstyle='-|>',mutation_scale=8,lw=0.9,color='#444'))
box(1,21,19,12,'2022 match sheet\n(13 matches)',NAVY,'#eaf0f8'); box(1,4,19,12,'2025 public CSV\n(19 matches)',NAVY,'#eaf0f8')
box(27,13,20,13,'Cleaning and merging\n(dates, toss, team\norder, units)',ORG,'#fbeee4'); box(27,0.5,20,8.5,'Pitch and weather\nnotes researched',ORG,'#fbeee4',fs=6.8)
box(54,13,18,13,'Master Dataset\n32 matches\n58 columns',GRN,'#e8f3e9',bold=True)
box(79,25,20,9.5,'Formula columns',GRN,'#e8f3e9',fs=6.8); box(79,13,20,9.5,'2025 scorecard detail',GRN,'#e8f3e9',fs=6.8); box(79,1,20,9.5,'Summary sheets',GRN,'#e8f3e9',fs=6.8)
arrow(20.7,27,26.5,22); arrow(20.7,10,26.5,17); arrow(37,9.6,37,12.4); arrow(47.6,19.5,53.4,19.5)
arrow(72.7,23,78.4,29); arrow(72.7,19.5,78.4,18); arrow(72.7,16,78.4,6)
fig.tight_layout(pad=0.2); fig.savefig('fig/fig_pipeline.png',dpi=300); plt.close()
F=json.load(open('dataset_facts.json'))['comp']; names=list(F.keys()); y=np.arange(len(names)); h=0.38
fig,ax=plt.subplots(figsize=(3.45,2.25))
v22=[F[n]['Asia Cup 2022'] for n in names]; v25=[F[n]['Asia Cup 2025'] for n in names]
ax.barh(y-h/2,v22,h,color=NAVY,label='2022'); ax.barh(y+h/2,v25,h,color=ORG,label='2025')
for i,b in enumerate(v25): ax.text(b+1.5,i+h/2,f'{b:.0f}%',va='center',fontsize=6)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=6.3); ax.invert_yaxis(); ax.set_xlim(0,115); ax.set_xlabel('Cells filled (%)',fontsize=7)
ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=6.5,loc='lower right')
fig.tight_layout(pad=0.3); fig.savefig('fig/fig_completeness.png',dpi=300); plt.close()
