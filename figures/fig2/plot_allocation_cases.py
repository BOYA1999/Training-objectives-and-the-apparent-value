from pathlib import Path
from collections import Counter
from io import BytesIO
import csv, hashlib, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lxml import etree
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import cairosvg

ROOT = Path(__file__).resolve().parent
STEM = 'CYP2C9_Assay_Allocation_Cases'
SEEDS = list(range(20261120, 20261125))
OBJECTIVES = ['point_l1', 'point_l2', 'interval']
REPS = ['FP', 'Mol2Mol', 'ChemGPT']
METHODS = [f'{r}_{o}' for o in OBJECTIVES for r in REPS]
COLORS = ['#436D86', '#B17C5B', '#778C6B']

def read(name):
    with (ROOT/'source_data'/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

pred = read('locked_predictions.csv')
labels = {r['candidate_id']: r for r in read('late_labels.csv')}
cases = json.loads((ROOT/'source_data/selection.json').read_text(encoding='utf8'))['cases']
assert len(pred) == 3750 and len(labels) == 750
scores, ranks, identities = {}, {}, {}
for seed in SEEDS:
    batch = [r for r in pred if int(r['seed']) == seed]
    assert len(batch) == 750
    for row in batch:
        cid = row['candidate_id']
        identities[cid] = row['identity']
        scores[seed, cid] = {m:float(row[m]) for m in METHODS}
        assert all(math.isfinite(v) for v in scores[seed, cid].values())
    for method in METHODS:
        ordered = sorted(batch, key=lambda r:(-float(r[method]),hashlib.sha256(r['identity'].encode()).hexdigest(),r['candidate_id']))
        ranks[seed, method] = {r['candidate_id']:i+1 for i,r in enumerate(ordered)}

def outcome(cid):
    return {'1.0':'hit', '0.0':'low', '':'unresolved'}[labels[cid]['hit']]

data = []
for n, case in enumerate(cases, 1):
    before, after = case['before_method'], case['after_method']
    a = {c for c in labels if ranks[SEEDS[0], before][c] <= 150}
    b = {c for c in labels if ranks[SEEDS[0], after][c] <= 150}
    entered, left = b-a, a-b
    assert Counter(outcome(c) for c in entered) == case['entered_labels']
    assert Counter(outcome(c) for c in left) == case['removed_labels']
    hit = sorted([c for c in entered if outcome(c)=='hit'], key=lambda c:(-ranks[SEEDS[0], after][c],hashlib.sha256(identities[c].encode()).hexdigest()))[0]
    low = sorted([c for c in left if outcome(c)=='low'], key=lambda c:(ranks[SEEDS[0], after][c],hashlib.sha256(identities[c].encode()).hexdigest()))[0]
    assert (hit,low) == (case['selected_hit_id'],case['selected_low_id'])
    for cid in (hit,low):
        for method in METHODS:
            assert ranks[SEEDS[0], method][cid] == case['compounds'][cid]['primary_ranks'][method]
        for seed in SEEDS:
            for method in METHODS:
                data.append(dict(case=n,candidate_id=cid,identity=identities[cid],outcome=outcome(cid),point=labels[cid]['point'],lower=labels[cid]['lower'],upper=labels[cid]['upper'],seed=seed,method=method,predicted_pIC50=scores[seed,cid][method],rank=ranks[seed,method][cid],selected=int(ranks[seed,method][cid]<=150)))
with (ROOT/'plotted_data.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

plt.style.use(ROOT/'academic.mplstyle')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'savefig.bbox':None,'svg.fonttype':'none','svg.hashsalt':'cyp2c9-allocation-20261123','axes.edgecolor':'#B8BEC2','axes.linewidth':0.65,'axes.grid':False})
fig = plt.figure(figsize=(7.2,8.0),facecolor='white')
fig.text(.5,.985,'Training objectives change the test shortlist',ha='center',va='top',fontsize=12.4,weight='bold')
fig.text(.5,.959,'Retrospective Octant CYP2C9 boundary cases  |  150 of 750  |  seed 20261120',ha='center',va='top',fontsize=8,color='#50585D')
ymax = min(750,50*math.ceil(max(ranks[s,m][cid] for c in cases for cid in c['compounds'] for s in SEEDS for m in METHODS)/50))
image_slots=[]
titles=['Absolute → squared FP','Squared → interval FP']
letters=['a','b','c','d']
for row,case in enumerate(cases):
    yhead=.927-row*.45
    gain=case['after_hits']-case['before_hits']
    fig.text(.5,yhead,f"{titles[row]}: {case['before_hits']} → {case['after_hits']} supported hits (+{gain})",ha='center',va='top',fontsize=9.4,weight='bold')
    e,l=case['entered_labels'],case['removed_labels']
    summary=f"{case['entered_count']} entered: {e['hit']} hit, {e['low']} below, {e['unresolved']} unresolved   |   {case['removed_count']} left: {l['hit']} hit, {l['low']} below, {l['unresolved']} unresolved"
    fig.text(.5,yhead-.021,summary,ha='center',va='top',fontsize=7.1,color='#535D63')
    for col,cid in enumerate((case['selected_hit_id'],case['selected_low_id'])):
        x=.09+col*.49
        head=.889-row*.45
        title='Newly selected supported hit' if col==0 else 'No longer selected: below threshold'
        fig.text(x-.025,head,f'{letters[row*2+col]}  {title}',fontsize=8.6,weight='bold',va='top')
        image_slots.append((cid,x-.025,.759-row*.45,.16,.109))
        xx=x+.145
        yy=.859-row*.45
        point,lower,upper=[float(labels[cid][k]) for k in ('point','lower','upper')]
        fig.text(xx,yy,cid,fontsize=8,weight='bold',va='top')
        fig.text(xx,yy-.023,'Measured pIC50 [95% CrI]',fontsize=7,va='top',color='#535D63')
        fig.text(xx,yy-.045,f'{point:.2f} [{lower:.2f}, {upper:.2f}]',fontsize=8,va='top')
        rb=ranks[SEEDS[0],case['before_method']][cid]
        ra=ranks[SEEDS[0],case['after_method']][cid]
        fig.text(xx,yy-.070,f'FP rank: {rb} → {ra}',fontsize=8,va='top',weight='bold')
        ax=fig.add_axes([x,.55-row*.45,.36,.192])
        ax.axhspan(1,150,color='#EEF2F4',zorder=0)
        ax.axhline(150,color='#6E7C85',ls=(0,(4,3)),lw=.85,zorder=1)
        for j,obj in enumerate(OBJECTIVES):
            for k,rep in enumerate(REPS):
                method=f'{rep}_{obj}'; xpos=j+(k-1)*.19
                for q,seed in enumerate(SEEDS[1:]):
                    ax.scatter(xpos+(q-1.5)*.025,ranks[seed,method][cid],marker='x',s=12,color=COLORS[k],alpha=.48,lw=.65,zorder=3)
                rank=ranks[SEEDS[0],method][cid]
                ax.scatter(xpos,rank,s=40,marker='o',facecolor=COLORS[k] if rank<=150 else 'white',edgecolor=COLORS[k],lw=1.25,zorder=5)
        ax.set(xlim=(-.45,2.45),ylim=(ymax,1),xticks=[0,1,2],xticklabels=['Absolute\npoint','Squared\npoint','Interval'])
        ax.set_yticks([v for v in (1,150,300,450,600,750) if v<=ymax])
        ax.tick_params(axis='both',labelsize=7.5,pad=3)
        ax.grid(axis='y',color='#E6E9EB',lw=.45,zorder=-1)
        ax.set_ylabel('Rank among 750 (1 = highest)',fontsize=7.5,labelpad=3)
        for t in ax.get_yticklabels():
            if t.get_text()=='150': t.set_weight('bold')
        ax.text(.98,.96,'Top 150',transform=ax.transAxes,ha='right',va='top',fontsize=7.1,color='#62737D')

handles=[Line2D([],[],marker='o',ls='',color=c,label=n,markersize=5) for c,n in zip(COLORS,['FP','FP + Mol2Mol','FP + ChemGPT'])]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.013),ncol=3,fontsize=8,handletextpad=.4,columnspacing=1.8)
state=[Line2D([],[],marker='o',ls='',color='#59666E',markerfacecolor='#59666E',label='Primary: selected'),Line2D([],[],marker='o',ls='',color='#59666E',markerfacecolor='white',label='Primary: outside'),Line2D([],[],marker='x',ls='',color='#89939A',label='Other four seeds')]
fig.legend(handles=state,loc='lower center',bbox_to_anchor=(.5,.032),ncol=3,fontsize=7.2,handletextpad=.4,columnspacing=1.2)
buf=BytesIO();fig.savefig(buf,format='svg',metadata={'Date':'2026-11-23'});plt.close(fig)
document=etree.fromstring(buf.getvalue())
view=[float(v) for v in document.get('viewBox').split()]
width,height=view[2:]
for cid,x,y,w,h in image_slots:
    mol=Chem.MolFromSmiles(identities[cid]);assert mol is not None
    rdDepictor.Compute2DCoords(mol)
    drawer=rdMolDraw2D.MolDraw2DSVG(300,210)
    opt=drawer.drawOptions();opt.useBWAtomPalette();opt.fixedFontSize=22;opt.bondLineWidth=1.8;opt.clearBackground=False;opt.padding=.045
    drawer.DrawMolecule(mol);drawer.FinishDrawing()
    chemical=etree.fromstring(drawer.GetDrawingText().encode())
    chemical.set('x',str(x*width));chemical.set('y',str((1-y-h)*height));chemical.set('width',str(w*width));chemical.set('height',str(h*height))
    chemical.set('viewBox','0 0 300 210')
    document.append(chemical)
svg=etree.tostring(document,xml_declaration=True,encoding='utf-8')
(ROOT/(STEM+'.svg')).write_bytes(svg)
cairosvg.svg2pdf(bytestring=svg,write_to=str(ROOT/(STEM+'.pdf')))
cairosvg.svg2png(bytestring=svg,write_to=str(ROOT/(STEM+'.png')),output_width=2160,output_height=2400)
high=cairosvg.svg2png(bytestring=svg,output_width=4320,output_height=4800)
Image.open(BytesIO(high)).convert('RGB').save(ROOT/(STEM+'.tiff'),compression='tiff_lzw',dpi=(600,600))
report={'data_rows':len(data),'cases':2,'individual_candidates':4,'methods':METHODS,'primary_seed':SEEDS[0],'other_seeds':SEEDS[1:],'slots':150,'candidate_pool':750,'all_selection_and_primary_rank_checks':'PASS','rank_axis_max':ymax,'rdkit':rdBase.rdkitVersion,'matplotlib':matplotlib.__version__,'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'source_data').iterdir()},'layout_reference':'Horwood et al. 2026, Fig. 6, doi:10.1038/s41467-026-79240-y','no_diffusion_trajectories_or_density_estimates':True}
(ROOT/'figure_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,indent=2))
