"""Build a self-contained Colab notebook that reproduces every result.

Everything is embedded: the source files and the full raw result CSVs, gzipped
and base64-encoded. The notebook needs no downloads, no uploads and no setup.
"""
from __future__ import annotations
import base64, gzip, json, os

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
SRC = _HERE + _os.sep
RES = _os.path.join(_ROOT, "results") + _os.sep
OUT = _os.path.join(_ROOT, "notebooks") + _os.sep


def blob(path):
    return base64.b64encode(gzip.compress(open(path, "rb").read(), 9)).decode()


def _src(lines):
    """nbformat wants every line to keep its newline except the last."""
    txt = "\n".join(lines)
    parts = txt.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _src(lines)}


def main():
    files = {n: blob(SRC + n) for n in
             ["cdk.py", "baselines.py", "datasets.py", "run_experiments.py"]}
    data = {n: blob(RES + n) for n in ["raw_results.csv", "raw_ablation.csv"]}

    payload = json.dumps({"src": files, "data": data})
    cells = []

    cells.append(md(
        "# CDK: reproduce and check every result",
        "",
        "**Determining $k$ without a range, and what current benchmarks cannot"
        " measure.**",
        "",
        "This notebook is self-contained. The algorithm, the 15 baselines, the"
        " datasets and the complete raw results (17,170 runs) are embedded"
        " below, so nothing is downloaded and nothing needs uploading. Run the"
        " cells in order.",
        "",
        "What you can check here:",
        "",
        "1. The algorithm running live, including returning $k=1$ on noise.",
        "2. **The headline result**: the same 16 methods on the same runs"
        " reverse their ranking when structureless data is added.",
        "3. That no criterion-over-a-range method ever returned $k=1$, in any"
        " of the 17,170 runs.",
        "4. The full significance tests (Friedman, then Wilcoxon with Holm"
        " adjustment).",
        "5. Where the method fails, reported honestly.",
        "6. The ablation.",
        "7. Every number quoted in the paper, recomputed from the raw data.",
        "",
        "Runtime: about two minutes for everything except the optional sweep at"
        " the end."))

    cells.append(md("---\n## Setup\n\nWrites the source files and the raw"
                    " results to disk. Nothing leaves this machine."))
    cells.append(code(
        "#@title Embedded source and results (click the arrow to inspect)",
        "import base64, gzip, json, os, sys, textwrap",
        "",
        "PAYLOAD = \"\"\"" + payload + "\"\"\"",
        "",
        "P = json.loads(PAYLOAD)",
        "os.makedirs('cdk/src', exist_ok=True)",
        "os.makedirs('cdk/results', exist_ok=True)",
        "for n, b in P['src'].items():",
        "    open('cdk/src/' + n, 'wb').write(gzip.decompress(base64.b64decode(b)))",
        "for n, b in P['data'].items():",
        "    open('cdk/results/' + n, 'wb').write(gzip.decompress(base64.b64decode(b)))",
        "sys.path.insert(0, 'cdk/src')",
        "",
        "import warnings; warnings.filterwarnings('ignore')",
        "import numpy as np, pandas as pd",
        "pd.set_option('display.width', 200, 'display.max_columns', 40)",
        "",
        "def load(name='raw_results.csv'):",
        "    d = pd.read_csv('cdk/results/' + name)",
        "    # pandas reads the family label 'null' as a missing value",
        "    d['family'] = d['family'].fillna('null')",
        "    return d",
        "",
        "df  = load()",
        "est = df[df.method != 'ORACLE k-means(k*)'].copy()",
        "CDKNAME = 'CDK (proposed)'",
        "print(f'{len(df):,} rows, {est.method.nunique()} methods, "
        "{est.groupby([\"dataset\",\"rep\"]).ngroups} dataset/replicate blocks')",
        "print('files written to cdk/')"))

    # ---------------- part 1: live demo ----------------
    cells.append(md(
        "---",
        "## 1. The algorithm, running live",
        "",
        "`CDK().fit(X)` takes no `k`, no `k_max` and no candidate range. Watch"
        " the last row in particular: on data with no clusters it returns 1,"
        " which is the answer no validity index can give."))
    cells.append(code(
        "import time",
        "from cdk import CDK",
        "from datasets import make, SPECS",
        "",
        "rows = []",
        "cases = [('pure Gaussian noise, d=2',  ('null','demo_null',dict(d=2,kind='gaussian',n=600)), 1),",
        "         ('uniform noise, d=10',       ('null','demo_u',   dict(d=10,kind='uniform', n=600)), 1),",
        "         ('2 clusters, d=2',           ('sep','demo_k2',   dict(k=2, d=2, sep=8.0,n=800)), 2),",
        "         ('5 clusters, d=10',          ('sep','demo_k5',   dict(k=5, d=10,sep=8.0,n=800)), 5),",
        "         ('10 clusters, d=50',         ('sep','demo_k10',  dict(k=10,d=50,sep=8.0,n=800)), 10),",
        "         ('20 clusters, d=10',         ('sep','demo_k20',  dict(k=20,d=10,sep=8.0,n=800)), 20)]",
        "",
        "for label, spec, truth in cases:",
        "    X, y, tk = make(spec, 0)",
        "    t0 = time.time(); r = CDK(random_state=0).fit(X); el = time.time()-t0",
        "    rows.append(dict(dataset=label, true_k=truth, k_hat=r.k,",
        "                     leaves=r.k_init, tests=r.n_tests,",
        "                     seconds=round(el,1)))",
        "    print(f'{label:26s}  true k={truth:2d}   CDK k_hat={r.k:2d}   ({el:.1f}s)')",
        "",
        "pd.DataFrame(rows)"))

    cells.append(md(
        "For contrast, here is what Silhouette does on the same noise. It is"
        " searched over $k \\in \\{2,\\dots,12\\}$, which is the standard"
        " protocol, and so it cannot answer 1 no matter how strong the evidence"
        " for 1 is."))
    cells.append(code(
        "from baselines import REGISTRY",
        "X, y, tk = make(('null','demo_null',dict(d=2,kind='gaussian',n=600)), 0)",
        "ks = range(2, 13)",
        "for name in ['Silhouette','Calinski-Har.','Davies-Bouldin','Gap statistic']:",
        "    k_hat, _ = REGISTRY[name](X, ks, 0)",
        "    print(f'{name:18s} on pure noise reports k = {k_hat}')",
        "print()",
        "print('CDK on the same data reports k =', CDK(random_state=0).fit(X).k)"))

    # ---------------- part 2: the headline ----------------
    cells.append(md(
        "---",
        "## 2. The headline result: the benchmark decides the conclusion",
        "",
        "Below, the same 16 methods are ranked twice on **the same stored"
        " runs**. The only difference is which dataset/replicate blocks are"
        " allowed into the comparison.",
        "",
        "* First on blocks that contain clusters ($k^\\star > 1$), which is what"
        " published comparisons do.",
        "* Then on all blocks, adding the structureless ones back.",
        "",
        "No method is retrained, no parameter is retuned, no dataset is"
        " replaced."))
    cells.append(code(
        "from scipy.stats import rankdata, friedmanchisquare, wilcoxon",
        "",
        "def holm(pv):",
        "    pv = np.asarray(pv, float); m = len(pv); o = np.argsort(pv)",
        "    adj = np.empty(m); run = 0.0",
        "    for i, j in enumerate(o):",
        "        run = max(run, (m - i) * pv[j]); adj[j] = min(run, 1.0)",
        "    return adj",
        "",
        "def compare(sub, label):",
        "    piv = sub.pivot_table(index=['dataset','rep'], columns='method',",
        "                          values='abs_err').dropna()",
        "    mr = piv.rank(axis=1).mean().sort_values()",
        "    others = [m for m in piv.columns if m != CDKNAME]",
        "    raw = [wilcoxon(piv[CDKNAME].values, piv[m].values,",
        "                    zero_method='zsplit').pvalue for m in others]",
        "    adj = holm(raw)",
        "    beats  = sum(1 for m,a in zip(others,adj) if a<.05 and piv[m].mean()>piv[CDKNAME].mean())",
        "    beaten = sum(1 for m,a in zip(others,adj) if a<.05 and piv[m].mean()<piv[CDKNAME].mean())",
        "    return dict(benchmark=label, blocks=len(piv),",
        "                CDK_rank=round(mr[CDKNAME],2),",
        "                CDK_place=f'{list(mr.index).index(CDKNAME)+1} of {len(mr)}',",
        "                best_method=mr.index[0], beaten_by=beaten, beats=beats), mr",
        "",
        "r1, mr1 = compare(est[est.true_k > 1], 'clusters only (k*>1)')",
        "r2, mr2 = compare(est,                 'including structureless data')",
        "display(pd.DataFrame([r1, r2]))",
        "",
        "print()",
        "print('Full ranking under each benchmark (lower is better):')",
        "display(pd.DataFrame({'clusters only': mr1.round(2),",
        "                      'all blocks':    mr2.round(2)})",
        "        .assign(moved=lambda d: (d['clusters only']-d['all blocks']).round(2))",
        "        .sort_values('all blocks'))"))

    cells.append(md(
        "The ordering inverts. CDK goes from the lower half of the table to"
        " second place, and several validity indices move the other way, purely"
        " because one family of datasets was restored to the comparison.",
        "",
        "This argument does not depend on CDK being a good method. It would hold"
        " for any method that can return $k=1$."))

    # ---------------- part 3: k=1 ----------------
    cells.append(md(
        "---",
        "## 3. No range-based method ever returned $k=1$",
        "",
        "Not once, on any dataset, in any of the 17,170 runs. This is not a"
        " matter of tuning. A criterion evaluated on $\\{2,\\dots,k_{\\max}\\}$"
        " has no way to express the answer 1."))
    cells.append(code(
        "range_based = ['Silhouette','Calinski-Har.','Davies-Bouldin','Gap statistic',",
        "               'Jump statistic','Elbow (Kneedle)','GMM+BIC','GMM+ICL',",
        "               'Bisecting+Sil.','X-means','ISODATA']",
        "offenders = [m for m in range_based if (est[est.method==m].k_hat==1).any()]",
        "print('range-based methods that ever returned k=1:',",
        "      offenders if offenders else 'NONE (out of %d runs)' % len(est[est.method.isin(range_based)]))",
        "print()",
        "",
        "syn  = est[est.dataset.str.startswith('null_')]",
        "real = est[est.dataset.str.endswith('(1cls)')]",
        "tab = pd.DataFrame({",
        "  'mean k_hat, synthetic null': syn.groupby('method').k_hat.mean().round(1),",
        "  'correct (k=1) %':            (100*syn.groupby('method').exact.mean()).round(0),",
        "  'mean k_hat, real 1-class':   real.groupby('method').k_hat.mean().round(1),",
        "  'correct (k=1) %, real':      (100*real.groupby('method').exact.mean()).round(0),",
        "}).sort_values('correct (k=1) %', ascending=False)",
        "display(tab)",
        "",
        "print('Per-dataset detail on the synthetic nulls (the answer is 1 everywhere):')",
        "display(syn.pivot_table(index='method', columns='dataset',",
        "                        values='k_hat', aggfunc='mean').round(1))"))

    cells.append(md(
        "The real 1-class rows are the seven single-class extracts: all samples"
        " carrying one label from a standard dataset. They are real data known"
        " to come from one labelled population. They are not a guarantee of"
        " unimodality, since a labelled class can carry genuine substructure"
        " such as handwriting style, so the paper reports them separately from"
        " the synthetic nulls and reads them as a directional signal."))

    # ---------------- part 4: stats ----------------
    cells.append(md(
        "---",
        "## 4. Full significance tests",
        "",
        "Friedman across all methods, then pairwise Wilcoxon signed-rank against"
        " CDK with Holm adjustment. Deliberately **not** Nemenyi, whose"
        " mean-rank statistic depends on which other methods happen to be in the"
        " pool (Benavoli, Corani and Mangili, JMLR 17(5):1-10, 2016)."))
    cells.append(code(
        "def full_stats(sub, label):",
        "    piv = sub.pivot_table(index=['dataset','rep'], columns='method',",
        "                          values='abs_err').dropna()",
        "    chi, p = friedmanchisquare(*[piv[m].values for m in piv.columns])",
        "    print(f'--- {label}: {len(piv)} paired blocks, {len(piv.columns)} methods')",
        "    print(f'    Friedman chi2 = {chi:.1f}, p = {p:.3g}')",
        "    others = [m for m in piv.columns if m != CDKNAME]",
        "    raw = [wilcoxon(piv[CDKNAME].values, piv[m].values,",
        "                    zero_method='zsplit').pvalue for m in others]",
        "    adj = holm(raw)",
        "    out = pd.DataFrame({'method': others,",
        "        'mean_diff': [round(piv[m].mean()-piv[CDKNAME].mean(),3) for m in others],",
        "        'p_raw': [f'{x:.2e}' for x in raw],",
        "        'p_holm': [f'{x:.2e}' for x in adj],",
        "        'verdict': ['n.s.' if a>=.05 else",
        "                    ('CDK better' if piv[m].mean()>piv[CDKNAME].mean() else 'CDK worse')",
        "                    for m,a in zip(others,adj)]})",
        "    return out.sort_values('p_holm').reset_index(drop=True)",
        "",
        "print()",
        "a = full_stats(est[est.true_k>1], 'clusters only')",
        "display(a)",
        "print()",
        "b = full_stats(est, 'all blocks')",
        "display(b)"))

    cells.append(md(
        "On the clusters-only benchmark, seven methods beat CDK significantly."
        " The paper says so. Any submission claiming general superiority over"
        " validity indices would be contradicted by its own table."))

    # ---------------- part 5: failures ----------------
    cells.append(md(
        "---",
        "## 5. Where CDK fails",
        "",
        "Reported here as it is in the paper, not hidden in an appendix."))
    cells.append(code(
        "cdk = est[est.method == CDKNAME]",
        "",
        "print('Per-family mean |k_hat - k*| (lower is better):')",
        "fam = est[~est.dataset.str.endswith('(1cls)')].pivot_table(",
        "    index='method', columns='family', values='abs_err', aggfunc='mean')",
        "display(fam.loc[est.groupby('method').abs_err.mean().sort_values().index].round(2))",
        "",
        "print()",
        "print('Labelled real datasets: mean k_hat (ARI in brackets)')",
        "rl = est[(est.family=='real') & ~est.dataset.str.endswith('(1cls)')]",
        "kh = rl.pivot_table(index='method', columns='dataset', values='k_hat', aggfunc='mean')",
        "ar = rl.pivot_table(index='method', columns='dataset', values='ARI',  aggfunc='mean')",
        "combo = kh.round(1).astype(str) + ' (' + ar.round(2).astype(str) + ')'",
        "combo.loc['TRUE k'] = [str(rl[rl.dataset==c].true_k.iloc[0]) for c in kh.columns]",
        "display(combo)"))

    cells.append(code(
        "print('Cases where CDK collapses to k_hat=1 although k* > 1:')",
        "col = cdk[(cdk.k_hat==1) & (cdk.true_k>1)]",
        "print(f'  {len(col)} of {len(cdk[cdk.true_k>1])} structured blocks"
        " ({100*len(col)/len(cdk[cdk.true_k>1]):.1f}%)')",
        "display(col.groupby('dataset').agg(blocks=('k_hat','size'),",
        "                                   true_k=('true_k','first'),",
        "                                   sep_evidence=('sep_evidence','mean'),",
        "                                   p_median=('p_median','mean')).round(3))",
        "",
        "print()",
        "print('Two failure modes, both real:')",
        "print('  1. Overlapping clusters at 2-3 sigma separation. Arguably k=1 is defensible there.')",
        "print('  2. High-dimensional real data (Digits, Breast cancer) whose classes are')",
        "print('     contiguous regions of one manifold rather than separated populations.')",
        "print('     Diagnosed as loss of test POWER, not loss of calibration: on Digits the')",
        "print('     median edge p-value sits near 1, so the test has nothing to reject.')"))

    cells.append(md(
        "### The applicability diagnostic did not work",
        "",
        "The paper reports this as a refuted hypothesis rather than dressing it"
        " up. A single-dataset probe suggested the median p-value would predict"
        " the collapse cases. Over the full sweep it does not."))
    cells.append(code(
        "from sklearn.metrics import roc_auc_score",
        "s = cdk[cdk.true_k>1].dropna(subset=['sep_evidence']).copy()",
        "s['collapse'] = (s.k_hat==1).astype(int)",
        "pm = s.dropna(subset=['p_median'])   # p_median is undefined when no edge was tested",
        "",
        "print(f\"median p-value : collapse {pm[pm.collapse==1].p_median.mean():.3f}\"",
        "      f\"  vs  no collapse {pm[pm.collapse==0].p_median.mean():.3f}\"",
        "      f\"   AUC {roc_auc_score(pm.collapse, pm.p_median):.2f}   <- does NOT separate\")",
        "print(f\"rejection frac : collapse {s[s.collapse==1].sep_evidence.mean():.3f}\"",
        "      f\"  vs  no collapse {s[s.collapse==0].sep_evidence.mean():.3f}\"",
        "      f\"   AUC {roc_auc_score(s.collapse, -s.sep_evidence):.2f}   <- better, still not a test\")",
        "",
        "best = None",
        "for t in np.arange(.10, .80, .02):",
        "    pred = s.sep_evidence < t",
        "    tp = int((pred & (s.collapse==1)).sum())",
        "    fp = int((pred & (s.collapse==0)).sum())",
        "    fn = int(((~pred) & (s.collapse==1)).sum())",
        "    if tp + fp == 0: continue",
        "    pr, rc = tp/(tp+fp), tp/(tp+fn)",
        "    f1 = 2*pr*rc/max(pr+rc, 1e-9)",
        "    if best is None or f1 > best[3]: best = (t, pr, rc, f1)",
        "t, pr, rc, f1 = best",
        "print(f'\\nbest threshold {t:.2f}:  precision {pr:.2f}   recall {rc:.2f}   F1 {f1:.2f}')",
        "print('Three flags in four are false alarms, so it is a signal and not yet a test.')",
        "print('The confound: the rejection fraction is also low when k* is small, because')",
        "print('most candidate merges are then genuinely within-cluster and should not reject.')"))

    # ---------------- part 6: ablation ----------------
    cells.append(md(
        "---",
        "## 6. Ablation",
        "",
        "One design choice changed at a time, over 320 blocks."))
    cells.append(code(
        "ab = load('raw_ablation.csv')",
        "ab = ab[ab.method.str.startswith('CDK')]",
        "res = ab.groupby('method').apply(lambda g: pd.Series({",
        "    'mean |k_hat-k*|':   round(g.abs_err.mean(), 3),",
        "    'exact struct. (%)': round(100*g[g.true_k>1].exact.mean(), 1),",
        "    'null k=1 (%)':      round(100*g[g.true_k==1].exact.mean(), 1),",
        "    'time (s)':          round(g.runtime.mean(), 2)}), include_groups=False)",
        "display(res.sort_values('mean |k_hat-k*|'))",
        "",
        "base = ab[ab.method=='CDK (proposed)']; nomc = ab[ab.method=='CDK / no multiplicity corr.']",
        "d = 100*base[base.true_k==1].exact.mean() - 100*nomc[nomc.true_k==1].exact.mean()",
        "print(f'\\nRemoving the multiplicity correction costs {d:.0f} percentage points of')",
        "print('null detection. That is the type-I inflation the design predicts: with a mean')",
        "print('of 37 tests per dataset at nominal 0.05, P(>=1 false rejection) = 1-0.95^37 = 0.85.')",
        "print()",
        "print('The one choice the evidence does NOT support is the specific value n_min=25.')",
        "print('n_min=15 differs from it by less than replicate noise. The paper says so.')"))

    # ---------------- part 7: verify ----------------
    cells.append(md(
        "---",
        "## 7. Every number in the paper, recomputed",
        "",
        "No figure in the paper is typed by hand. This cell recomputes the"
        " quoted values from the raw data and compares them."))
    cells.append(code(
        "PAPER = {  # as printed in the paper",
        "  'rows in the result file':                    17170,",
        "  'dataset/replicate blocks':                   1010,",
        "  'blocks with k*>1':                            780,",
        "  'methods compared':                              16,",
        "  'real datasets':                                 15,",
        "  'single-class real extracts':                     7,",
        "  'CDK mean rank, clusters only':                8.53,",
        "  'CDK mean rank, all blocks':                   7.18,",
        "  'CDK exact recovery on structured blocks (%)': 60.1,",
        "  'CDK k=1 on synthetic nulls (%)':              88.8,",
        "  'CDK k=1 on real 1-class (%)':                 90.0,",
        "  'oracle ARI on structured blocks':            0.896,",
        "  'CDK mean seconds per dataset':                9.45,",
        "  'sensitivity-range coverage':                 0.734,",
        "}",
        "",
        "piv1 = est[est.true_k>1].pivot_table(index=['dataset','rep'], columns='method', values='abs_err').dropna()",
        "piv2 = est.pivot_table(index=['dataset','rep'], columns='method', values='abs_err').dropna()",
        "cdk  = est[est.method==CDKNAME]",
        "GOT = {",
        "  'rows in the result file':                    len(df),",
        "  'dataset/replicate blocks':                   est.groupby(['dataset','rep']).ngroups,",
        "  'blocks with k*>1':                           est[est.true_k>1].groupby(['dataset','rep']).ngroups,",
        "  'methods compared':                           est.method.nunique(),",
        "  'real datasets':                              est[est.family=='real'].dataset.nunique(),",
        "  'single-class real extracts':                 est[est.dataset.str.endswith('(1cls)')].dataset.nunique(),",
        "  'CDK mean rank, clusters only':               round(piv1.rank(axis=1).mean()[CDKNAME], 2),",
        "  'CDK mean rank, all blocks':                  round(piv2.rank(axis=1).mean()[CDKNAME], 2),",
        "  'CDK exact recovery on structured blocks (%)':round(100*cdk[cdk.true_k>1].exact.mean(), 1),",
        "  'CDK k=1 on synthetic nulls (%)':             round(100*cdk[cdk.dataset.str.startswith('null_')].exact.mean(), 1),",
        "  'CDK k=1 on real 1-class (%)':                round(100*cdk[cdk.dataset.str.endswith('(1cls)')].exact.mean(), 1),",
        "  'oracle ARI on structured blocks':            round(df[(df.method=='ORACLE k-means(k*)')&(df.true_k>1)].ARI.mean(), 3),",
        "  'CDK mean seconds per dataset':               round(cdk.runtime.mean(), 2),",
        "  'sensitivity-range coverage':                 round((((cdk.conf_lo<=cdk.true_k)&(cdk.true_k<=cdk.conf_hi)).mean()), 3),",
        "}",
        "",
        "chk = pd.DataFrame({'paper': PAPER, 'recomputed': GOT})",
        "chk['match'] = np.where(np.isclose(chk.paper.astype(float),",
        "                                   chk.recomputed.astype(float), atol=5e-3),",
        "                        'ok', 'MISMATCH')",
        "display(chk)",
        "print('\\nALL CHECKS PASS' if (chk.match=='ok').all() else '\\nSOME CHECKS FAILED')"))

    # ---------------- part 8: figures ----------------
    cells.append(md("---\n## 8. Figures\n\nRegenerated from the raw data."))
    cells.append(code(
        "import matplotlib.pyplot as plt",
        "",
        "def spread(vals, gap):",
        "    idx = np.argsort(vals); out = np.array(vals, float)",
        "    for j in range(1, len(idx)):",
        "        i0, i1 = idx[j-1], idx[j]",
        "        if out[i1] - out[i0] < gap: out[i1] = out[i0] + gap",
        "    return out",
        "",
        "short = lambda m: 'CDK' if m.startswith('CDK') else m",
        "order = list(mr1.index)",
        "la, lb = spread([mr1[m] for m in order], .20), spread([mr2[m] for m in order], .20)",
        "fig, ax = plt.subplots(figsize=(9, 6))",
        "for i, m in enumerate(order):",
        "    c = m.startswith('CDK')",
        "    ax.plot([0,1], [mr1[m], mr2[m]], '-o', ms=4, lw=2.6 if c else 1.0,",
        "            color='#c1272d' if c else '#9aa0a6', zorder=3 if c else 1)",
        "    ax.plot([-.02,-.06],[mr1[m],la[i]], lw=.5, color='#c8ccd0', zorder=0)",
        "    ax.plot([1.02,1.06],[mr2[m],lb[i]], lw=.5, color='#c8ccd0', zorder=0)",
        "    ax.text(-.07, la[i], short(m), ha='right', va='center',",
        "            fontsize=10 if c else 8.5, color='#c1272d' if c else '#4a4f54',",
        "            fontweight='bold' if c else 'normal')",
        "    ax.text(1.07, lb[i], f'{short(m)}  {mr2[m]:.2f}', ha='left', va='center',",
        "            fontsize=10 if c else 8.5, color='#c1272d' if c else '#4a4f54',",
        "            fontweight='bold' if c else 'normal')",
        "ax.set_xlim(-.62, 1.72); ax.set_xticks([0,1])",
        "ax.set_xticklabels(['clusters only\\n(k*>1, 780 blocks)',",
        "                    'including structureless data\\n(1010 blocks)'], fontsize=10)",
        "ax.invert_yaxis(); ax.set_ylabel('mean rank on |k_hat - k*|  (lower is better)')",
        "ax.set_title('Same methods, same runs, one dataset family restored')",
        "for s_ in ('top','right'): ax.spines[s_].set_visible(False)",
        "ax.grid(axis='y', alpha=.22); plt.tight_layout(); plt.show()"))

    cells.append(code(
        "st = est[est.true_k>1].groupby('method').exact.mean()*100",
        "nu = est[est.true_k==1].groupby('method').exact.mean()*100",
        "both = pd.DataFrame({'structured k*>1': st, 'structureless k*=1': nu}).loc[",
        "    est.groupby('method').abs_err.mean().sort_values().index]",
        "fig, ax = plt.subplots(figsize=(8, 5.2))",
        "y = np.arange(len(both))",
        "ax.barh(y-.2, both['structured k*>1'],    .38, label='structured  $k^*>1$')",
        "ax.barh(y+.2, both['structureless k*=1'], .38, label='structureless  $k^*=1$')",
        "ax.set_yticks(y); ax.set_yticklabels([short(m) for m in both.index], fontsize=9)",
        "ax.invert_yaxis(); ax.set_xlabel('exact recovery of k  (%)'); ax.legend(fontsize=9)",
        "ax.set_title('The missing orange bars are not poor scores.\\n"
        "They are methods that cannot return k=1 at all.', fontsize=10)",
        "plt.tight_layout(); plt.show()"))

    # ---------------- part 9: own sweep ----------------
    cells.append(md(
        "---",
        "## 9. Optional: run a fresh sweep yourself",
        "",
        "Everything above reads the stored results. This cell generates new data"
        " and runs the methods from scratch, so you can confirm the results are"
        " not an artefact of the stored file. A few minutes on a Colab CPU.",
        "",
        "Change `SPECS_TO_RUN` and `N_REPS` to taste. The full sweep in the"
        " paper is 1010 blocks and takes about two hours on two cores."))
    cells.append(code(
        "SPECS_TO_RUN = ['sep_k2_d2','sep_k5_d10','sep_k20_d10',",
        "                'overlap_s4.0','null_gaussian_d2','null_uniform_d10']",
        "N_REPS = 2",
        "",
        "from datasets import SPECS as ALL_SPECS",
        "from sklearn.metrics import adjusted_rand_score",
        "",
        "chosen = [s for s in ALL_SPECS if s[1] in SPECS_TO_RUN]",
        "out = []",
        "for spec in chosen:",
        "    for rep in range(N_REPS):",
        "        X, y, tk = make(spec, rep)",
        "        ks = range(max(2, tk-5), max(12, tk+5)+1)",
        "        t0 = time.time(); r = CDK(random_state=rep).fit(X)",
        "        out.append(dict(dataset=spec[1], rep=rep, true_k=tk,",
        "                        method='CDK', k_hat=r.k, secs=round(time.time()-t0,1)))",
        "        for nm in ['Silhouette','Calinski-Har.','Gap statistic','X-means','dip-means']:",
        "            t0 = time.time()",
        "            try: kk, _ = REGISTRY[nm](X, ks, rep)",
        "            except Exception: kk = np.nan",
        "            out.append(dict(dataset=spec[1], rep=rep, true_k=tk,",
        "                            method=nm, k_hat=kk, secs=round(time.time()-t0,1)))",
        "    print('done', spec[1], flush=True)",
        "",
        "fresh = pd.DataFrame(out)",
        "display(fresh.pivot_table(index=['dataset','true_k'], columns='method',",
        "                          values='k_hat', aggfunc='mean').round(1))"))

    cells.append(md(
        "---",
        "## What to take away",
        "",
        "The contribution that survives scrutiny is the evaluation point. A"
        " benchmark containing no $k^\\star=1$ case cannot distinguish a method"
        " that determines $k$ from one that ranks a range the user supplied, and"
        " almost every published benchmark is of that kind.",
        "",
        "CDK itself is a competent but not dominant method. On data that"
        " contains clusters it is beaten by seven of the fifteen competitors."
        " It fails on high-dimensional real data whose classes are not"
        " geometrically separated. Both facts are in the paper, in the abstract"
        " and in the limitations section.",
        "",
        "**Known gap before journal submission:** the sandbox this was run in"
        " had no access to OpenML or the UCI repository, so the real datasets"
        " are the four that ship with scikit-learn plus derived subsets and"
        " single-class extracts. Adding ten or so independent real datasets is"
        " the first thing to do."))

    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"name": "python3",
                                      "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}

    path = OUT + "CDK_reproduce_results.ipynb"
    json.dump(nb, open(path, "w"))
    print(f"wrote {path}  ({os.path.getsize(path)/1e6:.2f} MB, {len(cells)} cells)")


if __name__ == "__main__":
    main()
