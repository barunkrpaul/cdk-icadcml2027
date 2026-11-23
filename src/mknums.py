"""Emit every number the paper quotes as a LaTeX macro, straight from the CSVs.

Nothing in the paper text is typed by hand. If a number changes when the sweep
is re-run, the paper changes with it.
"""
from __future__ import annotations
import pandas as pd, numpy as np, os

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
RES = _os.path.join(_ROOT, "results") + _os.sep
OUT = _os.path.join(_ROOT, "paper") + _os.sep
os.makedirs(OUT, exist_ok=True)
M = {}
esc = lambda x: str(x).replace('&', '\\&').replace(' (proposed)', '')


def put(k, v):
    M[k] = v


def main():
    df = pd.read_csv(RES + "raw_results.csv")
    df["family"] = df["family"].fillna("null")
    est = df[df.method != "ORACLE k-means(k*)"].copy()
    est["real"] = est.family == "real"
    # single-class real extracts are the real-null group
    est["realnull"] = est.dataset.str.contains(r"\(1cls\)", regex=True)
    est["synnull"] = (est.family == "null")
    est["struct"] = (est.true_k > 1)

    put("NBLOCK", f"{est.groupby(['dataset','rep']).ngroups}")
    put("NMETH", f"{est.method.nunique()}")
    put("NSYNSPEC", f"{est[~est.real].dataset.nunique()}")
    put("NREALSET", f"{est[est.real].dataset.nunique()}")
    put("NREALNULL", f"{est[est.realnull].dataset.nunique()}")
    put("NSTRUCT", f"{est[est.struct].groupby(['dataset','rep']).ngroups}")
    put("NNULLB", f"{est[~est.struct].groupby(['dataset','rep']).ngroups}")
    put("NROWS", f"{len(df)}")

    orc = df[(df.method == "ORACLE k-means(k*)") & (df.true_k > 1)].ARI.mean()
    put("ORCARI", f"{orc:.3f}")

    def g(m, mask=None):
        s = est[est.method == m]
        return s if mask is None else s[mask(s)]

    C = "CDK (proposed)"
    put("CDKMAE", f"{g(C).abs_err.mean():.2f}")
    put("CDKEX", f"{100*g(C).exact.mean():.1f}")
    put("CDKEXS", f"{100*g(C, lambda s: s.struct).exact.mean():.1f}")
    put("CDKNULL", f"{100*g(C, lambda s: s.synnull).exact.mean():.1f}")
    put("CDKARI", f"{g(C, lambda s: s.struct).ARI.mean():.3f}")
    put("CDKTIME", f"{g(C).runtime.mean():.2f}")
    put("CDKKINIT", f"{g(C).k_init.mean():.1f}")
    put("CDKNTEST", f"{g(C).n_tests.mean():.1f}")

    # ---- the central comparison: same methods, two benchmarks ----------
    from scipy.stats import rankdata, friedmanchisquare, wilcoxon

    def holm(pv):
        pv = np.asarray(pv, float); m = len(pv); o = np.argsort(pv)
        adj = np.empty(m); run = 0.0
        for i, j in enumerate(o):
            run = max(run, (m - i) * pv[j]); adj[j] = min(run, 1.0)
        return adj

    def compare(sub, pre):
        piv = sub.pivot_table(index=["dataset", "rep"], columns="method",
                              values="abs_err").dropna()
        rk = np.apply_along_axis(rankdata, 1, piv.values)
        mr = pd.Series(rk.mean(0), index=piv.columns).sort_values()
        put(pre + "RANK", f"{mr[C]:.2f}")
        put(pre + "POS", f"{list(mr.index).index(C)+1}")
        put(pre + "BESTM", esc(mr.index[0]))
        put(pre + "BEST", f"{mr.iloc[0]:.2f}")
        put(pre + "NB", f"{len(piv)}")
        st, p = friedmanchisquare(*[piv[m].values for m in piv.columns])
        put(pre + "CHI", f"{st:.0f}")
        pstr = ("<10^{-16}" if p < 1e-16 else
                f"{p:.0e}".replace("e-", r"\times 10^{-") + "}")
        put(pre + "P", "$" + pstr + "$")
        others = [m for m in piv.columns if m != C]
        raw = [1.0 if np.all(piv[C].values == piv[m].values) else
               wilcoxon(piv[C].values, piv[m].values,
                        zero_method="zsplit").pvalue for m in others]
        adj = holm(raw)
        nb = [m for m, a in zip(others, adj)
              if a < .05 and piv[m].mean() > piv[C].mean()]
        nw = [m for m, a in zip(others, adj)
              if a < .05 and piv[m].mean() < piv[C].mean()]
        ns = [m for m, a in zip(others, adj) if a >= .05]
        put(pre + "BETTER", f"{len(nb)}")
        put(pre + "WORSE", f"{len(nw)}")
        put(pre + "NS", f"{len(ns)}")
        put(pre + "NSLIST", ", ".join(esc(m) for m in sorted(ns)))
        put(pre + "WORSELIST", ", ".join(esc(m) for m in sorted(nw)))
        return mr

    compare(est[est.struct], "ST")     # structured blocks only
    compare(est, "AL")                 # every block
    put("NMETHM", f"{est.method.nunique()-1}")

    # who can express k=1 at all
    can1 = est[est.k_hat == 1].method.unique()
    put("NCANONE", f"{len(can1)}")
    nu = est[est.synnull].groupby("method").exact.mean() * 100
    def _nm(x):
        return x.replace("&", "\\&").replace(" (proposed)", "")
    put("CANONELIST", ", ".join(
        _nm(m) + " (" + f"{nu.get(m, 0):.0f}" + "\\%)"
        for m in sorted(can1, key=lambda x: -nu.get(x, 0))))
    # worst offenders on synthetic null
    n2 = est[est.synnull]
    for tag, ds, mth in [("SILU", "null_uniform_d10", "Silhouette"),
                         ("CHT", "null_t3_d2", "Calinski-Har.")]:
        v = n2[(n2.dataset == ds) & (n2.method == mth)].k_hat.mean()
        put(tag, f"{v:.1f}")

    # sep family by k
    s = est[est.family == "sep"]
    NUM = {2: "two", 3: "three", 5: "five", 10: "ten", 20: "twenty",
           50: "fifty"}
    for k in sorted(s.true_k.unique()):
        put(f"SEPK{NUM[int(k)]}", f"{s[(s.method==C)&(s.true_k==k)].k_hat.mean():.1f}")
        d_ = s[(s.method == "dip-means") & (s.true_k == k)]
        if len(d_):
            put(f"DIPSEPK{NUM[int(k)]}", f"{d_.k_hat.mean():.1f}")
    sd = s[s.method == C].copy()
    sd["dd"] = sd.dataset.str.extract(r"_d(\d+)").astype(int)
    for d in sorted(sd.dd.unique()):
        put(f"SEPD{NUM[int(d)]}", f"{sd[sd.dd==d].abs_err.mean():.2f}")

    # real: labelled vs single-class
    rl = est[est.real & ~est.realnull]
    rn = est[est.realnull]
    put("CDKREALMAE", f"{rl[rl.method==C].abs_err.mean():.2f}")
    put("CDKREALARI", f"{rl[rl.method==C].ARI.mean():.3f}")
    put("CDKRNULL", f"{100*rn[rn.method==C].exact.mean():.1f}")
    put("CDKRNULLK", f"{rn[rn.method==C].k_hat.mean():.1f}")
    for m in ["Silhouette", "Gap statistic", "Calinski-Har.", "HDBSCAN",
              "dip-means", "G-means", "X-means"]:
        key = m.replace(" ", "").replace("-", "").replace(".", "")
        sub = rn[rn.method == m]
        if len(sub):
            put("RN" + key, f"{sub.k_hat.mean():.1f}")
            put("RNEX" + key, f"{100*sub.exact.mean():.0f}")

    # applicability diagnostic.  NOTE: the single-dataset probe that motivated
    # this suggested the median p-value would separate the collapse cases.  Over
    # the full sweep it does not; the rejection fraction does better but is
    # still not precise enough to publish as a test.  Both numbers are reported
    # as measured.
    from sklearn.metrics import roc_auc_score
    c = est[est.method == C].copy()
    sflag = c[c.struct].dropna(subset=["sep_evidence"]).copy()
    sflag["collapse"] = (sflag.k_hat == 1).astype(int)
    put("NCOLLAPSE", f"{int(sflag.collapse.sum())}")
    put("NSTRUCTCDK", f"{len(sflag)}")
    put("AUCSEP", f"{roc_auc_score(sflag.collapse, -sflag.sep_evidence):.2f}")
    pm = sflag.dropna(subset=["p_median"])
    put("AUCPMED", f"{roc_auc_score(pm.collapse, pm.p_median):.2f}")
    put("PMEDCOL", f"{sflag[sflag.collapse == 1].p_median.mean():.2f}")
    put("PMEDOK", f"{sflag[sflag.collapse == 0].p_median.mean():.2f}")
    put("SEPEVCOL", f"{sflag[sflag.collapse == 1].sep_evidence.mean():.2f}")
    put("SEPEVOK", f"{sflag[sflag.collapse == 0].sep_evidence.mean():.2f}")
    best = None
    for t in np.arange(0.10, 0.80, 0.02):
        pred = sflag.sep_evidence < t
        tp = int((pred & (sflag.collapse == 1)).sum())
        fp = int((pred & (sflag.collapse == 0)).sum())
        fn = int(((~pred) & (sflag.collapse == 1)).sum())
        if tp + fp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        if best is None or f1 > best[3]:
            best = (t, pr, rc, f1)
    put("SEPTHR", f"{best[0]:.2f}")
    put("SEPPREC", f"{100*best[1]:.0f}")
    put("SEPREC", f"{100*best[2]:.0f}")
    pred = sflag.sep_evidence < best[0]
    put("SEPFLAGPCT", f"{100*pred.mean():.0f}")
    put("SEPCOLFLAG", f"{100*sflag.collapse[pred].mean():.0f}")
    put("SEPCOLUNFLAG", f"{100*sflag.collapse[~pred].mean():.1f}")

    # coverage of the sensitivity range
    c["covg"] = ((c.conf_lo <= c.true_k) & (c.true_k <= c.conf_hi)).astype(float)
    put("COV", f"{c.covg.mean():.3f}")
    put("COVW", f"{(c.conf_hi - c.conf_lo + 1).mean():.1f}")

    # runtime ratios
    tt = est.groupby("method").runtime.mean()
    put("RATIOGAP", f"{tt[C]/tt['Gap statistic']:.0f}")
    put("RATIOSIL", f"{tt[C]/tt['Silhouette']:.0f}")

    # ablation
    if os.path.exists(RES + "raw_ablation.csv"):
        ab = pd.read_csv(RES + "raw_ablation.csv")
        ab["family"] = ab["family"].fillna("null")
        ab = ab[ab.method.str.startswith("CDK")]
        for m, g_ in ab.groupby("method"):
            key = "ABL" + m.replace("CDK", "")
            for a, b in [("/", ""), (" ", ""), ("_", ""), ("=", ""), (".", ""),
                         ("(", ""), (")", ""), ("0", "zero"), ("1", "one"),
                         ("2", "two"), ("3", "three"), ("4", "four"),
                         ("5", "five"), ("6", "six"), ("7", "seven"),
                         ("8", "eight"), ("9", "nine"), ("+", "")]:
                key = key.replace(a, b)
            put(key + "MAE", f"{g_.abs_err.mean():.3f}")
            put(key + "NULL", f"{100*g_[g_.true_k==1].exact.mean():.1f}")
        put("NABL", f"{ab.method.nunique()-1}")
        base = ab[ab.method == C]
        put("CDKMAEabl", f"{base.abs_err.mean():.3f}")
        put("CDKNULLabl", f"{100*base[base.true_k==1].exact.mean():.1f}")
        nomc = ab[ab.method == "CDK / no multiplicity corr."]
        put("MULTDROP", f"{100*base[base.true_k==1].exact.mean() - 100*nomc[nomc.true_k==1].exact.mean():.0f}")
        put("NABLB", f"{ab.groupby(['dataset','rep']).ngroups}")

    with open(OUT + "numbers.tex", "w") as f:
        for k, v in sorted(M.items()):
            f.write("\\newcommand{\\n%s}{%s}\n" % (k, v))
    print(f"wrote {len(M)} macros to {OUT}numbers.tex")
    for k in sorted(M):
        print(f"  \\n{k} = {M[k]}")


if __name__ == "__main__":
    main()
