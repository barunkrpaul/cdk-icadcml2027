"""Tables, statistical tests and figures from raw_results.csv."""
from __future__ import annotations
import numpy as np, pandas as pd, itertools, warnings
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
warnings.filterwarnings("ignore")

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
RES = _os.path.join(_ROOT, "results") + _os.sep

def holm(pv):
    pv = np.asarray(pv, float); m = len(pv)
    o = np.argsort(pv); adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * pv[j]); adj[j] = min(run, 1.0)
    return adj

def main():
    df = pd.read_csv(RES + "raw_results.csv")
    df["family"] = df["family"].fillna("null")
    df["is_null"] = df.true_k == 1
    est = df[df.method != "ORACLE k-means(k*)"].copy()

    # ---------------- headline table --------------------------------------
    def agg(g):
        return pd.Series(dict(
            n_runs=len(g),
            mean_k=g.k_hat.mean(),
            mean_abs_err=g.abs_err.mean(),
            median_abs_err=g.abs_err.median(),
            mean_signed_err=g.signed_err.mean(),
            exact_pct=100 * g.exact.mean(),
            ARI=g.ARI.mean(),
            AMI=g.AMI.mean(),
            runtime_s=g.runtime.mean()))
    T = est.groupby("method").apply(agg).sort_values("mean_abs_err")

    # k=1 detection, computed only on the null family
    nu = est[est.is_null].groupby("method").exact.mean() * 100
    T["null_k1_pct"] = nu
    # structured-only performance
    st = est[~est.is_null].groupby("method").agg(
        exact_struct=("exact", lambda s: 100 * s.mean()),
        abserr_struct=("abs_err", "mean"),
        ARI_struct=("ARI", "mean"))
    T = T.join(st)
    orc = df[df.method == "ORACLE k-means(k*)"].ARI.mean()
    T["ARI_vs_oracle"] = T.ARI_struct / df[(df.method == "ORACLE k-means(k*)") &
                                           (df.true_k > 1)].ARI.mean()
    T.round(3).to_csv(RES + "table_overall.csv")

    # ---------------- per-family table ------------------------------------
    fam = est.pivot_table(index="method", columns="family", values="abs_err",
                          aggfunc="mean").round(2)
    fam.to_csv(RES + "table_family_abserr.csv")
    famx = est.pivot_table(index="method", columns="family", values="exact",
                           aggfunc=lambda s: 100 * s.mean()).round(1)
    famx.to_csv(RES + "table_family_exact.csv")

    # ---------------- statistical comparison ------------------------------
    # paired over (dataset, rep) on absolute error
    piv = est.pivot_table(index=["dataset", "rep"], columns="method",
                          values="abs_err")
    piv = piv.dropna()
    methods = list(piv.columns)
    stat, p = friedmanchisquare(*[piv[m].values for m in methods])
    lines = [f"Friedman over {len(piv)} paired blocks and {len(methods)} methods:"
             f" chi2={stat:.1f}, p={p:.3e}"]
    ranks = np.apply_along_axis(rankdata, 1, piv.values)
    mr = pd.Series(ranks.mean(0), index=methods).sort_values()
    lines.append("\nMean rank on |k_hat - k*| (lower is better):")
    for m, v in mr.items():
        lines.append(f"   {v:6.2f}  {m}")

    ref = "CDK (proposed)"
    if ref in methods:
        lines.append(f"\nPairwise Wilcoxon signed-rank against {ref},"
                     " Holm-adjusted (NOT Nemenyi; see Benavoli et al. 2016):")
        others = [m for m in methods if m != ref]
        raw = []
        for m in others:
            a, b = piv[ref].values, piv[m].values
            if np.all(a == b):
                raw.append(1.0)
            else:
                raw.append(wilcoxon(a, b, zero_method="zsplit").pvalue)
        adj = holm(raw)
        for m, r_, a_ in sorted(zip(others, raw, adj), key=lambda t: t[2]):
            d = piv[m].mean() - piv[ref].mean()
            verdict = ("CDK better" if d > 0 else "CDK worse") if a_ < .05 else "n.s."
            lines.append(f"   {m:22s} mean-diff={d:+.3f}  p_raw={r_:.2e}"
                         f"  p_holm={a_:.3e}   {verdict}")
    open(RES + "stats.txt", "w").write("\n".join(lines))
    print("\n".join(lines))
    return T, fam, famx

if __name__ == "__main__":
    T, fam, famx = main()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== OVERALL ===")
    print(T[["mean_abs_err", "exact_pct", "exact_struct", "null_k1_pct",
             "ARI", "ARI_struct", "runtime_s"]].round(3).to_string())
