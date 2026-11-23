"""LaTeX tables for the conference paper, generated from the result CSVs."""
from __future__ import annotations
import pandas as pd, numpy as np, os

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
RES = _os.path.join(_ROOT, "results") + _os.sep
OUT = _os.path.join(_ROOT, "paper", "generated_tables") + _os.sep
os.makedirs(OUT, exist_ok=True)
C = "CDK (proposed)"
esc = lambda s: str(s).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
sh = lambda m: "\\textbf{CDK}" if m.startswith("CDK") else esc(m)


def load():
    df = pd.read_csv(RES + "raw_results.csv")
    df["family"] = df["family"].fillna("null")
    est = df[df.method != "ORACLE k-means(k*)"].copy()
    est["realnull"] = est.dataset.str.contains(r"\(1cls\)", regex=True)
    est["synnull"] = est.family == "null"
    est["struct"] = est.true_k > 1
    return df, est


def main():
    df, est = load()
    order = est.groupby("method").abs_err.mean().sort_values().index
    orc = df[(df.method == "ORACLE k-means(k*)") & (df.true_k > 1)].ARI.mean()

    # ---------------- Table 1: overall (two-column, table*) ---------------
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Headline results over all dataset and replicate blocks. "
         r"``struct.'' restricts to blocks with $k^\star>1$; ``syn.\ null'' to "
         r"the synthetic structureless blocks; ``real 1-class'' to the "
         r"single-class real extracts. The oracle running $K$-means at the true "
         r"$k$ attains ARI " + f"{orc:.3f}" + r" on the structured blocks. "
         r"Sorted by mean absolute error in $k$.}",
         r"\label{tab:overall}",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Method & $\overline{|\hat k-k^\star|}$ & exact all (\%) & "
         r"exact struct.\ (\%) & syn.\ null $\hat k{=}1$ (\%) & "
         r"real 1-class $\hat k{=}1$ (\%) & ARI all & ARI struct.\ & time (s) "
         r"\\ \midrule"]
    for m in order:
        g = est[est.method == m]
        sn = g[g.synnull].exact.mean() * 100
        rn = g[g.realnull].exact.mean() * 100
        L.append(f"{sh(m)} & {g.abs_err.mean():.2f} & {100*g.exact.mean():.1f} & "
                 f"{100*g[g.struct].exact.mean():.1f} & {sn:.0f} & {rn:.0f} & "
                 f"{g.ARI.mean():.3f} & {g[g.struct].ARI.mean():.3f} & "
                 f"{g.runtime.mean():.2f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    open(OUT + "tab_overall.tex", "w").write("\n".join(L))

    # ---------------- Table 2: structureless data -------------------------
    nu = est[est.synnull]
    piv = nu.pivot_table(index="method", columns="dataset", values="k_hat",
                         aggfunc="mean").loc[order]
    piv.columns = [c.replace("null_", "").replace("_", " ") for c in piv.columns]
    rn = est[est.realnull].groupby("method").k_hat.mean()
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Mean $\hat k$ on data with no cluster structure. The first "
         r"eight columns are synthetic nulls; the last is the mean over the "
         r"single-class real extracts. The correct answer everywhere is 1. "
         r"Every range-based method is searched over $k\in\{2,\dots,12\}$ and "
         r"is therefore unable to give that answer.}",
         r"\label{tab:null}",
         r"\begin{tabular}{l" + "r" * (len(piv.columns) + 1) + "}", r"\toprule",
         "Method & " + " & ".join(esc(c) for c in piv.columns) +
         r" & real 1-class \\ \midrule"]
    for m in order:
        L.append(sh(m) + " & " + " & ".join(f"{v:.1f}" for v in piv.loc[m].values)
                 + f" & {rn.get(m, float('nan')):.1f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    open(OUT + "tab_null.tex", "w").write("\n".join(L))

    # ---------------- Table 3: per-family error ---------------------------
    fam = est[~est.realnull].pivot_table(index="method", columns="family",
                                         values="abs_err", aggfunc="mean").loc[order]
    best = fam.min(0)
    L = [r"\begin{table*}[t]\centering\small",
         r"\caption{Mean $|\hat k-k^\star|$ by dataset family, excluding the "
         r"single-class extracts. Bold marks the best entry in each column.}",
         r"\label{tab:family}",
         r"\begin{tabular}{l" + "r" * len(fam.columns) + "}", r"\toprule",
         "Method & " + " & ".join(esc(c) for c in fam.columns) + r" \\ \midrule"]
    for m in order:
        cells = []
        for c in fam.columns:
            v = fam.loc[m, c]
            t = "--" if not np.isfinite(v) else f"{v:.2f}"
            if np.isfinite(v) and abs(v - best[c]) < 1e-9:
                t = "\\textbf{" + t + "}"
            cells.append(t)
        L.append(sh(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    open(OUT + "tab_family.tex", "w").write("\n".join(L))

    # ---------------- Table 4: ablation (single column) -------------------
    if os.path.exists(RES + "raw_ablation.csv"):
        ab = pd.read_csv(RES + "raw_ablation.csv")
        ab["family"] = ab["family"].fillna("null")
        ab = ab[ab.method.str.startswith("CDK")]
        rows = []
        for m, g in ab.groupby("method"):
            rows.append((m, g.abs_err.mean(), 100 * g[g.true_k > 1].exact.mean(),
                         100 * g[g.true_k == 1].exact.mean(), g.runtime.mean()))
        A = pd.DataFrame(rows, columns=["m", "mae", "exs", "nk", "t"]).sort_values("mae")
        L = [r"\begin{table}[t]\centering\small",
             r"\caption{Ablation. Each row removes or changes one design "
             r"choice. ``null'' is the exact recovery of $k=1$ on "
             r"structureless blocks.}", r"\label{tab:ablation}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Variant & $\overline{|\hat k-k^\star|}$ & exact struct.\ (\%) & "
             r"null (\%) \\ \midrule"]
        for _, r in A.iterrows():
            nm = "\\textbf{CDK (default)}" if r.m == C else esc(r.m.replace("CDK / ", ""))
            L.append(f"{nm} & {r.mae:.3f} & {r.exs:.1f} & {r.nk:.1f} \\\\")
        L += [r"\bottomrule\end{tabular}\end{table}"]
        open(OUT + "tab_ablation.tex", "w").write("\n".join(L))
    else:
        open(OUT + "tab_ablation.tex", "w").write("% ablation pending\n")
    print("paper tables written")


if __name__ == "__main__":
    main()
