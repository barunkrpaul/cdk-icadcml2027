"""Independent re-computation of every claim the paper makes.

This deliberately does NOT import mknums. It recomputes each quantity from the
raw CSV by a separate route and compares against the macro file, so a bug in
the macro generator does not silently propagate into the paper.
"""
from __future__ import annotations
import pandas as pd, numpy as np, re, sys

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
RES = _os.path.join(_ROOT, "results") + _os.sep
PAP = _os.path.join(_ROOT, "paper") + _os.sep
FAIL = []


def macros():
    out = {}
    for line in open(PAP + "numbers.tex"):
        m = re.match(r"\\newcommand\{\\n([A-Za-z]+)\}\{(.*)\}\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def fmt_round(x, nd):
    return f"{float(x):.{nd}f}"


def fmt(x, nd):
    """Match the paper's formatting exactly, so rounding is not a false alarm."""
    return f"{float(x):.{nd}f}"


def chk(name, claimed, recomputed, tol=1e-6):
    try:
        ok = abs(float(claimed) - float(recomputed)) <= tol
    except (TypeError, ValueError):
        ok = str(claimed).strip() == str(recomputed).strip()
    status = "ok " if ok else "FAIL"
    if not ok:
        FAIL.append(f"{name}: paper says {claimed!r}, recomputed {recomputed!r}")
    print(f"  [{status}] {name:16s} paper={claimed!r:>28}  recomputed={recomputed!r}")


def main():
    M = macros()
    df = pd.read_csv(RES + "raw_results.csv")
    df["family"] = df["family"].fillna("null")
    est = df[df.method != "ORACLE k-means(k*)"]
    C = "CDK (proposed)"
    cdk = est[est.method == C]

    print("\n== counts ==")
    chk("NBLOCK", M["NBLOCK"], len(est.groupby(["dataset", "rep"])))
    chk("NROWS", M["NROWS"], len(df))
    chk("NMETH", M["NMETH"], est.method.nunique())
    chk("NSTRUCT", M["NSTRUCT"], len(est[est.true_k > 1].groupby(["dataset", "rep"])))
    chk("NREALSET", M["NREALSET"], est[est.family == "real"].dataset.nunique())
    chk("NREALNULL", M["NREALNULL"],
        est[est.dataset.str.endswith("(1cls)")].dataset.nunique())

    print("\n== CDK headline ==")
    chk("CDKMAE", M["CDKMAE"], round(cdk.abs_err.mean(), 2), 5e-3)
    chk("CDKEXS", M["CDKEXS"],
        round(100 * cdk[cdk.true_k > 1].exact.mean(), 1))
    syn = cdk[cdk.dataset.str.startswith("null_")]
    chk("CDKNULL", M["CDKNULL"], round(100 * syn.exact.mean(), 1), 5e-2)
    rn = cdk[cdk.dataset.str.endswith("(1cls)")]
    chk("CDKRNULL", M["CDKRNULL"], round(100 * rn.exact.mean(), 1), 5e-2)
    chk("CDKTIME", M["CDKTIME"], round(cdk.runtime.mean(), 2), 5e-3)
    chk("ORCARI", M["ORCARI"],
        round(df[(df.method == "ORACLE k-means(k*)") & (df.true_k > 1)].ARI.mean(), 3),
        5e-4)

    print("\n== the rank inversion (recomputed by an independent route) ==")
    for tag, sub, pre in [("clusters only", est[est.true_k > 1], "ST"),
                          ("all blocks", est, "AL")]:
        # rank each block by hand rather than with apply_along_axis
        p = sub.pivot_table(index=["dataset", "rep"], columns="method",
                            values="abs_err").dropna()
        rk = p.rank(axis=1, method="average")
        mr = rk.mean(0).sort_values()
        print(f"  -- {tag} ({len(p)} blocks)")
        chk(pre + "RANK", M[pre + "RANK"], round(mr[C], 2), 5e-3)
        chk(pre + "POS", M[pre + "POS"], list(mr.index).index(C) + 1)
        chk(pre + "BESTM", M[pre + "BESTM"], mr.index[0])
        chk(pre + "NB", M[pre + "NB"], len(p))

    print("\n== range methods cannot return k=1 ==")
    rangey = ["Silhouette", "Calinski-Har.", "Davies-Bouldin", "Gap statistic",
              "Jump statistic", "Elbow (Kneedle)", "GMM+BIC", "GMM+ICL",
              "Bisecting+Sil.", "X-means", "ISODATA"]
    bad = [m for m in rangey if (est[est.method == m].k_hat == 1).any()]
    chk("no range method ever returns k=1", "none", ", ".join(bad) or "none")

    print("\n== single claims quoted in the text ==")
    n2 = est[est.dataset.str.startswith("null_")]
    chk("SILU", M["SILU"], fmt_round(n2[(n2.dataset == "null_uniform_d10") &
                                    (n2.method == "Silhouette")].k_hat.mean(), 1))
    chk("CHT", M["CHT"], fmt_round(n2[(n2.dataset == "null_t3_d2") &
                                  (n2.method == "Calinski-Har.")].k_hat.mean(), 1))
    s = est[est.family == "sep"]
    chk("SEPKtwenty", M["SEPKtwenty"],
        round(s[(s.method == C) & (s.true_k == 20)].k_hat.mean(), 1), 5e-2)
    chk("DIPSEPKten", M["DIPSEPKten"],
        round(s[(s.method == "dip-means") & (s.true_k == 10)].k_hat.mean(), 1), 5e-2)
    rl = est[(est.family == "real") & ~est.dataset.str.endswith("(1cls)")]
    chk("CDKREALMAE", M["CDKREALMAE"],
        round(rl[rl.method == C].abs_err.mean(), 2), 5e-3)
    chk("CDKREALARI", M["CDKREALARI"],
        round(rl[rl.method == C].ARI.mean(), 3), 5e-4)

    print("\n== Iris claim: CDK returns k=2 at every replicate ==")
    ir = cdk[cdk.dataset == "Iris"].k_hat.unique()
    chk("Iris k_hat set", "[2]", str(sorted(int(v) for v in ir)))

    print("\n== diagnostic ==")
    st = cdk[cdk.true_k > 1].dropna(subset=["sep_evidence"]).copy()
    st["col"] = (st.k_hat == 1).astype(int)
    chk("NCOLLAPSE", M["NCOLLAPSE"], int(st.col.sum()))
    chk("SEPEVCOL", M["SEPEVCOL"], round(st[st.col == 1].sep_evidence.mean(), 2), 5e-3)
    chk("SEPEVOK", M["SEPEVOK"], round(st[st.col == 0].sep_evidence.mean(), 2), 5e-3)

    print("\n== coverage ==")
    cv = ((cdk.conf_lo <= cdk.true_k) & (cdk.true_k <= cdk.conf_hi)).mean()
    chk("COV", M["COV"], round(cv, 3), 5e-4)

    print("\n== citation sanity: every \\cite key exists in refs.bib ==")
    tex = open(PAP + "paper.tex").read()
    keys = set(re.findall(r"\\cite\{([^}]*)\}", tex))
    keys = {k.strip() for grp in keys for k in grp.split(",")}
    have = set(re.findall(r"@\w+\{([^,]+),", open(PAP + "refs.bib").read()))
    missing = keys - have
    unused = have - keys
    chk("all cite keys defined", "none missing", ", ".join(sorted(missing)) or "none missing")
    if unused:
        print(f"  [note] unused bib entries: {', '.join(sorted(unused))}")

    print("\n" + "=" * 62)
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
