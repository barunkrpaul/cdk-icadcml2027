"""Figures from raw_results.csv."""
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
RES = _os.path.join(_ROOT, "results") + _os.sep
FIG = _os.path.join(_ROOT, "figs") + _os.sep
import os; os.makedirs(FIG, exist_ok=True)

SHORT = {"CDK (proposed)": "CDK", "ORACLE k-means(k*)": "oracle"}
def sh(m): return SHORT.get(m, m)

def main():
    df = pd.read_csv(RES + "raw_results.csv")
    df["family"] = df["family"].fillna("null")
    est = df[df.method != "ORACLE k-means(k*)"].copy()

    # --- Fig 1: k_hat vs k* scatter, CDK against four representative baselines
    panel = ["CDK (proposed)", "Silhouette", "Gap statistic", "X-means", "GMM+BIC"]
    panel = [m for m in panel if m in set(est.method)]
    fig, ax = plt.subplots(1, len(panel), figsize=(3.0 * len(panel), 3.2),
                           sharex=True, sharey=True)
    for a, m in zip(np.atleast_1d(ax), panel):
        g = est[est.method == m]
        jx = g.true_k + np.random.uniform(-.18, .18, len(g))
        jy = g.k_hat + np.random.uniform(-.18, .18, len(g))
        a.scatter(jx, jy, s=7, alpha=.35, lw=0)
        lim = [0, max(est.k_hat.max(), est.true_k.max()) + 1]
        a.plot(lim, lim, "k--", lw=.8)
        a.set_xlim(0, est.true_k.max() + 2); a.set_ylim(0, min(45, est.k_hat.max() + 2))
        a.set_title(sh(m), fontsize=10); a.set_xlabel("$k^*$")
    np.atleast_1d(ax)[0].set_ylabel(r"$\hat k$")
    fig.tight_layout(); fig.savefig(FIG + "fig1_khat_vs_k.png", dpi=160); plt.close(fig)

    # --- Fig 2: mean |err| by family, top methods
    fam = est.pivot_table(index="method", columns="family", values="abs_err",
                          aggfunc="mean")
    order = est.groupby("method").abs_err.mean().sort_values().index
    fam = fam.loc[order]
    fig, a = plt.subplots(figsize=(1.0 * len(fam.columns) + 4, .32 * len(fam) + 2))
    im = a.imshow(fam.values, aspect="auto", cmap="RdYlGn_r",
                  vmin=0, vmax=np.nanpercentile(fam.values, 95))
    a.set_xticks(range(len(fam.columns))); a.set_xticklabels(fam.columns, rotation=45,
                                                             ha="right", fontsize=8)
    a.set_yticks(range(len(fam))); a.set_yticklabels([sh(m) for m in fam.index], fontsize=8)
    for i in range(fam.shape[0]):
        for j in range(fam.shape[1]):
            v = fam.values[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, label=r"mean $|\hat k-k^*|$")
    a.set_title("Mean absolute error in $k$ by dataset family")
    fig.tight_layout(); fig.savefig(FIG + "fig2_family_heatmap.png", dpi=160); plt.close(fig)

    # --- Fig 3: exact-recovery rate, structured vs null
    st = est[est.true_k > 1].groupby("method").exact.mean() * 100
    nu = est[est.true_k == 1].groupby("method").exact.mean() * 100
    both = pd.DataFrame({"structured": st, "null (k*=1)": nu}).loc[order]
    fig, a = plt.subplots(figsize=(7, .3 * len(both) + 1.6))
    y = np.arange(len(both))
    a.barh(y - .2, both["structured"], .38, label="structured $k^*>1$")
    a.barh(y + .2, both["null (k*=1)"], .38, label="null $k^*=1$")
    a.set_yticks(y); a.set_yticklabels([sh(m) for m in both.index], fontsize=8)
    a.invert_yaxis(); a.set_xlabel(r"exact recovery of $k$  (%)"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG + "fig3_exact.png", dpi=160); plt.close(fig)

    # --- Fig 4: accuracy vs runtime
    s = est.groupby("method").agg(err=("abs_err", "mean"), t=("runtime", "mean"))
    fig, a = plt.subplots(figsize=(6, 4.4))
    a.scatter(s.t, s.err, s=28)
    for m, r in s.iterrows():
        a.annotate(sh(m), (r.t, r.err), fontsize=7,
                   xytext=(3, 3), textcoords="offset points")
    a.set_xscale("log"); a.set_xlabel("mean runtime per dataset (s, log)")
    a.set_ylabel(r"mean $|\hat k-k^*|$")
    a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG + "fig4_cost.png", dpi=160); plt.close(fig)
    print("figures written to", FIG)

if __name__ == "__main__":
    main()


def fig_inversion():
    """The paper's headline: mean rank under two benchmarks, same runs."""
    from scipy.stats import rankdata
    df = pd.read_csv(RES + "raw_results.csv")
    df["family"] = df["family"].fillna("null")
    est = df[df.method != "ORACLE k-means(k*)"]

    def ranks(sub):
        p = sub.pivot_table(index=["dataset", "rep"], columns="method",
                            values="abs_err").dropna()
        r = np.apply_along_axis(rankdata, 1, p.values)
        return pd.Series(r.mean(0), index=p.columns)

    a, b = ranks(est[est.true_k > 1]), ranks(est)

    def spread(vals, gap):
        """Nudge label positions apart, keeping order."""
        idx = np.argsort(vals); out = np.array(vals, float)
        for j in range(1, len(idx)):
            i0, i1 = idx[j - 1], idx[j]
            if out[i1] - out[i0] < gap:
                out[i1] = out[i0] + gap
        return out

    order = list(a.index)
    la = spread([a[m] for m in order], 0.20)
    lb = spread([b[m] for m in order], 0.20)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, m in enumerate(order):
        cdk = m.startswith("CDK")
        col = "#c1272d" if cdk else "#9aa0a6"
        tcol = "#c1272d" if cdk else "#4a4f54"
        ax.plot([0, 1], [a[m], b[m]], "-o", ms=4, lw=2.6 if cdk else 1.0,
                color=col, zorder=3 if cdk else 1)
        # leader lines from point to nudged label
        ax.plot([-0.02, -0.06], [a[m], la[i]], lw=.5, color="#c8ccd0", zorder=0)
        ax.plot([1.02, 1.06], [b[m], lb[i]], lw=.5, color="#c8ccd0", zorder=0)
        ax.text(-0.07, la[i], sh(m), ha="right", va="center",
                fontsize=9 if cdk else 8, color=tcol,
                fontweight="bold" if cdk else "normal")
        ax.text(1.07, lb[i], f"{sh(m)}  {b[m]:.2f}", ha="left", va="center",
                fontsize=9 if cdk else 8, color=tcol,
                fontweight="bold" if cdk else "normal")
    ax.set_xlim(-0.62, 1.72); ax.set_xticks([0, 1])
    ax.set_xticklabels(["clusters only\n($k^\\star>1$, 780 blocks)",
                        "including structureless data\n(1010 blocks)"],
                       fontsize=9.5)
    ax.invert_yaxis()
    ax.set_ylabel("mean rank on $|\\hat k - k^\\star|$  (lower is better)")
    ax.set_title("Same methods, same runs, one dataset family restored",
                 fontsize=10.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(FIG + "fig0_inversion.png", dpi=170)
    plt.close(fig)
    print("inversion figure written")


if __name__ == "__main__":
    fig_inversion()
