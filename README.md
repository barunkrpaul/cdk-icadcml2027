# CDK: Certified Divisive k

Code, raw results and paper source for

> B. Kr. Paul and K. Giri, **"Determining k Without a Range: A Certified Divisive
> Test and What Current Benchmarks Cannot Measure"**, 8th International
> Conference on Advances in Distributed Computing and Machine Learning
> (ICADCML 2027), NIT Puducherry, Karaikal, India. Springer Lecture Notes in
> Networks and Systems. Paper ID 139.

CDK determines the number of clusters with no candidate range and no `k_max`,
so it can return `k = 1` when the data carry no cluster structure. The paper
also shows that benchmarks made only of clustered data cannot tell such a
method apart from one that ranks a user-supplied range.

## Main result

The same 16 methods and the same runs are ranked under two benchmarks. Only
the set of blocks entering the comparison differs.

| Benchmark | Blocks | CDK mean rank | Beaten by | Beats |
|---|---|---|---|---|
| clusters only (k\* > 1) | 780 | 8.35 (9 of 16) | 7 | 7 |
| all blocks, incl. structureless data | 1010 | 7.00 (2 of 16) | 0 | 14 |

CDK returns `k = 1` on 93.8% of synthetic structureless blocks and on 90.0% of
real single-class blocks. Every range-based method scores 0% there by
construction.

## Repository layout

    src/cdk.py               the algorithm (scikit-learn style .fit)
    src/baselines.py         15 competing methods behind one interface
    src/datasets.py          43 synthetic specifications and 15 real datasets
    src/run_experiments.py   the full sweep
    src/analyze.py           result tables, Friedman and Holm-adjusted Wilcoxon tests
    src/figs.py              figures
    src/mknums.py            every number quoted in the paper, as LaTeX macros
    src/mkpapertables.py     table values (written to paper/generated_tables/)
    src/verify.py            recomputes every quoted number by a separate route
    src/mknotebook.py        builds the self-contained Colab notebook

    results/raw_results.csv  main sweep, 17,170 rows
    results/raw_ablation.csv ablation sweep (reduced subset of 320 blocks)
    results/stats.txt        significance tests
    results/table_*.csv      derived tables
    figs/                    figures
    notebooks/               self-contained reproduction notebook (Colab)
    paper/                   camera-ready LaTeX source (Springer llncs) and PDF

## Installation

    pip install -r requirements.txt

Step 1 below was last checked with Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, matplotlib 3.11.2.

## Using CDK

    import sys; sys.path.insert(0, "src")
    from cdk import CDK

    res = CDK().fit(X)     # no k, no k_max, no candidate range
    res.k                  # estimated number of clusters
    res.labels             # cluster labels
    res.pvals              # p-value for every candidate merge
    res.p_median, res.sep_evidence   # the diagnostic signals of Section 7.4

## Reproducing the paper

**Step 1. Check every number from the stored results (about one minute).**

    cd src
    python3 analyze.py
    python3 figs.py
    python3 mknums.py          # rewrites paper/numbers.tex
    python3 mkpapertables.py   # writes paper/generated_tables/
    python3 verify.py          # must end with "all checks passed"

`paper/numbers.tex` regenerated this way is identical to the one used in the
camera-ready paper. The table files in `paper/` hold the same values as
`paper/generated_tables/`, reformatted for the Springer single-column layout.

**Step 2. Build the paper.**

    cd ../paper
    pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper

**Step 3 (optional). Re-run the experiments from scratch.** The main sweep took
about 126 minutes on two cores.

    cd src
    OMP_NUM_THREADS=1 python3 run_experiments.py
    ABLATE=1 REPS=10 OUT=../results/raw_ablation.csv \
        ONLY=sep,null,nongauss,overlap python3 run_experiments.py

Then repeat Step 1. Seeds are fixed, and the data generator is seeded through a
fixed byte hash, so the synthetic datasets are byte-identical on any machine.

**Without a local install**, open `notebooks/CDK_reproduce_results.ipynb` in
Google Colab. It embeds the source and the raw results, recomputes the
headline results and runs CDK live.

## Limitations

See Section 8 of the paper. In brief, CDK does not beat strong classical
indices on data that contain clusters, it loses power on high-dimensional real
data with overlapping classes, it is slower than the baselines, and on real
single-class extracts its per-edge rejection rate is higher than a correct null
would produce. The failure diagnostic is preliminary, and the acceptance-level
sweep gives a stability interval and not a confidence set. The real datasets
are those shipped with scikit-learn, with derived subsets and single-class
extracts.

## Citation

See `CITATION.cff`. The proceedings DOI will be added after publication.

## Licence

MIT, see `LICENSE`.
