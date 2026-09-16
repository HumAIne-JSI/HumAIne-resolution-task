# Annex — HITL KPI Methodology

This document defines how every KPI value in §3.5.4 was computed.

## 1. Feedback Database Construction

The feedback database (`comprehensive_feedback_v3.db`) was built in three stages:

1. **Evaluation runs.** The full RAG pipeline was executed on batches of 250 query
   tickets across approximately 227 evaluation runs. Each run processed every query
   ticket and produced candidate retrieval sets.
2. **LLM judge scoring.** For each (query, candidate) pair, an LLM judge
   (`openai/gpt-4o-mini`, temperature 0) received the query title, description, the
   ground-truth expected first reply, and the candidate ticket's title and first
   reply. The prompt instructed the judge to reward only candidates whose reply is
   close in structure and content to the expected reply. Each candidate received a
   score in [0, 1].
3. **Aggregation.** Scores from all runs were merged into 102,722 scored rows (250
   queries, ~1,580 deduplicated physical candidates, 58,318 unique query--candidate
   pairs). A candidate is **positive** when its mean judge score is ≥ 0.80 and
   **negative** when ≤ 0.40; the middle band is neutral and ignored.

**Key design property:** the judge scored retrieval candidates (historical tickets),
not generated answers. This allows the feedback database to be built once and reused
across all routing scopes (M1–M4) and all generator models — only the retrieval
ranking changes.

## 2. Lift Formula

For candidate *c* and query *q*, let *pos* and *neg* be positive/negative counts,
*n* = *pos* + *neg*. The Laplace-corrected lift is:

```
p = (pos + α) / (pos + neg + α + β)
lift = (p − 0.5) × min(1, n/2) × m
```

with α = β = 1.0 (uniform Beta prior), m = 0.80 (multiplier), and C = 0.20 (cap).
The lift is clamped to ±0.20. The enhanced retrieval score is:

```
enhanced_score(c, q) = FAISS(c, q) + lift(c)
```

The top-5 by enhanced score become the feedback retrieval. The factor `min(1, n/2)`
shrinks the lift for candidates with fewer than two observations, ensuring untested
candidates stay at their FAISS score.

Three alternative formulas were compared (tanh shrinkage, Bayesian LCB) and rejected
[see SIKDD 2026 paper §4.2]. Laplace was chosen for: widest dynamic range (σ=0.090),
highest fidelity to judge scores (r=0.85), and weakest dependence on observation
count (r=0.19).

## 3. Routing Scopes

| Scope | Positive/Negative counts from | n | Notes |
|-------|-------------------------------|----|-------|
| M1 (global) | All 250 query tickets | 250 | Primary scope; largest evidence pool |
| M2 (team-only) | Only queries with the same resolver team | variable | Uses known team labels |
| M3 (type-only) | Only queries with the same intent label | variable | Uses known type labels |
| M4 (team∩type) | Queries sharing BOTH team and type | smallest | 33% of tickets receive zero nonzero lift |

For M2–M4, the query's own feedback rows are excluded from its own pool (strict LOO).

## 4. Evaluation Protocol

**Leave-one-out (LOO):** 250 folds. For each query *t*:
- Remove all feedback rows involving query *t* from the feedback pool.
- Exclude query *t*'s own ticket from the FAISS candidate pool.
- Keep the candidate pool and generator identical between baseline and feedback.
- The primary metric is the within-ticket delta:

  δ_cosine(t) = cosine(feedback_reply_t, ref_t) − cosine(baseline_reply_t, ref_t)

- Cosine correlates r > 0.85 with ROUGE-L and BERTScore.

**Gating simulation:** A gate routes feedback only to a subset of query tickets.
For a gate *g*: δ_gated = δ(t) if ticket t meets the gate criterion, else 0.
The aggregate metric is mean(δ_gated) across all 250 tickets.

**Bootstrap confidence intervals:** 10,000 resamples with replacement, reporting
the 2.5th and 97.5th percentiles.

**Generator:** `openai/gpt-5.6-luna-20260709` via OpenRouter API (standard default
temperature). Validation runs used gpt-4o-mini and gpt-5.4-nano.

**Embedding model:** `all-MiniLM-L6-v2` (384-dimensional). FAISS index with inner
product similarity over ~1,596 knowledge-base tickets.

**Judging metric:** All replies (baseline, feedback, reference) were embedded with
`multi-qa-MiniLM-L6-cos-v1` for cosine similarity computation. This model is
independent of the retrieval embedding model.

## 5. Gating Signal Definitions

| Signal | Computation | Available pre-generation? |
|--------|-------------|--------------------------|
| Top-1 FAISS | FAISS(c_1, query) — similarity of closest candidate | Yes |
| Top-5 FAISS mean | Mean of FAISS(c, query) for top-5 | Yes |
| Retrieval margin | FAISS(c_1, query) − FAISS(c_2, query) | Yes |
| Top5 − Top1 spread | Mean top-5 FAISS − Top-1 FAISS | Yes |
| Oracle: baseline answer quality | cosine(baseline_reply, reference) | No — requires reference reply |

## 6. Pool Analysis Definitions

**Structural score:** Binary features counting form-redirect markers
("below you will find", "form information"), bullet lists, numbered lists,
and field markers in the reference reply. Higher = more structured.

**Procedural mismatch:** Boolean — true when the query text contains problem
patterns ("cannot open", "not working", "doesn't work") AND the reference reply
contains form-redirect markers.

**Within-pool query diversity:** Mean pairwise Jaccard similarity of tokenized
ticket descriptions within a team or type pool. Higher Jaccard = more homogeneous
queries. Computed across pools with ≥5 tickets (n=12 teams).

## 7. Reproducibility

All results are reproducible from the repository:
- Run: `notebooks/SIKDD_2026_experiments_gpt5.6-luna.ipynb` (step-by-step)
- Pre-computed: `notebooks/test_results_modular_looe/M1(global)_results_laplace_global_noGate_gpt56-luna_v3_*/`
- Analysis: `notebooks/test_results_modular_looe/analysis_out_sikdd_gpt56luna/`
- Paper: `SIKDD_2026_paper_final.tex`

Environment: Python 3.12+, `sentence-transformers`, `faiss-cpu`, `scikit-learn`,
`sqlite3`, `numpy`, `pandas`, `matplotlib`. The feedback database
(`comprehensive_feedback_v3.db`, 102,722 rows) and knowledge base
(`tickets.db`, 1,596 tickets) are available on the project repository.