# Annex — HITL Detailed Results

Supplementary tables and figures supporting §3.5.4. All values are from
gpt-5.6-luna (primary generator) on 250 tickets with strict LOO.

---

## A. Oracle Decile Analysis

Tickets stratified by baseline answer quality (cosine similarity to reference
reply). Decile 1 = weakest baseline (poorest baseline). Decile 10 = strongest
baseline (best baseline).

| Decile | n | Baseline cosine (mean) | Δ cosine (mean) | % Improved |
|--------|---|------------------------|-----------------|------------|
| 1 (weakest) | 25 | 0.218 | **+0.206** | 76% |
| 2 | 25 | 0.415 | +0.045 | 56% |
| 3 | 25 | 0.520 | +0.009 | 44% |
| 4 | 25 | 0.601 | −0.024 | 32% |
| 5 | 25 | 0.681 | −0.019 | 52% |
| 6 | 25 | 0.774 | −0.023 | 56% |
| 7 | 25 | 0.856 | −0.043 | 32% |
| 8 | 25 | 0.913 | +0.001 | 48% |
| 9 | 25 | 0.951 | −0.021 | 36% |
| 10 (strongest) | 25 | 0.979 | **−0.090** | 8% |

Δ = feedback_delta − baseline_delta.
Correlation r(baseline quality, Δ) = **−0.41**.

The pattern is near-monotonic: feedback helps the weakest baselines and hurts the
strongest. Both extremes are large in magnitude (D1 +0.21 vs. D10 −0.09), ruling
out a null effect masked by averaging. The full decile table is reproduced from
cell 26 of the experiments notebook.

---

## B. Full Per-Team Delta (M1 Global, n ≥ 5 tickets)

| Team | n | Mean δ | Direction |
|------|---|--------|-----------|
| (GI-UX) Group | 45 | +0.048 | Most benefited — diverse queries (VPN, passwords, screen savers) |
| (GI-UX) File & Print | 36 | +0.033 | Consistently benefits |
| (GI-IaaS) Account Management | 19 | +0.020 | Mild benefit |
| (GI-UX) Development Platforms | 13 | +0.013 | Mild benefit |
| (GI-UX) Network Access | 13 | +0.009 | Near-neutral |
| (GI-UX) General Support | 8 | +0.004 | Near-neutral |
| (GI-UX) Service Desk | 10 | −0.010 | Near-neutral |
| (GI-CF) Application Management | 11 | −0.012 | Mild harm |
| (GI-UX) Robot Process Autom. | 21 | −0.019 | Mild harm |
| (GI-UX) Database Administr. | 6 | −0.021 | Mild harm |
| (GI-SaaS) SAP | 9 | −0.028 | Mild harm |
| (GI-SaaS) Salesforce | 8 | **−0.116** | Most harmed — heterogeneous procedures under one label |

Teams not listed had fewer than 5 tickets.

Cross-scope stability: team delta rank-order from M1 correlates with M2 at r=0.87,
with M3 at r=0.69, and with M4 at r=0.55 (all n=12 teams with ≥5 tickets in all
four methods).

---

## C. Full Per-Type Delta (M1 Global)

| Ticket type | n | Mean δ |
|------------|---|--------|
| software_request | 24 | **+0.028** |
| admin_rights | 50 | **+0.024** |
| other | 107 | +0.000 |
| vpn_request | 61 | −0.009 |
| email_support | 8 | **−0.074** |

Types reflect the same pattern: reusable procedure templates (admin_rights,
software_request) benefit from feedback; heterogeneous procedures under similar
wording (email_support) are harmed.

---

## D. Cross-Validated Gate Cutoff Sweep

A 5-fold stratified cross-validation sweep over deployable signals. For each fold,
the optimal threshold was chosen on the training set and evaluated on the held-out
test set. Reported values are the CV mean across folds.

| Signal | CV Mean δ | Gate Open | vs. Always-On |
|--------|-----------|-----------|---------------|
| Always-on (no gate) | +0.004 | 100% | — |
| Top-1 FAISS | −0.006 | 29% | −0.010 |
| Top-5 FAISS mean | −0.003 | 34% | −0.007 |
| Retrieval margin (top1−top2 gap) | −0.006 | 40% | −0.010 |
| Top5-minus-Top1 FAISS spread | **+0.006** | 69% | +0.002 |

The best CV signal (top5−top1) reaches +0.006 vs. +0.004 always-on — within CV
noise (±0.03). No signal cleanly separates.

**In-sample optima** (same data for threshold selection and evaluation — report
for completeness, these are overfit):

| Signal | In-Sample Best δ |
|--------|------------------|
| Top-1 FAISS | +0.007 |
| Top-5 FAISS mean | +0.009 |
| Retrieval margin | +0.011 |
| Top5−Top1 spread | +0.008 |

---

## E. Rescue vs. Disruption Profile

Rescues: δ > +0.02 (n=70 tickets); Disruptions: δ < −0.02 (n=73 tickets);
Neutral: |δ| ≤ 0.02 (n=107 tickets).

| Group | n | Baseline cosine | Top-1 FAISS | Mean δ |
|-------|---|-----------------|-------------|--------|
| Rescues | 70 | 0.59 | 0.76 | +0.154 |
| Disruptions | 73 | 0.80 | 0.75 | −0.150 |
| Neutral | 107 | 0.69 | 0.76 | +0.000 |

Rescues and disruptions have near-identical top-1 FAISS scores (0.76 vs. 0.75).
Retrieval confidence cannot distinguish tickets that will benefit from those that
will be harmed. This is the concrete failure mode underlying the structural gap.

---

## F. Cross-Model Validation Summary

The oracle pattern (feedback helps weak baselines, hurts strong ones) holds in
direction across all three generators. Magnitude compresses as generator quality
improves (better models leave less room for feedback to help).

| Generator | Baseline cosine mean | Δ cosine mean | Oracle r | CV Gate Beat? |
|-----------|---------------------|---------------|----------|---------------|
| gpt-4o-mini | 0.610 | +0.012 | −0.39 | No |
| gpt-5.4-nano | 0.660 | +0.007 | −0.37 | No |
| **gpt-5.6-luna** | **0.691** | **+0.004** | **−0.41** | **No** |

Team delta rankings correlate r = 0.71–0.81 across generator pairs. The newest
generator (gpt-5.6-luna) is the most conservative test: it produces the strongest
baselines, leaving the least room for feedback, yet the oracle signal remains.

---

## G. Qualitative Examples (Most Extreme δ Values)

**R-163 — Strongest Rescue (δ = +0.85).** Ticket: "Opening .ica file doesn't
work" (File & Print team). Baseline retrieves Citrix troubleshooting neighbours
and produces a generic reply (cosine 0.13). Feedback re-ranks the correct
form-redirect candidate (already at FAISS rank 4) to rank 1. Generator reproduces
the software-installation form template verbatim (cosine 0.98). The delta was
achieved without any new candidate entering the top-5 — a pure re-ranking effect.

**R-143 — Worst Disruption (δ = −0.71).** Ticket: "Travel requisition IT02"
(Salesforce team). Baseline produces a sound reply about network authorisation
(cosine 0.83). Feedback promotes popular Salesforce candidates from deep ranks
about different business processes (user permissions, onboarding). Generator
abandons the network reply and follows the promoted templates (cosine 0.12).
Salesforce is the most consistently feedback-hostile team (mean δ = −0.116):
heterogeneous sub-procedures pooled under one team label make within-team
popularity an unreliable guide.

These are the most extreme cases in the dataset (mean δ = +0.004), chosen to
illustrate the mechanism rather than represent the typical ticket.

---

## H. Source Data

All values reproducible from:
- Evaluation JSON: `test_results_modular_looe/M1(global)_results_laplace_global_noGate_gpt56-luna_v3_--lift-multiplier0.80--tanh-sensitivity3.5/modular_looe_comprehensive_feedback_v3_laplace_global_none_20260801_154209_details.json`
- Analysis CSVs: `analysis_out_sikdd_gpt56luna/*.csv`
- Experiment notebook: `SIKDD_2026_experiments_gpt5.6-luna.ipynb`
- Feedback DB: `comprehensive_feedback_v3.db` (102,722 rows)
- Paper: `SIKDD_2026_paper_final.tex` (SIKDD 2026 submission)