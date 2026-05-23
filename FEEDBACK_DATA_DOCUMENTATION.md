# Definitive Technical Manual: Feedback Intelligence & Retrieval Lift Infrastructure

This document provides a low-level, exhaustive specification of the data engineering pipeline used to generate the retrieval reranking datasets. It is designed to ensure that any researcher can audit the exact mathematical origin of every numerical value in the resulting CSV files.

---

## 1. Executive Summary of Data Lineage

The datasets are built from a multi-generational collection of **227 evaluation logs**. 
**IMPORTANT:** While the judge was disabled for many "Paper" runs to save cost, this pipeline performed a **Recursive Deep Scan**. It only extracted records where the `judge_scores` or `judge_scores_al` arrays were explicitly populated with LLM-assigned helpfulness values.

### **Valid Data Sources (Vetted for Judge Presence):**
*   **Early Era (`test_results_api/`):** Captured initial query-target interactions with heuristic and early GPT-3.5 judgements.
*   **Batch Era (`test_results_multi/`):** Captured the transition to multi-retrieval evaluations.
*   **Async Era (`test_results_multi_async/`):** The primary source of high-volume feedback, where the GPT-4o judge was active for hundreds of batches (e.g., `eval_1_results_20260429_123629.json`).
*   **Paper Validation Era (`test_results_paper/`):** Final end-to-end validation runs where the judge was re-enabled for gold-standard verification (e.g., `eval_999_results_20260518_211908.json`).

---

## 2. CSV Dataset Specifications

### **File 1: `pair_analytics_250x250.csv`**
This file contains the **entire history of interactions** among the 227 gold-standard probe queries and 250 targets.

#### **Section 1: The Raw Interaction (Observation)**
*   **`query_id`**: The ID of the ticket that acted as the "Input Question."
*   **`target_id`**: The ID of the Knowledge Base ticket that was retrieved as a potential answer.
*   **`pair_cosine_similarity`**: The **exact vector distance** between the `query_id` text and the `target_id` text (Recalculated via MiniLM).
*   **`judge_helpfulness_score` ($S_i$):** The raw quality score assigned by the LLM Judge ($0.0$ to $1.0$).
*   **`source_file`**: The specific JSON log file that provided the `judge_helpfulness_score`.

#### **Section 2: Categorical Context**
*   **`query_class`**: The canonical IT category of the user's problem (e.g., `vpn_request`).
*   **`query_team`**: The IT team assigned to handle that problem.
*   **`target_class`**: The canonical category of the Knowledge Base solution.
*   **`target_team`**: The IT team that wrote/owns the KB solution.

#### **Section 3: The "Reputation" Lifts (Calculated Intelligence)**
These columns append the **consolidated knowledge** the system has learned about the `target_id` across the entire project history.

*   **`global_target_lift_laplace`**: 
    *   *Mathematical Formula:* $Lift = ( \frac{\sum S_i + 1.0}{N_{global} + 2.0} - 0.5 ) \times 0.3$
    *   *Meaning:* The intrinsic quality of the target ticket across all users. It is "smoothed" to prevent low-sample tickets from getting extreme lifts.
*   **`global_target_lift_accum`**:
    *   *Mathematical Formula:* $Lift = \tanh( \frac{\sum (S_i - 0.5) \times \sigma_i}{5.0} ) \times 0.15$
    *   *Meaning:* The "Helpfulness Volume." It rewards tickets that have a high volume of positive interactions across the entire organization.

*   **`class_target_lift_laplace`**:
    *   *Mathematical Formula:* Identical to Global Laplace, but the $\sum$ and $N$ only count interactions where the `query_class` matches the current row's `query_class`.
    *   *Meaning:* How good this ticket is **for this specific type of problem**.
*   **`class_target_lift_accum`**:
    *   *Mathematical Formula:* Identical to Global Accum, but restricted to the current `query_class`.
    *   *Meaning:* The context-specific volume lift. This is the **Primary Signal** used by the Active Learning system to identify "Best-in-Class" solutions.

---

### **File 2: `pair_analytics_strict.csv`**
A scientifically filtered subset of the main CSV.

*   **Filter Logic:** Only contains rows where **`query_class == target_class`**.
*   **Purpose:** This represents **"Pure Retrieval Accuracy."** It removes the noise of cases where the system accidentally showed a VPN ticket to a user asking about Hardware. 
*   **Columns:** Exactly identical to the 250x250 CSV. By using this file, researchers can see how Active Learning ranks the best solutions when the intent category is already correctly identified.

---

## 3. Derived Statistical Logic (How it works in the system)

To replicate the retrieval behavior for the paper, the final score of a ticket is calculated as:
$$TotalScore = CosineSimilarity_{Pair} + Lift_{Target}$$

1.  **The Baseline:** Start with the `pair_cosine_similarity`.
2.  **The Context Check:** Identify the `query_class` of the incoming user ticket.
3.  **The Intelligence Injection:** Look up the `class_target_lift_accum` for that specific `target_id` and `query_class`.
4.  **The Rerank:** Add the lift to the similarity. This shifts the "vetted" high-quality solutions to the top of the list, even if their initial vector match was slightly lower than a noisy neighbor.

---

## 5. Audit Trail & Reproducibility
*   **Normalization:** All ticket IDs were converted to `R-XXXX` format.
*   **Classes:** Descriptive strings were normalized to a 10-class canonical set (VPN, Software, etc.) to ensure that feedback from similar user problems is grouped together.
*   **Precision:** All float values are rounded to **6 decimal places** to maintain high-fidelity results during reranking simulations.
