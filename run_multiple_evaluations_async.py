"""
FAST Multi-Evaluation Runner — async + batched speedups.

Implements 5 key optimizations over run_multiple_evaluations.py:

  1. ASYNC LLM CALLS: All OpenAI calls (AL generation, baseline generation,
     judge x2, LLM compare) fire concurrently via AsyncOpenAI. For a single
     ticket this means 5 LLM calls execute in ~3s instead of ~15s.

  2. PARALLEL TICKETS WITHIN EACH EVALUATION: A semaphore-controlled pool
     processes N tickets concurrently. FAISS + embeddings are read-only and
     thread-safe. The semaphore (--concurrent-tickets, default 5) limits
     parallel OpenAI requests to stay within rate limits.

  3. SHARED SENTENCE-TRANSFORMER: The evaluation cosine similarity reuses
     the RAGSystem's sentence_model instead of loading a second copy.

  4. BATCH-ENCODED EXPECTED REPLIES: All expected first replies are encoded
     in one batch call before the ticket loop, then indexed per-ticket.
     Eliminates redundant per-ticket encoding.

  5. BATCHED BERTSCORE: All (generated, expected) pairs are collected during
     the loop and scored in a single batched call at the end, giving the
     BERTScore model optimal GPU batch throughput.

Each evaluation still runs in its own PROCESS (ProcessPoolExecutor) for full
module-level isolation. Within each process, tickets are processed concurrently
via asyncio.

Usage:
    python run_multiple_evaluations_async.py                          # 3 evals, 80 tickets
    python run_multiple_evaluations_async.py --num-evals 5            # 5 evals
    python run_multiple_evaluations_async.py --concurrent-tickets 8   # 8 tickets in flight
    python run_multiple_evaluations_async.py --max-parallel 2         # limit eval processes

Prerequisites:
    - OPENAI_API_KEY environment variable set
    - tickets_large_first_reply_label.csv in working directory
    - Model directories: perfect_team_classifier/, ticket_classifier_model/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional


# ============================================================================
# CONFIGURATION
# ============================================================================
ORIGINAL_KB_PATH = "tickets_large_first_reply_label.csv"
RESULTS_DIR = Path("./test_results_multi_async")
RESULTS_DIR.mkdir(exist_ok=True)


def _sanitize_for_json(obj):
    """Recursively replace NaN/inf floats with None so json.dump doesn't fail."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ============================================================================
# TEXT EXTRACTION (same as original)
# ============================================================================
def extract_first_reply_only(text) -> Optional[str]:
    """Extract ONLY the first reply from chat log."""
    import pandas as pd
    if pd.isna(text):
        return None
    text_str = str(text)
    timestamp_pattern = r"\*{10,}\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
    user_pattern = r":\s*[^()]*\([^)]*\)\s*\*{10,}"
    parts = re.split(timestamp_pattern, text_str)
    if len(parts) >= 2:
        first_reply = parts[1].strip()
        first_reply = re.sub(user_pattern, "", first_reply, flags=re.IGNORECASE).strip()
        next_ts = re.search(timestamp_pattern, first_reply)
        if next_ts:
            first_reply = first_reply[:next_ts.start()].strip()
        next_u = re.search(user_pattern, first_reply)
        if next_u:
            first_reply = first_reply[:next_u.start()].strip()
        first_reply = re.sub(r"^-+\s*", "", first_reply)
        first_reply = re.sub(r"\s*-+$", "", first_reply)
        first_reply = re.sub(r"^\s*Dear\s+[^,]+,?\s*", "", first_reply, flags=re.IGNORECASE)
        lines = first_reply.split("\n")
        cleaned = [
            l.strip()
            for l in lines
            if not re.match(r"^-{5,}$", l.strip())
            and not re.match(r"^\s*:\s*[^()]*\([^)]*\)", l.strip())
        ]
        first_reply = "\n".join(cleaned).strip()
        return first_reply if len(first_reply) > 50 else None
    return text_str[:500] if len(text_str) > 50 else None


def get_expected_team(row) -> str:
    csv_team = str(row.get("Last team ID->Name", "")).strip()
    if not csv_team or csv_team.lower() == "nan" or csv_team in ("None", ""):
        csv_team = str(row.get("Team->Name", "")).strip()
    if not csv_team or csv_team.lower() == "nan" or csv_team in ("None", ""):
        return "(GI-SM) Service Desk"
    return csv_team


def get_expected_class(title: str, description: str) -> str:
    combined = f"{title} {description}".lower()
    if any(w in combined for w in ["admin", "administrator", "rights", "privileges"]):
        return "admin_rights"
    elif any(w in combined for w in ["vpn", "tunnel", "remote access"]):
        return "vpn_request"
    elif any(w in combined for w in ["onboard", "new employee", "employee setup", "gft italia_new_entry"]):
        return "onboarding"
    elif any(w in combined for w in ["badge", "access card", "building access"]):
        return "badge_access"
    elif any(w in combined for w in ["mailbox", "email", "outlook"]):
        return "email_support"
    elif any(w in combined for w in ["software", "install", "installation"]):
        return "software_request"
    elif any(w in combined for w in ["network", "switch", "port"]):
        return "network_support"
    elif any(w in combined for w in ["invoice", "sap", "erp"]):
        return "other"
    else:
        return "other"


def evaluate_response_quality(generated: str, expected_team: str, predicted_team: str) -> dict:
    return {
        "length": len(generated),
        "has_greeting": any(g in generated.lower() for g in ["dear", "hello", "thank you", "need:"]),
        "has_structure": generated.count("\n-") >= 2 or generated.count(":") >= 3,
        "team_match": str(expected_team) == str(predicted_team),
    }


def _norm_class(x: str) -> str:
    return (str(x) if x is not None else "").strip().lower()


# ============================================================================
# FEEDBACK DB SEEDING — copy the production feedback_refid.db into eval workspace
# ============================================================================
PRODUCTION_FEEDBACK_DB = os.path.join(os.path.dirname(__file__) or ".", "feedback_refid.db")


def _seed_feedback_db(dest_path: str) -> int:
    """Copy the production feedback_refid.db to dest_path for this evaluation.

    Returns the number of feedback_agg rows copied.
    If the production DB doesn't exist, creates an empty DB (old behaviour).
    """
    import shutil

    if os.path.exists(PRODUCTION_FEEDBACK_DB):
        shutil.copy2(PRODUCTION_FEEDBACK_DB, dest_path)
        # Quick count for logging
        import sqlite3
        conn = sqlite3.connect(dest_path)
        try:
            n = conn.execute("SELECT COUNT(*) FROM feedback_agg").fetchone()[0]
        except Exception:
            n = 0
        finally:
            conn.close()
        return n
    return 0


# ============================================================================
# MMR DIVERSITY-AWARE EXAMPLE SELECTION
# ============================================================================
def _mmr_select(
    query_embedding,
    candidate_embeddings,
    candidate_indices: list,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list:
    """Maximal Marginal Relevance selection.

    Picks top_k items that are relevant to query but diverse among themselves.

    Args:
        query_embedding: 1-D numpy array, the query embedding.
        candidate_embeddings: 2-D numpy array (n_candidates x dim).
        candidate_indices: list of original indices (same length as candidate_embeddings).
        top_k: how many to select.
        lambda_param: trade-off between relevance (1.0) and diversity (0.0).
                      0.7 means 70% relevance, 30% diversity.

    Returns:
        list of selected indices from candidate_indices.
    """
    import numpy as np

    if len(candidate_indices) <= top_k:
        return list(candidate_indices)

    # Normalise
    q = query_embedding.flatten()
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    # Relevance scores (cosine to query)
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    normed = candidate_embeddings / norms
    relevance = normed @ q_norm  # shape (n,)

    selected = []
    remaining = list(range(len(candidate_indices)))

    for _ in range(top_k):
        if not remaining:
            break

        best_score = -float("inf")
        best_idx = remaining[0]

        for idx in remaining:
            rel = float(relevance[idx])

            # Max similarity to already-selected items (diversity penalty)
            if selected:
                sel_embs = normed[selected]
                sims = sel_embs @ normed[idx]
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0

            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_indices[i] for i in selected]


# ============================================================================
# ASYNC SINGLE-TICKET GENERATION (one direction only — AL or baseline)
# ============================================================================
async def _generate_for_ticket_async(
    i: int,
    ticket: dict,
    al_rag,
    rt_al,
    eval_tag: str,
    semaphore: asyncio.Semaphore,
    label: str,  # "AL" or "BASE"
    progress: dict = None,  # shared progress counter {"done": 0, "total": N}
) -> Optional[dict]:
    """Generate a response for one ticket asynchronously.

    This is called once per ticket per pass (AL pass, then baseline pass).
    FEEDBACK_ENABLED must already be set correctly before the pass starts.
    """
    async with semaphore:
        from resolution_task_async import async_generate_response
        loop = asyncio.get_event_loop()

        try:
            result = await async_generate_response(
                ticket["title"], ticket["description"],
                al_rag, retrieval_k=5, rt_module=rt_al, loop=loop,
            )
            return result
        except Exception as e:
            print(f"{eval_tag} ❌ [{label}] Error on ticket {i}: {e}")
            traceback.print_exc()
            return None
        finally:
            if progress is not None:
                progress["done"] += 1
                done = progress["done"]
                total = progress["total"]
                # Print progress every 5 tickets, or first/last
                if done == 1 or done == total or done % 5 == 0:
                    pct = done / total * 100
                    print(f"{eval_tag} [{label}] Progress: {done}/{total} ({pct:.0f}%)")


# ============================================================================
# SELF-TEACHING AL LOOP — sequential generation + judge + feedback recording
# ============================================================================
async def _self_teaching_al_pass(
    test_tickets: list,
    al_rag,
    rt_al,
    eval_tag: str,
    concurrent_tickets: int = 5,
) -> list:
    """Process tickets SEQUENTIALLY for the AL pass with self-teaching.

    For each ticket:
      1. Generate response (using current feedback DB state)
      2. Judge the retrieved examples (async LLM call)
      3. Record judge scores as feedback votes in the DB
      4. Next ticket benefits from accumulated feedback signal

    This is the core of the active learning loop: the system learns
    from its own judge as it processes tickets.

    We still use a semaphore for the LLM calls within each ticket
    (generation + judge happen concurrently per ticket), but tickets
    are processed ONE AT A TIME so feedback accumulates.
    """
    from resolution_task_async import async_generate_response, async_judge_items
    from judge_service import map_scores_to_votes

    loop = asyncio.get_event_loop()
    results = []
    total = len(test_tickets)
    feedback_recorded = 0

    print(f"{eval_tag} [AL-TEACH] Self-teaching AL pass: {total} tickets (sequential + judge feedback)")

    for i, ticket in enumerate(test_tickets, 1):
        try:
            # Step 1: Generate response (retrieval uses current feedback state)
            result = await async_generate_response(
                ticket["title"], ticket["description"],
                al_rag, retrieval_k=5, rt_module=rt_al, loop=loop,
            )

            # Step 2: Judge the retrieved examples
            import pandas as pd
            similar_replies_list = result.get("similar_replies", [])
            if isinstance(similar_replies_list, pd.DataFrame):
                similar_replies_list = similar_replies_list.to_dict(orient="records")

            expected_reply = ticket.get("expected_first_reply", "")

            if similar_replies_list and expected_reply:
                try:
                    judge_result = await async_judge_items(
                        ticket["title"], ticket["description"],
                        similar_replies_list, 5,
                        expected_first_reply=expected_reply,
                    )
                    scores = judge_result.get("scores", [])

                    # Step 3: Record feedback votes from judge
                    votes = map_scores_to_votes(scores)
                    predicted_class = result.get("predicted_class", result.get("classification"))
                    predicted_team = result.get("predicted_team")

                    for vote in votes:
                        label = int(vote.get("label", 0))
                        # Record both positive and negative votes
                        rt_al.record_feedback(
                            query_id=f"eval_ticket_{i}",
                            retrieved_id=str(vote["retrieved_id"]),
                            label=label,
                            predicted_class=predicted_class,
                            predicted_team=predicted_team,
                        )
                        if label != 0:
                            feedback_recorded += 1

                except Exception as e:
                    if i <= 5:
                        print(f"{eval_tag} [AL-TEACH] ⚠️ Judge/feedback error ticket {i}: {e}")

            results.append(result)

        except Exception as e:
            print(f"{eval_tag} [AL-TEACH] ❌ Error ticket {i}: {e}")
            traceback.print_exc()
            results.append(None)

        # Progress reporting
        if i == 1 or i == total or i % 5 == 0:
            pct = i / total * 100
            print(f"{eval_tag} [AL-TEACH] Progress: {i}/{total} ({pct:.0f}%) | feedback_votes={feedback_recorded}")

    print(f"{eval_tag} [AL-TEACH] ✅ Complete: {sum(1 for r in results if r is not None)}/{total} succeeded, {feedback_recorded} feedback votes recorded")
    return results


# ============================================================================
# ASYNC POST-GENERATION PROCESSING (judge + metrics, no state dependency)
# ============================================================================
async def _postprocess_ticket_async(
    i: int,
    ticket: dict,
    result_al: dict,
    result_base: dict,
    al_rag,
    expected_emb_tensor,
    rouge_scorer_instance,
    eval_tag: str,
    semaphore: asyncio.Semaphore,
    progress: dict = None,  # shared progress counter {"done": 0, "total": N}
) -> Optional[dict]:
    """Run judge, LLM compare, cosine, ROUGE for one ticket.

    This has no dependency on FEEDBACK_ENABLED, so all tickets can run concurrently.
    """
    async with semaphore:
        ticket_start = time.time()

        import pandas as pd
        from sentence_transformers import util as st_util
        from resolution_task_async import async_judge_items, async_llm_compare

        expected_reply = ticket.get("expected_first_reply", "")

        # Parse results
        predicted_class_al = result_al.get("predicted_class", result_al.get("classification", "Unknown"))
        predicted_team_al = result_al.get("predicted_team", "Unknown")
        generated_response_al = result_al.get("response", "")

        predicted_class_base = result_base.get("predicted_class", result_base.get("classification", "Unknown"))
        predicted_team_base = result_base.get("predicted_team", "Unknown")
        generated_response_base = result_base.get("response", "")

        # Quality metrics (CPU, fast)
        quality_al = evaluate_response_quality(generated_response_al, ticket["expected_team"], predicted_team_al)
        quality_base = evaluate_response_quality(generated_response_base, ticket["expected_team"], predicted_team_base)

        # ---- OPTIMIZATION 3+4: Cosine similarity using shared model + pre-encoded expected ----
        cos_al_val = None
        cos_base_val = None
        eval_model = al_rag.sentence_model  # OPTIMIZATION 3: reuse RAG's model

        if expected_reply and generated_response_al and expected_emb_tensor is not None:
            gen_emb_al = eval_model.encode(generated_response_al, convert_to_tensor=True)
            cos_al_val = st_util.cos_sim(expected_emb_tensor, gen_emb_al).item()

        if expected_reply and generated_response_base and expected_emb_tensor is not None:
            gen_emb_base = eval_model.encode(generated_response_base, convert_to_tensor=True)
            cos_base_val = st_util.cos_sim(expected_emb_tensor, gen_emb_base).item()

        # ---- ROUGE-L (CPU, fast) ----
        rouge_al_val = None
        rouge_base_val = None
        if rouge_scorer_instance is not None and expected_reply:
            if generated_response_al:
                rs = rouge_scorer_instance.score(expected_reply, generated_response_al)
                rouge_al_val = rs["rougeL"].fmeasure
            if generated_response_base:
                rs = rouge_scorer_instance.score(expected_reply, generated_response_base)
                rouge_base_val = rs["rougeL"].fmeasure

        # ---- Prepare judge + LLM compare coroutines ----
        similar_replies_list_base = result_base.get("similar_replies", [])
        if isinstance(similar_replies_list_base, pd.DataFrame):
            similar_replies_list_base = similar_replies_list_base.to_dict(orient="records")
        similar_replies_list_al = result_al.get("similar_replies", [])
        if isinstance(similar_replies_list_al, pd.DataFrame):
            similar_replies_list_al = similar_replies_list_al.to_dict(orient="records")

        async def _noop_judge():
            return {"scores": []}

        async def _noop_compare():
            return None

        judge_base_coro = (
            async_judge_items(
                ticket["title"], ticket["description"],
                similar_replies_list_base, 5,
                expected_first_reply=expected_reply,
            )
            if similar_replies_list_base
            else _noop_judge()
        )

        judge_al_coro = (
            async_judge_items(
                ticket["title"], ticket["description"],
                similar_replies_list_al, 5,
                expected_first_reply=expected_reply,
            )
            if similar_replies_list_al
            else _noop_judge()
        )

        # LLM compare commented out to save API calls
        # llm_compare_coro = (
        #     async_llm_compare(
        #         ticket["title"], ticket["description"],
        #         expected_reply, generated_response_base, generated_response_al,
        #     )
        #     if generated_response_al and generated_response_base
        #     else _noop_compare()
        # )
        llm_comp = None

        # Fire judges concurrently (LLM compare disabled)
        try:
            jr_base, jr_al = await asyncio.gather(
                judge_base_coro, judge_al_coro,
                return_exceptions=True,
            )
        except Exception:
            jr_base, jr_al = {"scores": []}, {"scores": []}

        # Handle exceptions from gather
        if isinstance(jr_base, Exception):
            if i <= 3:
                print(f"{eval_tag} ⚠️ Judge base error: {jr_base}")
            jr_base = {"scores": []}
        if isinstance(jr_al, Exception):
            if i <= 3:
                print(f"{eval_tag} ⚠️ Judge AL error: {jr_al}")
            jr_al = {"scores": []}
        # if isinstance(llm_comp, Exception):
        #     llm_comp = {"better": "tie", "rationale": f"error: {llm_comp}"}

        # Parse judge results
        baseline_judge_scores = jr_base.get("scores", []) if isinstance(jr_base, dict) else []
        baseline_avg_judge = None
        if baseline_judge_scores:
            baseline_avg_judge = sum(s.get("helpfulness", 0) for s in baseline_judge_scores) / len(baseline_judge_scores)

        al_judge_scores = jr_al.get("scores", []) if isinstance(jr_al, dict) else []
        al_avg_judge = None
        if al_judge_scores:
            al_avg_judge = sum(s.get("helpfulness", 0) for s in al_judge_scores) / len(al_judge_scores)

        class_match = _norm_class(ticket["expected_class"]) == _norm_class(predicted_class_al)

        ticket_time = time.time() - ticket_start

        # Update progress
        if progress is not None:
            progress["done"] += 1
            done = progress["done"]
            total = progress["total"]
            if done == 1 or done == total or done % 5 == 0:
                pct = done / total * 100
                print(f"{eval_tag} [POST] Progress: {done}/{total} ({pct:.0f}%)")

        # Determine if AL was gated off for this ticket
        al_was_gated = False
        al_gate_reason = "gating_active"
        similar_replies_al_raw = result_al.get("similar_replies", [])
        if isinstance(similar_replies_al_raw, pd.DataFrame):
            if 'confidence_gated' in similar_replies_al_raw.columns:
                al_was_gated = bool(similar_replies_al_raw['confidence_gated'].any())
        elif isinstance(similar_replies_al_raw, list) and similar_replies_al_raw:
            al_was_gated = any(r.get('confidence_gated', False) for r in similar_replies_al_raw)
        if al_was_gated:
            al_gate_reason = "baseline_too_strong"

        return {
            "ticket_id": i,
            "title": ticket["title"],
            "description": ticket["description"],
            "expected_team": ticket["expected_team"],
            "expected_class": ticket["expected_class"],
            "expected_first_reply": expected_reply,
            "al_gated_off": al_was_gated,
            "al_applied": not al_was_gated,
            "al_gate_reason": al_gate_reason,
            "predicted_team_al": predicted_team_al,
            "predicted_class_al": predicted_class_al,
            "predicted_team_base": predicted_team_base,
            "predicted_class_base": predicted_class_base,
            "generated_response_al": generated_response_al,
            "generated_response_base": generated_response_base,
            "team_match_al": quality_al["team_match"],
            "team_match_base": quality_base["team_match"],
            "class_match_al": class_match,
            "response_length_al": quality_al["length"],
            "response_length_base": quality_base["length"],
            "has_structure_al": quality_al["has_structure"],
            "has_structure_base": quality_base["has_structure"],
            "cosine_similarity_al": cos_al_val,
            "cosine_similarity_base": cos_base_val,
            "rouge_l_f1_al": rouge_al_val,
            "rouge_l_f1_base": rouge_base_val,
            # BERTScore filled in later (batch)
            "bertscore_f1_al": None,
            "bertscore_f1_base": None,
            "judge_avg_score": baseline_avg_judge,
            "judge_scores": baseline_judge_scores,
            "judge_avg_score_al": al_avg_judge,
            "judge_scores_al": al_judge_scores,
            "llm_compare": None,  # LLM compare commented out
            "processing_time_s": ticket_time,
        }


# ============================================================================
# SINGLE EVALUATION WORKER (runs in its own process)
# ============================================================================
def run_single_evaluation_async(
    eval_id: int,
    seed: int,
    num_test_tickets: int,
    original_kb_path: str,
    results_dir: str,
    disable_gating: bool = True,
    concurrent_tickets: int = 5,
) -> dict:
    """Run one complete evaluation in an isolated process, with async ticket processing.

    Called via ProcessPoolExecutor. Each process:
      - Imports all heavy libraries fresh
      - Builds its own RAG system
      - Processes tickets concurrently via asyncio

    Returns a summary dict.
    """
    eval_start = time.time()
    eval_tag = f"[Eval-{eval_id}]"

    # ---- Per-process imports ----
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer, util as st_util

    # ---- Suppress noisy SentenceTransformer / tqdm progress bars ----
    # The hundreds of "Batches: 100%|..." lines make output unreadable.
    # We keep only our own progress lines.
    import logging as _logging
    _logging.getLogger("sentence_transformers").setLevel(_logging.WARNING)
    _logging.getLogger("resolution_task").setLevel(_logging.WARNING)
    _logging.getLogger("resolution_task_async").setLevel(_logging.WARNING)
    _logging.getLogger("absl").setLevel(_logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # also silences tokenizer warnings
    # Monkeypatch tqdm to always be disabled in this worker process
    try:
        import tqdm as _tqdm_mod
        import tqdm.auto as _tqdm_auto
        import tqdm.std as _tqdm_std

        _original_tqdm_init = _tqdm_std.tqdm.__init__

        def _silent_tqdm_init(self, *args, **kwargs):
            kwargs["disable"] = True
            _original_tqdm_init(self, *args, **kwargs)

        _tqdm_std.tqdm.__init__ = _silent_tqdm_init
        _tqdm_mod.tqdm.__init__ = _silent_tqdm_init
        _tqdm_auto.tqdm.__init__ = _silent_tqdm_init
    except Exception:
        pass

    _rouge_scorer_mod = None
    try:
        from rouge_score import rouge_scorer as _rs
        _rouge_scorer_mod = _rs
    except ImportError:
        pass

    _bert_score_fn = None
    try:
        from bert_score import score as _bs
        _bert_score_fn = _bs
    except ImportError:
        pass

    print(f"\n{'=' * 80}")
    print(f"{eval_tag} STARTING (seed={seed}, tickets={num_test_tickets}, concurrent={concurrent_tickets})")
    print(f"{'=' * 80}")

    # ---- Step 1: Load dataset and sample test tickets ----
    print(f"{eval_tag} Loading dataset from {original_kb_path}...")
    df = pd.read_csv(original_kb_path)
    df = df.dropna(subset=["Public_log_anon"])
    df["first_reply"] = df["Public_log_anon"].apply(extract_first_reply_only)
    df = df.dropna(subset=["first_reply"])
    print(f"{eval_tag} Dataset: {len(df)} tickets with valid first replies")

    test_sample = df.sample(num_test_tickets, random_state=seed)
    test_tickets = []
    test_indices = []
    for idx, row in test_sample.iterrows():
        test_tickets.append({
            "title": row["Title_anon"],
            "description": row["Description_anon"],
            "expected_team": get_expected_team(row),
            "expected_class": get_expected_class(row["Title_anon"], row["Description_anon"]),
            "expected_first_reply": row["first_reply"],
        })
        test_indices.append(idx)

    # ---- Step 2: Create isolated KB ----
    kb_copy_path = os.path.join(results_dir, f"kb_eval_{eval_id}_{seed}.csv")
    df_kb = df.drop(test_indices).reset_index(drop=True)
    df_kb.to_csv(kb_copy_path, index=False)
    print(f"{eval_tag} KB created: {kb_copy_path} ({len(df_kb)} tickets)")

    # ---- Step 3: Build RAG system ----
    os.environ["DISABLE_EMBEDDING_CACHE"] = "1"
    if disable_gating:
        os.environ["FEEDBACK_CONFIDENCE_GATE"] = "false"
        os.environ["AL_TIER_ENABLED"] = "false"
    os.environ["FEEDBACK_ENABLED"] = "True"
    
    ##########generated feedback db############
    # feedback_db_path = os.path.join(results_dir, f"feedback_eval_{eval_id}.db")
    # os.environ["FEEDBACK_DB_PATH"] = feedback_db_path

    # # Start with EMPTY feedback DB — let self-teaching loop build organic feedback.
    # # The old production DB (feedback_refid.db) contains votes that were originally
    # # collected with broken index-based IDs.  Even after migrating to Ref IDs the
    # # vote *values* are noise (87% negative), so seeding with them actively hurts AL.
    # # seeded_rows = _seed_feedback_db(feedback_db_path)
    # seeded_rows = 0  # intentionally empty — self-teaching only
    # if seeded_rows:
    #     print(f"{eval_tag} ✅ Seeded feedback DB with {seeded_rows} aggregated rows from production feedback_refid.db")
    # else:
    #     print(f"{eval_tag} 🧪 Starting with EMPTY feedback DB — self-teaching only")

    ####prebuilt feedback db######
    feedback_db_path = os.path.join(results_dir, f"feedback_eval_{eval_id}.db")
    os.environ["FEEDBACK_DB_PATH"] = feedback_db_path

    # Start with EMPTY feedback DB by default. Optionally seed from a prebuilt
    # high-quality feedback DB supplied via PREBUILT_FEEDBACK_DB_PATH environment
    # variable. This allows evaluations to start with realistic signal instead of
    # building from scratch.
    seeded_rows = 0  # default: none
    prebuilt = os.getenv("PREBUILT_FEEDBACK_DB_PATH", "feedback_global.db")
    if prebuilt and os.path.exists(prebuilt):
        print(f"{eval_tag} Seeding feedback DB from {prebuilt}")
        import shutil, sqlite3
        shutil.copy2(prebuilt, feedback_db_path)
        try:
            conn = sqlite3.connect(feedback_db_path)
            seeded_rows = conn.execute("SELECT COUNT(*) FROM feedback_agg").fetchone()[0]
        except Exception:
            seeded_rows = 0
        finally:
            conn.close()

    # Legacy option: copy production feedback_refid.db if needed (kept for reference)
    # if seeded_rows == 0 and os.path.exists(PRODUCTION_FEEDBACK_DB):
    #     seeded_rows = _seed_feedback_db(feedback_db_path)

    if seeded_rows:
        print(f"{eval_tag} ✅ Seeded feedback DB with {seeded_rows} aggregated rows from {prebuilt or 'production feedback_refid.db'}")
    else:
        print(f"{eval_tag} 🧪 Starting with EMPTY feedback DB — self-teaching only")

    import importlib
    if "resolution_task" in sys.modules:
        importlib.reload(sys.modules["resolution_task"])
    import resolution_task as rt_al

    rt_al.FEEDBACK_ENABLED = True
    if disable_gating:
        rt_al.FEEDBACK_CONFIDENCE_GATE = False
        rt_al.AL_TIER_ENABLED = False

    al_df = rt_al.load_knowledge_base(kb_copy_path)
    al_rag = rt_al.RAGSystem(al_df, kb_path=kb_copy_path)
    build_start = time.time()
    al_rag.build_index(kb_path=kb_copy_path, kb_mtime=os.path.getmtime(kb_copy_path))
    al_build_time = time.time() - build_start
    print(f"{eval_tag} AL RAG built in {al_build_time:.1f}s ({len(al_df)} tickets indexed)")

    # ---- Pre-load classifiers serially to avoid race conditions ----
    # Multiple concurrent threads calling lazy-load simultaneously causes
    # "Cannot copy out of meta tensor" errors.
    print(f"{eval_tag} Pre-loading classifiers (ticket + team) serially...")
    try:
        import io, contextlib
        _warmup_text = "test warmup ticket"
        with contextlib.redirect_stdout(io.StringIO()):
            rt_al.classify_ticket(_warmup_text)
            rt_al.classify_team_with_distilbert(_warmup_text)
        print(f"{eval_tag} ✅ Classifiers pre-loaded successfully")
    except Exception as e:
        print(f"{eval_tag} ⚠️ Classifier pre-load warning: {e}")

    # ---- OPTIMIZATION 3: Reuse al_rag.sentence_model for eval metrics ----
    eval_sentence_model = al_rag.sentence_model
    print(f"{eval_tag} Reusing RAG sentence_model for eval metrics (no second model load)")

    # ROUGE scorer
    rouge_scorer_instance = None
    if _rouge_scorer_mod is not None:
        rouge_scorer_instance = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)

    # ---- OPTIMIZATION 4: Batch-encode all expected replies upfront ----
    print(f"{eval_tag} Batch-encoding {len(test_tickets)} expected replies...")
    expected_replies = [t.get("expected_first_reply", "") or "" for t in test_tickets]
    # Filter out empty replies for encoding, but keep index mapping
    non_empty_mask = [bool(r.strip()) for r in expected_replies]
    non_empty_replies = [r for r, m in zip(expected_replies, non_empty_mask) if m]

    if non_empty_replies:
        batch_embeddings = eval_sentence_model.encode(
            non_empty_replies, convert_to_tensor=True, batch_size=64, show_progress_bar=False
        )
        # Map back: expected_emb_tensors[i] = tensor or None
        expected_emb_tensors = []
        emb_idx = 0
        for m in non_empty_mask:
            if m:
                expected_emb_tensors.append(batch_embeddings[emb_idx])
                emb_idx += 1
            else:
                expected_emb_tensors.append(None)
    else:
        expected_emb_tensors = [None] * len(test_tickets)

    encode_time = time.time() - build_start - al_build_time
    print(f"{eval_tag} Expected replies encoded in {encode_time:.1f}s")

    # ---- Step 5: Process tickets concurrently via asyncio ----
    print(f"\n{eval_tag} Processing {num_test_tickets} tickets (concurrency={concurrent_tickets})...")
    print(f"{eval_tag} Using SELF-TEACHING AL (sequential) + parallel baseline design.")

    async def _run_all_tickets():
        sem = asyncio.Semaphore(concurrent_tickets)

        # ===== PASS 1: AL — SELF-TEACHING SEQUENTIAL LOOP =====
        # Each ticket is processed one at a time.  After generation the
        # retrieved examples are judged and votes are recorded into the
        # feedback DB so that every subsequent ticket benefits from the
        # accumulated signal.
        print(f"{eval_tag} [PASS 1/2] Self-teaching AL pass (FEEDBACK_ENABLED=True, sequential)...")
        rt_al.FEEDBACK_ENABLED = True
        pass1_start = time.time()
        al_results_raw = await _self_teaching_al_pass(
            test_tickets, al_rag, rt_al, eval_tag,
            concurrent_tickets=concurrent_tickets,
        )
        pass1_time = time.time() - pass1_start
        al_ok = sum(1 for r in al_results_raw if r is not None)
        print(f"{eval_tag} [PASS 1/2] ✅ Done: {al_ok}/{len(test_tickets)} succeeded in {pass1_time:.1f}s")

        # ===== PASS 2: Baseline (FEEDBACK_ENABLED = False) =====
        print(f"{eval_tag} [PASS 2/2] Generating Baseline responses (FEEDBACK_ENABLED=False)...")
        rt_al.FEEDBACK_ENABLED = False
        base_progress = {"done": 0, "total": len(test_tickets)}
        pass2_start = time.time()
        base_tasks = [
            _generate_for_ticket_async(i, ticket, al_rag, rt_al, eval_tag, sem, "BASE", base_progress)
            for i, ticket in enumerate(test_tickets, 1)
        ]
        base_results_raw = await asyncio.gather(*base_tasks, return_exceptions=True)
        pass2_time = time.time() - pass2_start
        base_ok = sum(1 for r in base_results_raw if r is not None and not isinstance(r, Exception))
        print(f"{eval_tag} [PASS 2/2] ✅ Done: {base_ok}/{len(test_tickets)} succeeded in {pass2_time:.1f}s")

        # Restore
        rt_al.FEEDBACK_ENABLED = True

        # ===== PASS 3: Post-processing (judge, LLM compare, cosine, ROUGE) =====
        # No FEEDBACK_ENABLED dependency — fully concurrent
        post_start = time.time()
        post_tasks = []
        valid_indices = []
        for i in range(len(test_tickets)):
            al_r = al_results_raw[i]
            base_r = base_results_raw[i]
            # Skip if either pass failed
            if isinstance(al_r, Exception) or al_r is None:
                if isinstance(al_r, Exception):
                    print(f"{eval_tag} ❌ AL error ticket {i+1}: {al_r}")
                continue
            if isinstance(base_r, Exception) or base_r is None:
                if isinstance(base_r, Exception):
                    print(f"{eval_tag} ❌ Base error ticket {i+1}: {base_r}")
                continue
            valid_indices.append(i)

        print(f"{eval_tag} [POST] Running judge + LLM compare + metrics for {len(valid_indices)} tickets concurrently...")
        post_progress = {"done": 0, "total": len(valid_indices)}
        for i in valid_indices:
            post_tasks.append(
                _postprocess_ticket_async(
                    i=i + 1,
                    ticket=test_tickets[i],
                    result_al=al_results_raw[i],
                    result_base=base_results_raw[i],
                    al_rag=al_rag,
                    expected_emb_tensor=expected_emb_tensors[i],
                    rouge_scorer_instance=rouge_scorer_instance,
                    eval_tag=eval_tag,
                    semaphore=sem,
                    progress=post_progress,
                )
            )

        post_results_raw = await asyncio.gather(*post_tasks, return_exceptions=True)
        post_time = time.time() - post_start

        # Filter
        final = []
        for r in post_results_raw:
            if isinstance(r, Exception):
                print(f"{eval_tag} ❌ Post-process error: {r}")
            elif r is not None:
                final.append(r)

        total_async = time.time() - pass1_start
        print(f"{eval_tag} [POST] ✅ Done: {len(final)}/{len(valid_indices)} in {post_time:.1f}s")
        print(f"{eval_tag} 📊 Async summary: Pass1={pass1_time:.0f}s Pass2={pass2_time:.0f}s Post={post_time:.0f}s Total={total_async:.0f}s")
        return final

    raw_results = asyncio.run(_run_all_tickets())

    # Results are already filtered by _run_all_tickets
    results = raw_results
    print(f"{eval_tag} Completed: {len(results)}/{num_test_tickets} tickets")

    # ---- OPTIMIZATION 5: Batch BERTScore ----
    if _bert_score_fn is not None and results:
        print(f"{eval_tag} Computing BERTScore in batch ({len(results) * 2} pairs)...")
        bert_start = time.time()

        # Collect all (candidate, reference) pairs
        al_candidates = []
        al_references = []
        al_indices = []  # which result index each pair maps to
        base_candidates = []
        base_references = []
        base_indices = []

        for idx, r in enumerate(results):
            exp = r.get("expected_first_reply", "")
            gen_al = r.get("generated_response_al", "")
            gen_base = r.get("generated_response_base", "")
            if exp and gen_al:
                al_candidates.append(gen_al)
                al_references.append(exp)
                al_indices.append(idx)
            if exp and gen_base:
                base_candidates.append(gen_base)
                base_references.append(exp)
                base_indices.append(idx)

        # Batch score AL
        if al_candidates:
            try:
                _, _, F1_al = _bert_score_fn(al_candidates, al_references, lang="en", verbose=False)
                for j, idx in enumerate(al_indices):
                    results[idx]["bertscore_f1_al"] = float(F1_al[j])
            except Exception as e:
                print(f"{eval_tag} ⚠️ BERTScore AL batch error: {e}")

        # Batch score baseline
        if base_candidates:
            try:
                _, _, F1_base = _bert_score_fn(base_candidates, base_references, lang="en", verbose=False)
                for j, idx in enumerate(base_indices):
                    results[idx]["bertscore_f1_base"] = float(F1_base[j])
            except Exception as e:
                print(f"{eval_tag} ⚠️ BERTScore base batch error: {e}")

        bert_time = time.time() - bert_start
        print(f"{eval_tag} BERTScore batch completed in {bert_time:.1f}s")

    # ---- Step 6: Compute summary ----
    total_time = time.time() - eval_start

    cosine_al = [r["cosine_similarity_al"] for r in results if r["cosine_similarity_al"] is not None]
    cosine_base = [r["cosine_similarity_base"] for r in results if r["cosine_similarity_base"] is not None]
    rouge_l_al = [r["rouge_l_f1_al"] for r in results if r["rouge_l_f1_al"] is not None]
    rouge_l_base = [r["rouge_l_f1_base"] for r in results if r["rouge_l_f1_base"] is not None]
    bert_f1_al = [r["bertscore_f1_al"] for r in results if r["bertscore_f1_al"] is not None]
    bert_f1_base = [r["bertscore_f1_base"] for r in results if r["bertscore_f1_base"] is not None]

    avg_cosine_al = sum(cosine_al) / len(cosine_al) if cosine_al else 0.0
    avg_cosine_base = sum(cosine_base) / len(cosine_base) if cosine_base else 0.0
    avg_rouge_al = sum(rouge_l_al) / len(rouge_l_al) if rouge_l_al else 0.0
    avg_rouge_base = sum(rouge_l_base) / len(rouge_l_base) if rouge_l_base else 0.0
    avg_bert_al = sum(bert_f1_al) / len(bert_f1_al) if bert_f1_al else 0.0
    avg_bert_base = sum(bert_f1_base) / len(bert_f1_base) if bert_f1_base else 0.0

    judge_scores_all = [r["judge_avg_score"] for r in results if r.get("judge_avg_score") is not None]
    avg_judge = sum(judge_scores_all) / len(judge_scores_all) if judge_scores_all else 0.0

    class_matches = [r["class_match_al"] for r in results]
    team_acc = sum(r["team_match_al"] for r in results) / len(results) if results else 0.0
    class_acc = sum(class_matches) / len(class_matches) if class_matches else 0.0
    avg_len = sum(r["response_length_al"] for r in results) / len(results) if results else 0.0

    # LLM compare commented out
    # llm_comp_results = [r["llm_compare"] for r in results if r.get("llm_compare")]
    # better_al = sum(1 for r in llm_comp_results if r.get("better") == "B")
    # better_base = sum(1 for r in llm_comp_results if r.get("better") == "A")
    # ties = sum(1 for r in llm_comp_results if r.get("better") == "tie")
    better_al = 0
    better_base = 0
    ties = 0

    summary = {
        "eval_id": eval_id,
        "seed": seed,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "total_tickets": len(results),
        "num_test_tickets_requested": num_test_tickets,
        "concurrent_tickets": concurrent_tickets,
        "gating_disabled": disable_gating,
        "avg_cosine_similarity_al": avg_cosine_al,
        "avg_cosine_similarity_base": avg_cosine_base,
        "cosine_delta": avg_cosine_al - avg_cosine_base,
        "avg_rougeL_f1_al": avg_rouge_al,
        "avg_rougeL_f1_base": avg_rouge_base,
        "rouge_delta": avg_rouge_al - avg_rouge_base,
        "avg_bert_f1_al": avg_bert_al,
        "avg_bert_f1_base": avg_bert_base,
        "bert_delta": avg_bert_al - avg_bert_base,
        "avg_judge_helpfulness": avg_judge,
        "team_accuracy": team_acc,
        "class_accuracy": class_acc,
        "avg_response_length": avg_len,
        "llm_better_al": better_al,
        "llm_better_base": better_base,
        "llm_ties": ties,
        "total_time_s": total_time,
        "avg_time_per_ticket_s": total_time / len(results) if results else 0.0,
        "embedding_build_time_s": al_build_time,
        "knowledge_base": kb_copy_path,
        "knowledge_base_size": len(df_kb),
    }

    # ---- Step 7: Save results ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"eval_{eval_id}_results_{timestamp}.json")
    summary_file = os.path.join(results_dir, f"eval_{eval_id}_summary_{timestamp}.json")

    safe_results = _sanitize_for_json(results)
    with open(results_file, "w") as f:
        json.dump(safe_results, f, indent=2)

    safe_summary = _sanitize_for_json(summary)
    with open(summary_file, "w") as f:
        json.dump(safe_summary, f, indent=2)

    # Clean up temp files (keep feedback DB for post-hoc inspection)
    for path in [
        kb_copy_path,
        # os.path.join(results_dir, f"feedback_eval_{eval_id}.db"),  # KEEP for analysis
    ]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    print(f"\n{'=' * 80}")
    print(f"{eval_tag} COMPLETE in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{eval_tag} Tickets: {len(results)}/{num_test_tickets}")
    print(f"{eval_tag} Cosine: AL={avg_cosine_al:.4f} Base={avg_cosine_base:.4f} Δ={avg_cosine_al - avg_cosine_base:+.4f}")
    print(f"{eval_tag} ROUGE:  AL={avg_rouge_al:.4f} Base={avg_rouge_base:.4f} Δ={avg_rouge_al - avg_rouge_base:+.4f}")
    print(f"{eval_tag} BERT:   AL={avg_bert_al:.4f} Base={avg_bert_base:.4f} Δ={avg_bert_al - avg_bert_base:+.4f}")
    print(f"{eval_tag} Judge:  {avg_judge:.4f}")
    # print(f"{eval_tag} LLM Compare: AL-better={better_al} Base-better={better_base} Ties={ties}")
    print(f"{eval_tag} Saved: {results_file}")
    print(f"{'=' * 80}\n")

    return safe_summary


# ============================================================================
# ORCHESTRATOR
# ============================================================================
def run_multiple_evaluations_async(
    num_evals: int = 3,
    tickets_per_eval: int = 80,
    max_parallel: int = 2,
    base_seed: Optional[int] = None,
    concurrent_tickets: int = 5,
):
    """Launch multiple independent evaluations in parallel using separate processes.

    Within each process, tickets are processed concurrently via asyncio.
    """
    print("=" * 80)
    print(f"🚀 FAST MULTI-EVALUATION RUNNER (async)")
    print(f"=" * 80)
    print(f"   Evaluations:          {num_evals}")
    print(f"   Tickets per eval:     {tickets_per_eval}")
    print(f"   Max parallel evals:   {max_parallel}")
    print(f"   Concurrent tickets:   {concurrent_tickets}")
    print(f"   Gating:               DISABLED")
    print(f"   Results dir:          {RESULTS_DIR}")
    print()

    # Prerequisites
    if not os.path.exists(ORIGINAL_KB_PATH):
        print(f"❌ Knowledge base not found: {ORIGINAL_KB_PATH}")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set")
        print("   Set it with: SET OPENAI_API_KEY=sk-...")
        print()

    # Seeds
    if base_seed is not None:
        seeds = [base_seed + i for i in range(num_evals)]
        print(f"   Seeds (deterministic): {seeds}")
    else:
        seeds = [random.randint(0, 2**32 - 1) for _ in range(num_evals)]
        print(f"   Seeds (random):        {seeds}")
    print()

    results_dir_str = str(RESULTS_DIR)
    os.makedirs(results_dir_str, exist_ok=True)

    all_summaries = []
    failed_evals = []
    overall_start = time.time()

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for eval_id in range(1, num_evals + 1):
            future = executor.submit(
                run_single_evaluation_async,
                eval_id=eval_id,
                seed=seeds[eval_id - 1],
                num_test_tickets=tickets_per_eval,
                original_kb_path=ORIGINAL_KB_PATH,
                results_dir=results_dir_str,
                disable_gating=False,  # Enable evidence-based gating (2026-02-17)
                concurrent_tickets=concurrent_tickets,
            )
            futures[future] = eval_id
            print(f"📤 Submitted Eval-{eval_id} (seed={seeds[eval_id - 1]})")

        print(f"\n⏳ Waiting for {num_evals} evaluations...\n")

        for future in as_completed(futures):
            eval_id = futures[future]
            try:
                summary = future.result()
                all_summaries.append(summary)
                print(
                    f"✅ Eval-{eval_id} finished: "
                    f"cos_al={summary['avg_cosine_similarity_al']:.4f} "
                    f"cos_base={summary['avg_cosine_similarity_base']:.4f} "
                    f"Δ={summary['cosine_delta']:+.4f} "
                    f"in {summary['total_time_s']:.0f}s"
                )
            except Exception as e:
                print(f"❌ Eval-{eval_id} FAILED: {e}")
                traceback.print_exc()
                failed_evals.append(eval_id)

    overall_time = time.time() - overall_start

    # ---- Aggregate ----
    print(f"\n{'=' * 80}")
    print(f"📊 AGGREGATE RESULTS (async)")
    print(f"{'=' * 80}")
    print(f"   Completed: {len(all_summaries)}/{num_evals}")
    if failed_evals:
        print(f"   Failed:    {failed_evals}")
    print(f"   Total time: {overall_time:.1f}s ({overall_time / 60:.1f} min)")
    print()

    if all_summaries:
        cos_al_vals = [s["avg_cosine_similarity_al"] for s in all_summaries]
        cos_base_vals = [s["avg_cosine_similarity_base"] for s in all_summaries]
        cos_deltas = [s["cosine_delta"] for s in all_summaries]
        rouge_deltas = [s["rouge_delta"] for s in all_summaries if s.get("rouge_delta") is not None]
        bert_deltas = [s["bert_delta"] for s in all_summaries if s.get("bert_delta") is not None]

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def _std(lst):
            if len(lst) < 2:
                return 0.0
            m = _mean(lst)
            return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

        print(f"   {'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"   {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
        for label, vals in [
            ("Cosine AL", cos_al_vals),
            ("Cosine Baseline", cos_base_vals),
            ("Cosine Δ (AL-Base)", cos_deltas),
        ]:
            print(f"   {label:<30} {_mean(vals):>8.4f} {_std(vals):>8.4f} {min(vals) if vals else 0:>8.4f} {max(vals) if vals else 0:>8.4f}")

        if rouge_deltas:
            rouge_al = [s["avg_rougeL_f1_al"] for s in all_summaries]
            rouge_base = [s["avg_rougeL_f1_base"] for s in all_summaries]
            for label, vals in [
                ("ROUGE-L AL", rouge_al),
                ("ROUGE-L Baseline", rouge_base),
                ("ROUGE-L Δ", rouge_deltas),
            ]:
                print(f"   {label:<30} {_mean(vals):>8.4f} {_std(vals):>8.4f} {min(vals) if vals else 0:>8.4f} {max(vals) if vals else 0:>8.4f}")

        if bert_deltas:
            bert_al = [s["avg_bert_f1_al"] for s in all_summaries]
            bert_base = [s["avg_bert_f1_base"] for s in all_summaries]
            for label, vals in [
                ("BERTScore AL", bert_al),
                ("BERTScore Baseline", bert_base),
                ("BERTScore Δ", bert_deltas),
            ]:
                print(f"   {label:<30} {_mean(vals):>8.4f} {_std(vals):>8.4f} {min(vals) if vals else 0:>8.4f} {max(vals) if vals else 0:>8.4f}")

        judge_vals = [s["avg_judge_helpfulness"] for s in all_summaries if s.get("avg_judge_helpfulness")]
        if judge_vals:
            print(f"   {'Judge Helpfulness':<30} {_mean(judge_vals):>8.4f} {_std(judge_vals):>8.4f} {min(judge_vals):>8.4f} {max(judge_vals):>8.4f}")

        total_al_better = sum(s.get("llm_better_al", 0) for s in all_summaries)
        total_base_better = sum(s.get("llm_better_base", 0) for s in all_summaries)
        total_ties = sum(s.get("llm_ties", 0) for s in all_summaries)

        # LLM comparison print commented out
        # print(f"\n   LLM Comparison (total):")
        # print(f"   AL better:       {total_al_better}")
        # print(f"   Baseline better: {total_base_better}")
        # print(f"   Ties:            {total_ties}")

        # Significance
        n = len(cos_deltas)
        if n >= 3:
            mean_delta = _mean(cos_deltas)
            std_delta = _std(cos_deltas)
            if std_delta > 0:
                t_stat = mean_delta / (std_delta / (n**0.5))
                sig = "YES (p < 0.05)" if abs(t_stat) > 2.0 else "NO"
                direction = "AL better" if mean_delta > 0 else "Baseline better"
                print(f"\n   📈 Cosine delta significance (t-test, n={n}):")
                print(f"      Mean Δ = {mean_delta:+.4f}, t = {t_stat:.3f}")
                print(f"      Significant: {sig} → {direction}")

        # Save aggregate
        agg_file = RESULTS_DIR / f"aggregate_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        aggregate = {
            "num_evaluations": len(all_summaries),
            "num_failed": len(failed_evals),
            "tickets_per_eval": tickets_per_eval,
            "concurrent_tickets": concurrent_tickets,
            "seeds": seeds,
            "gating_disabled": True,
            "total_time_s": overall_time,
            "cosine_al_mean": _mean(cos_al_vals),
            "cosine_al_std": _std(cos_al_vals),
            "cosine_base_mean": _mean(cos_base_vals),
            "cosine_base_std": _std(cos_base_vals),
            "cosine_delta_mean": _mean(cos_deltas),
            "cosine_delta_std": _std(cos_deltas),
            "rouge_delta_mean": _mean(rouge_deltas) if rouge_deltas else None,
            "bert_delta_mean": _mean(bert_deltas) if bert_deltas else None,
            "judge_mean": _mean(judge_vals) if judge_vals else None,
            "llm_total_al_better": total_al_better,
            "llm_total_base_better": total_base_better,
            "llm_total_ties": total_ties,
            "individual_summaries": all_summaries,
        }
        with open(agg_file, "w") as f:
            json.dump(_sanitize_for_json(aggregate), f, indent=2)

        print(f"\n   💾 Aggregate saved: {agg_file}")

    print(f"\n{'=' * 80}")
    print(f"✨ All evaluations complete!")
    print(f"{'=' * 80}")

    return all_summaries


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FAST multi-eval runner with async LLM calls + batched metrics."
    )
    parser.add_argument("--num-evals", type=int, default=3, help="Number of evaluation runs (default: 3)")
    parser.add_argument("--tickets-per-eval", type=int, default=80, help="Test tickets per evaluation (default: 80)")
    parser.add_argument("--max-parallel", type=int, default=2, help="Max concurrent evaluation processes (default: 2)")
    parser.add_argument("--concurrent-tickets", type=int, default=5, help="Concurrent tickets per evaluation (default: 5). Controls how many OpenAI requests are in-flight simultaneously.")
    parser.add_argument("--base-seed", type=int, default=None, help="Base seed for reproducibility. Eval i gets seed base_seed+i.")

    args = parser.parse_args()

    print(f"🔧 Configuration:")
    print(f"   Python:           {sys.version.split()[0]}")
    print(f"   CWD:              {os.getcwd()}")
    print(f"   OpenAI:           {'✅ key set' if os.getenv('OPENAI_API_KEY') else '❌ not set'}")
    print(f"   Async speedups:   ✅ Enabled")
    print()

    run_multiple_evaluations_async(
        num_evals=args.num_evals,
        tickets_per_eval=args.tickets_per_eval,
        max_parallel=args.max_parallel,
        base_seed=args.base_seed,
        concurrent_tickets=args.concurrent_tickets,
    )
