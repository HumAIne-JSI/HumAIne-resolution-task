"""
Async wrapper around resolution_task for concurrent LLM calls.

This module does NOT duplicate resolution_task.py. Instead, it provides:
  1. An async OpenAI chat completion function (_async_chat_completion)
  2. An async version of generate_response that:
     - Runs CPU-bound work (classification, FAISS retrieval) synchronously (fast, ~50ms)
     - Runs the LLM generation step asynchronously (slow, ~2-5s)
  3. An async judge_items function for concurrent judge scoring

The sync resolution_task module is imported and used directly for all
non-LLM operations (classification, retrieval, prompt building, etc.).

Usage:
    import resolution_task as rt
    from resolution_task_async import async_generate_response, async_judge_items

    rag = rt.RAGSystem(df, kb_path=path)
    rag.build_index(...)

    result = await async_generate_response(title, desc, rag, retrieval_k=5)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# ASYNC OPENAI CLIENT
# ============================================================================
_async_openai_client = None
_async_client_mode = "unavailable"

try:
    from openai import AsyncOpenAI

    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        _async_openai_client = AsyncOpenAI(api_key=_api_key)
    else:
        _async_openai_client = AsyncOpenAI()
    _async_client_mode = "async_v1"
    logger.info("AsyncOpenAI client initialized")
except ImportError:
    logger.warning("openai package not installed or missing AsyncOpenAI")
except Exception as e:
    logger.warning(f"AsyncOpenAI init error: {e}")


async def _async_chat_completion(model: str, messages: list, **kwargs) -> str:
    """Async version of resolution_task._chat_completion."""
    if _async_openai_client is None:
        return "[AsyncOpenAI client unavailable]"
    resp = await _async_openai_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return resp.choices[0].message.content.strip()


# ============================================================================
# ASYNC GENERATE RESPONSE
# ============================================================================
async def async_generate_response(
    ticket_title: str,
    ticket_description: str,
    rag_system,  # resolution_task.RAGSystem
    retrieval_k: int = 5,
    rt_module=None,  # pass the imported resolution_task module
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> dict:
    """Async version of resolution_task.generate_response.

    Architecture:
      - Steps 1-3 (classify, retrieve) are CPU-bound and fast (~50ms total).
        We run them in a thread via run_in_executor to avoid blocking the event loop,
        but they complete almost instantly.
      - Step 4 (LLM generation) is I/O-bound and slow (~2-5s).
        We call AsyncOpenAI directly for true async concurrency.

    Args:
        rt_module: The imported resolution_task module (allows caller to patch
                   FEEDBACK_ENABLED etc. on their own module reference).
    """
    if rt_module is None:
        import resolution_task as rt_module

    loop = loop or asyncio.get_event_loop()

    # --- Step 1-3: CPU-bound work in thread ---
    # We extract the classify + retrieve steps from generate_response and run them
    # synchronously, then do the LLM call async.

    def _cpu_work():
        """Run classification and retrieval (no LLM call)."""
        import pandas as pd
        import io
        import contextlib

        profile_enabled = os.getenv("PROFILE_API", "false").lower() in ("1", "true", "yes")
        quiet = os.getenv("QUIET_CPU_WORK", "1") == "1"
        timings = {} if profile_enabled else None

        ticket_text = str(ticket_title) + " " + str(ticket_description)

        # Suppress verbose prints from resolution_task (📋 DistilBERT input, etc.)
        cm = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        with cm:
            # Step 1: Classification
            if profile_enabled:
                t0 = time.time()
            word_class, template = rt_module.classify_ticket(ticket_text)
            if profile_enabled:
                timings["classify_ticket"] = time.time() - t0

            category_mapping = {
                "vpn_request": "to enable Client-to-Site VPN tunnel requests",
                "onboarding": "to start the IT On-boarding of a new internal employee",
                "software_request": "to request a software or a licence on my Windows device",
                "offboarding": "to start the IT Off-boarding process",
                "absence_request": "to request an Absence ticket",
                "admin_rights": "to request admin rights on the computer",
            }
            predicted_category = category_mapping.get(word_class, None)

            # Step 2: Team classification
            if profile_enabled:
                t0 = time.time()
            predicted_team, team_confidence = rt_module.classify_team_with_distilbert(ticket_text)
            if profile_enabled:
                timings["classify_team"] = time.time() - t0

            temporal_context = rt_module.detect_temporal_context(ticket_title, ticket_description)

            # Step 3: Retrieval
            if profile_enabled:
                t0 = time.time()
            similar_replies = rag_system.retrieve_similar_replies(
                ticket_text,
                top_k=retrieval_k,
                predicted_category=predicted_category,
                predicted_class=word_class,
                predicted_team=predicted_team,
            )
            if profile_enabled:
                timings["retrieval"] = time.time() - t0

            # Step 3b: MMR diversity re-ranking (if enabled via env var)
            # Re-orders the top-k to reduce redundancy while keeping relevance
            mmr_enabled = os.getenv("MMR_RERANK", "1") in ("1", "true", "yes")
            if mmr_enabled and len(similar_replies) > 1:
                try:
                    import numpy as np
                    query_emb = rag_system.sentence_model.encode([ticket_text])
                    # Get embeddings for each retrieved item
                    sr_indices = similar_replies.index.tolist()
                    sr_embeddings = np.array([
                        rag_system.embeddings[idx] if idx < len(rag_system.embeddings)
                        else rag_system.sentence_model.encode([str(similar_replies.loc[idx].get("first_reply", ""))])[0]
                        for idx in sr_indices
                    ])
                    from run_multiple_evaluations_async import _mmr_select
                    mmr_order = _mmr_select(
                        query_emb, sr_embeddings,
                        list(range(len(sr_indices))),
                        top_k=len(sr_indices),
                        lambda_param=float(os.getenv("MMR_LAMBDA", "0.7")),
                    )
                    similar_replies = similar_replies.iloc[mmr_order].reset_index(drop=True)
                except Exception as e:
                    logger.debug(f"MMR re-ranking skipped: {e}")

            return {
                "word_class": word_class,
                "predicted_team": predicted_team,
                "team_confidence": team_confidence,
                "similar_replies": similar_replies,
                "predicted_category": predicted_category,
                "temporal_context": temporal_context,
                "timings": timings,
            }

    cpu_result = await loop.run_in_executor(None, _cpu_work)

    word_class = cpu_result["word_class"]
    predicted_team = cpu_result["predicted_team"]
    team_confidence = cpu_result["team_confidence"]
    similar_replies = cpu_result["similar_replies"]
    temporal_context = cpu_result["temporal_context"]

    # --- Step 4: Async LLM generation ---
    # We need to replicate the prompt-building logic from generate_response_with_openai
    # but call _async_chat_completion instead of _chat_completion.
    response = await _async_generate_response_with_openai(
        ticket_title,
        ticket_description,
        word_class,
        predicted_team,
        team_confidence,
        similar_replies,
        temporal_context=temporal_context,
        rt_module=rt_module,
    )

    result = {
        "classification": word_class,
        "predicted_team": predicted_team,
        "team_confidence": team_confidence,
        "response": response,
        "similar_replies": similar_replies.to_dict(orient="records"),
        "predicted_class": word_class,
        "retrieval_k": retrieval_k,
    }

    if cpu_result.get("timings"):
        result["_generation_profiling"] = cpu_result["timings"]

    return result


async def _async_generate_response_with_openai(
    ticket_title,
    ticket_description,
    classification,
    predicted_team,
    team_confidence,
    similar_replies,
    temporal_context="standard",
    rt_module=None,
):
    """Async version of generate_response_with_openai.

    Replicates the prompt-building logic but uses _async_chat_completion.
    """
    import pandas as pd

    if rt_module is None:
        import resolution_task as rt_module

    # Analyze similar replies to determine response type
    template_examples = []
    personalized_examples = []
    short_examples = []

    for _, reply in similar_replies.iterrows():
        first_reply = str(reply.get("first_reply", ""))
        response_type, confidence, length_category = rt_module.analyze_response_type_simple(first_reply)

        if length_category == "short" and len(first_reply) < 400:
            short_examples.append(reply)
        elif response_type == "template" and confidence > 0.5:
            template_examples.append(reply)
        elif response_type == "personalized" and confidence > 0.5:
            personalized_examples.append(reply)

    has_short_replies = len(short_examples) >= 2 or (
        len(short_examples) > 0 and len(short_examples) >= len(template_examples)
    )

    if has_short_replies:
        # Short response path
        short_df = pd.DataFrame(short_examples) if short_examples else pd.DataFrame()
        return await _async_generate_short(
            ticket_title, ticket_description, classification, predicted_team, short_df, rt_module
        )

    use_template = len(template_examples) > len(personalized_examples)
    if temporal_context == "temporal_update":
        use_template = False

    if use_template and template_examples:
        template_df = pd.DataFrame(template_examples)
        return await _async_generate_template(
            ticket_title, ticket_description, classification, predicted_team, template_df, rt_module
        )
    else:
        if personalized_examples:
            personalized_df = pd.DataFrame(personalized_examples)
        else:
            personalized_df = similar_replies

        if temporal_context == "temporal_update":
            return await _async_generate_status_update(
                ticket_title, ticket_description, classification,
                predicted_team, team_confidence, personalized_df, rt_module
            )

        return await _async_generate_personal(
            ticket_title, ticket_description, classification,
            predicted_team, team_confidence, personalized_df, rt_module
        )


async def _async_generate_template(title, desc, classification, team, template_examples, rt_module):
    """Async template response generation."""
    prompt = (
        f"You are an IT support system that generates STRUCTURED TEMPLATE RESPONSES. "
        f"You must follow the EXACT format and structure shown in the examples below.\n\n"
        f"IMPORTANT INSTRUCTIONS:\n"
        f"- Copy the EXACT format from similar tickets\n"
        f"- Use structured lists with dashes (-)\n"
        f"- Include form fields like 'Need:', 'Project Manager Full name:', etc.\n"
        f"- Do NOT write conversational responses\n"
        f"- Start with 'Below you will find the additional form information' when appropriate\n"
        f"- For onboarding tickets, use brief auto-generated messages\n\n"
        f"Ticket Title: {title}\n"
        f"Ticket Description: {desc}\n"
        f"Request Type: {classification}\n"
        f"Assigned Team: {team}\n\n"
        f"TEMPLATE EXAMPLES FROM SIMILAR TICKETS:\n"
    )
    count = 0
    for _, reply in template_examples.iterrows():
        if count >= 3:
            break
        prompt += (
            f"EXAMPLE {count + 1}:\n"
            f"Title: {reply.get('Title_anon', 'N/A')}\n"
            f"Description: {reply.get('Description_anon', 'N/A')}\n"
            f"Template Response:\n{reply.get('first_reply', 'N/A')}\n"
            f"{'=' * 50}\n\n"
        )
        count += 1
    prompt += (
        f"GENERATE THE RESPONSE:\n"
        f"Follow the EXACT structure and format from the examples above. "
        f"Use the same field names, formatting, and template structure. "
        f"Replace only the specific values (names, computer IDs, etc.) that are relevant to the new ticket. "
        f"Do NOT add conversational language or explanations."
    )
    return await _async_chat_completion(
        model=rt_module.GEN_MODEL_TEMPLATE,
        messages=[
            {"role": "system", "content": "You are an IT support system that generates responses matching the exact format and style of provided examples."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=rt_module.GEN_MAX_TOKENS_TEMPLATE,
        temperature=rt_module.GEN_TEMPERATURE_TEMPLATE,
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )


async def _async_generate_personal(title, desc, classification, team, confidence, similar_replies, rt_module):
    """Async personalized response generation."""
    import pandas as pd

    context_analysis = rt_module.analyze_ticket_context(title, desc, classification)
    if len(similar_replies) > 2:
        similar_replies = rt_module.select_best_similar_replies(title, desc, similar_replies, top_k=2)

    prompt = (
        f"You are an expert IT support specialist generating a professional first reply. "
        f"Use the context and similar examples to create a response that matches the expected format and tone.\n\n"
        f"CURRENT TICKET:\n"
        f"Title: {title}\n"
        f"Description: {desc}\n"
        f"Request Type: {classification}\n"
        f"Assigned Team: {team} (confidence: {confidence:.2f})\n"
        f"Context: {context_analysis}\n\n"
        f"SIMILAR RESOLVED TICKETS FOR REFERENCE:\n"
    )
    for i, (_, reply) in enumerate(similar_replies.iterrows(), 1):
        prompt += (
            f"{i}. Similar Ticket Title: {reply['Title_anon']}\n"
            f"   Similar Ticket Description: {reply['Description_anon'][:300]}...\n"
            f"   First Reply: {reply['first_reply']}\n\n"
        )
    response_style = rt_module.determine_response_style(similar_replies)
    prompt += (
        f"RESPONSE REQUIREMENTS:\n"
        f"- Style: {response_style}\n"
        f"- Match the tone and structure of similar replies\n"
        f"- Address the specific request clearly and professionally\n"
        f"- Include relevant next steps or requirements\n"
        f"- Keep length appropriate for first reply (150-300 words)\n\n"
        f"Generate a professional first reply that follows the patterns shown in the similar tickets above:"
    )
    return await _async_chat_completion(
        model=rt_module.GEN_MODEL_PERSONAL,
        messages=[
            {"role": "system", "content": "You are an expert IT support specialist who writes clear, professional responses that match organizational standards."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=rt_module.GEN_MAX_TOKENS_PERSONAL,
        temperature=rt_module.GEN_TEMPERATURE_PERSONAL,
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )


async def _async_generate_short(title, desc, classification, team, short_examples, rt_module):
    """Async short personalized response generation.

    Mirrors generate_short_personalized_response() from resolution_task.py:
    - Re-filters the passed DataFrame for len in (30, 400)
    - Truncates reply[:300] and title[:100]
    - Caps at 3 examples
    """
    # Re-filter exactly as the sync version does
    filtered = []
    for _, reply in short_examples.iterrows():
        first_reply = str(reply.get("first_reply", ""))
        if 30 < len(first_reply) < 400:
            filtered.append({
                "reply": first_reply[:300],
                "title": str(reply.get("Title_anon", ""))[:100],
            })
            if len(filtered) >= 3:
                break

    examples_text = ""
    if filtered:
        examples_text = "\n\nExamples of short, personalized responses:\n"
        for i, ex in enumerate(filtered, 1):
            examples_text += f"\nExample {i}:\nTicket: {ex['title']}\nResponse: {ex['reply']}\n"

    prompt = f"""You are an IT support agent writing a SHORT, PERSONALIZED response (maximum 200 characters).

CRITICAL RULES:
- Keep it under 200 characters
- Be direct and concise
- Match the user's language (English/Italian)
- Answer their specific question or acknowledge their request
- NO form fields, NO templates, NO structured lists
- NO email headers or signatures

Current Ticket:
Title: {title}
Description: {desc}
Classification: {classification}
Team: {team}
{examples_text}

Write a SHORT, personalized response (max 200 chars):"""

    try:
        result = await _async_chat_completion(
            model=rt_module.GEN_MODEL_SHORT,
            messages=[
                {"role": "system", "content": "You are an IT support agent who writes very short, personalized responses. Maximum 200 characters. No templates."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=rt_module.GEN_MAX_TOKENS_SHORT,
            temperature=rt_module.GEN_TEMPERATURE_SHORT,
            top_p=0.9,
        )
        if len(result) > 300:
            result = result[:297] + "..."
        return result
    except Exception as e:
        logger.error(f"Error generating short response: {e}")
        return f"Thank you for your message. The {team} team will review your request."


async def _async_generate_status_update(title, desc, classification, team, confidence, similar_replies, rt_module):
    """Async status-update response generation."""
    examples_text = ""
    for i, (_, r) in enumerate(similar_replies.head(3).iterrows(), 1):
        fr = r.get("first_reply", "")
        examples_text += f"{i}. {str(fr)[:400].strip()}\n\n"

    prompt = (
        "You are an IT operations communicator. Produce a clear STATUS UPDATE first reply for users reporting an ongoing "
        "system issue. Follow this structure EXACTLY:\n\n"
        "- Short subject line starting with 'Status Update:'\n"
        "- Opening paragraph acknowledging the incident and sourcing the communication (e.g., 'Yesterday, from Group IT')\n"
        "- Current status (what we know)\n"
        "- Impacted services / example identifiers (invoices, systems)\n"
        "- Known workarounds or temporary steps (if any)\n"
        "- Next steps & ETA (if available) and contact for urgent issues\n"
        "- Short closing\n\n"
        f"TICKET:\nTitle: {title}\nDescription: {desc}\nAssigned Team: {team} (confidence: {confidence:.2f})\n\n"
        f"SIMILAR_EXAMPLES:\n{examples_text}\n\n"
        "Generate the STATUS UPDATE now. Keep it factual, short paragraphs, 120-220 words. If you don't have ETA, say 'investigating' and provide workarounds if present."
    )
    return await _async_chat_completion(
        model=rt_module.GEN_MODEL_STATUS,
        messages=[
            {"role": "system", "content": "You are an IT operations communicator writing concise status updates."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=rt_module.GEN_MAX_TOKENS_STATUS,
        temperature=rt_module.GEN_TEMPERATURE_STATUS,
        top_p=0.8,
    )


# ============================================================================
# ASYNC JUDGE
# ============================================================================
async def async_judge_items(
    title: str,
    description: str,
    similar_replies: List[Dict[str, Any]],
    top_k: int,
    expected_first_reply: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of judge_service.judge_items."""
    from judge_service import _build_judge_prompt, JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_TOPK_MULT

    n = min(len(similar_replies), max(top_k * JUDGE_TOPK_MULT, top_k))
    items = similar_replies[:n]

    if _async_openai_client is None:
        # Fallback heuristic (same as sync version)
        scores = []
        for it in items:
            es = float(it.get("enhanced_score", 0.0))
            helpfulness = max(0.0, min(1.0, es))
            scores.append({"retrieved_id": it.get("retrieved_id"), "helpfulness": helpfulness, "rationale": "heuristic"})
        return {"scores": scores, "used": "fallback"}

    messages = _build_judge_prompt(title, description, items, expected_first_reply=expected_first_reply)

    resp = await _async_openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        temperature=JUDGE_TEMPERATURE,
    )
    content = resp.choices[0].message.content

    scores = []
    try:
        arr = json.loads(content)
        if not isinstance(arr, list):
            raise ValueError("Judge must return a JSON array")
        for obj in arr:
            rid = str(obj.get("retrieved_id"))
            h = float(obj.get("helpfulness", 0.0))
            rationale = str(obj.get("rationale", ""))
            if rid:
                scores.append({"retrieved_id": rid, "helpfulness": max(0.0, min(1.0, h)), "rationale": rationale})
    except Exception as e:
        logger.warning(f"Async judge parse failed: {e}")
        for it in items:
            scores.append({"retrieved_id": it.get("retrieved_id"), "helpfulness": 0.5, "rationale": "parse-fallback"})

    return {"scores": scores, "used": "async_v1"}


# ============================================================================
# ASYNC LLM COMPARISON
# ============================================================================
async def async_llm_compare(
    title: str,
    description: str,
    expected_first_reply: str,
    baseline_resp: str,
    al_resp: str,
) -> dict:
    """Async version of the LLM side-by-side comparison."""
    if _async_openai_client is None:
        return {"better": "tie", "rationale": "no_async_openai_client"}

    user_prompt = (
        f"You are an expert IT support reviewer.\n\n"
        f"TICKET TITLE: {title}\n"
        f"TICKET DESCRIPTION: {description}\n"
        f"EXPECTED EXPERT REPLY:\n{expected_first_reply}\n\n"
        f"CANDIDATE A:\n{baseline_resp}\n\n"
        f"CANDIDATE B:\n{al_resp}\n\n"
        f"Which is more helpful? Respond ONLY with JSON: "
        f'{{"better": "A" or "B" or "tie", "rationale": "short explanation"}}'
    )
    try:
        resp = await _async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a strict support ticket answer evaluator. Return only JSON."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        obj = json.loads(content)
        return {"better": obj.get("better", "tie"), "rationale": obj.get("rationale", "")}
    except Exception as e:
        return {"better": "tie", "rationale": f"error: {e}"}
