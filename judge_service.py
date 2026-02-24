"""
LLM Judge Service: scores how helpful each retrieved ticket is for the current query.
- Separated module; optional usage via API AUTO_JUDGE=true.
- Strict JSON output; safe fallback scoring when LLM unavailable.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import logging

# Optional OpenAI imports; remain functional if unavailable
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_CLIENT_MODE = "unavailable"
_openai_client = None
try:
    # Try new SDK first
    import openai as openai_v1
    if hasattr(openai_v1, "OpenAI"):
        _openai_client = openai_v1.OpenAI(api_key=_OPENAI_API_KEY)
        _OPENAI_CLIENT_MODE = "v1"
    else:
        # Legacy v0.28
        import openai as openai_v028
        _openai_client = openai_v028
        _openAI_CLIENT_MODE = "v028"
except Exception:
    _openai_client = None
    _OPENAI_CLIENT_MODE = "unavailable"

logger = logging.getLogger(__name__ + ".judge")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-3.5-turbo")
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
JUDGE_TOPK_MULT = int(os.getenv("JUDGE_TOPK_MULT", "2"))  # judge up to top_k * 2
# Conservative binary thresholds with a neutral band.
# Default: strong upvote only for very high helpfulness (>=0.8), clear downvote for low (<=0.4),
# and ignore middling scores (0.4 < h < 0.8).
JUDGE_POS_THRESHOLD = float(os.getenv("JUDGE_POS_THRESHOLD", "0.8"))
JUDGE_NEG_THRESHOLD = float(os.getenv("JUDGE_NEG_THRESHOLD", "0.4"))


def _build_judge_prompt(title: str, description: str, items: List[Dict[str, Any]], expected_first_reply: Optional[str] = None) -> List[Dict[str, str]]:
    lines = []
    lines.append("You are a strict support ticket reviewer. Score each retrieved item for helpfulness (0 to 1) for the current ticket.")
    lines.append(
        "You also see the expected (ground-truth) first reply from an expert. "
        "Your job is to reward ONLY items whose content (especially first_reply) is very close in structure "
        "and content to this expected first reply."
    )
    lines.append(
        "Give a HIGH score only when the retrieved ticket would clearly help generate an answer that is almost the "
        "same as the expected first reply (same business process, same key steps, same level of detail)."
    )
    lines.append(
        "Give a LOW score when an item belongs to the wrong team or category, covers a different business process, "
        "is generic, only partially relevant, or would lead to an answer that is noticeably different from the expected reply."
    )
    lines.append(
        "Do NOT give high scores just because something is vaguely helpful; your goal is to maximise similarity to the "
        "expected first reply, not to reward any mildly useful advice."
    )
    lines.append("Respond ONLY with a JSON array of objects: [{\"retrieved_id\": string, \"helpfulness\": number, \"rationale\": string}]\n")
    lines.append(f"TICKET TITLE: {title}\n")
    lines.append(f"TICKET DESCRIPTION: {description}\n")
    if expected_first_reply:
        lines.append("EXPECTED FIRST REPLY (ground truth):\n")
        lines.append(str(expected_first_reply)[:1000] + "\n")
    lines.append("RETRIEVED ITEMS:\n")
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. id={it.get('retrieved_id')}\n   title={it.get('Title_anon','')}\n   description={str(it.get('Description_anon',''))[:300]}\n   first_reply={str(it.get('first_reply',''))[:500]}\n")
    user = "\n".join(lines)
    return [
        {"role": "system", "content": "You are a strict support ticket reviewer. Return only JSON."},
        {"role": "user", "content": user},
    ]

def judge_items(title: str, description: str, similar_replies: List[Dict[str, Any]], top_k: int, expected_first_reply: Optional[str] = None) -> Dict[str, Any]:
    """Return judge scores and mapped votes. Safe fallback if LLM unavailable."""
    # Select subset to judge
    n = min(len(similar_replies), max(top_k * JUDGE_TOPK_MULT, top_k))
    items = similar_replies[:n]
    # Fallback heuristic if LLM unavailable
    if _openai_client is None:
        scores = []
        for it in items:
            # simple heuristic: reuse enhanced_score normalized
            es = float(it.get("enhanced_score", 0.0))
            helpfulness = max(0.0, min(1.0, es))
            scores.append({"retrieved_id": it.get("retrieved_id"), "helpfulness": helpfulness, "rationale": "heuristic"})
        return {"scores": scores, "used": "fallback"}
    # Build prompt and call
    messages = _build_judge_prompt(title, description, items, expected_first_reply=expected_first_reply)
    if _OPENAI_CLIENT_MODE == "v1":
        resp = _openai_client.chat.completions.create(model=JUDGE_MODEL, messages=messages, temperature=JUDGE_TEMPERATURE)
        content = resp.choices[0].message.content
    else:
        resp = _openai_client.ChatCompletion.create(model=JUDGE_MODEL, messages=messages, temperature=JUDGE_TEMPERATURE)
        content = resp.choices[0].message["content"]
    # Parse JSON strictly
    import json
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
        logger.warning(f"Judge parse failed: {e}")
        # fallback: neutral lifts
        for it in items:
            scores.append({"retrieved_id": it.get("retrieved_id"), "helpfulness": 0.5, "rationale": "parse-fallback"})
    return {"scores": scores, "used": _OPENAI_CLIENT_MODE}

def map_scores_to_votes(scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    votes = []
    for s in scores:
        h = float(s.get("helpfulness", 0.0))
        # Conservative mapping with neutral band:
        #   - h >= JUDGE_POS_THRESHOLD  -> upvote (+1)
        #   - h <= JUDGE_NEG_THRESHOLD  -> downvote (-1)
        #   - otherwise                 -> neutral (no vote recorded)
        label: int | None
        if h >= JUDGE_POS_THRESHOLD:
            label = +1
        elif h <= JUDGE_NEG_THRESHOLD:
            label = -1
        else:
            label = None

        votes.append(
            {
                "retrieved_id": s.get("retrieved_id"),
                "label": label if label is not None else 0,
                "helpfulness": h,
                "rationale": s.get("rationale"),
            }
        )
    return votes

