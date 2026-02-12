"""
FastAPI wrapper for resolution_task.
Endpoints:
- POST /generate: run full pipeline (classification, retrieval, LLM)
- POST /retrieve: retrieval-only (feedback-aware reranking)
- POST /feedback: record thumbs up/down for a retrieved ticket
- POST /save_ticket: save resolved ticket back to CSV and update embeddings
- POST /rebuild_embeddings: force rebuild of embedding cache

Run:
    uvicorn api_resolution_task:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
import os
import sqlite3
try:
    from judge_service import judge_items, map_scores_to_votes
except Exception:
    judge_items = None
    map_scores_to_votes = None
import pathlib

# Backend imports (existing module)
try:
    import resolution_task
    from resolution_task import (
        process_new_ticket,
        retrieve_only,
        record_feedback,
        save_resolved_ticket_with_feedback,
        rebuild_embeddings,
    )
except Exception:
    resolution_task = None
    process_new_ticket = None
    retrieve_only = None
    record_feedback = None
    save_resolved_ticket_with_feedback = None
    rebuild_embeddings = None

app = FastAPI(title="Resolution Task API", version="1.0.0")

# Basic CORS (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("resolution_api")
logging.basicConfig(level=logging.INFO)

DEFAULT_KB = os.getenv("KNOWLEDGE_BASE_PATH", "tickets_large_first_reply_label_copy.csv")
AUTO_JUDGE = os.getenv("AUTO_JUDGE", "false").lower() in ("1","true","yes")

class GenerateRequest(BaseModel):
    title: str
    description: str
    top_k: int = 5
    knowledge_base_path: Optional[str] = None

class JudgeRequest(BaseModel):
    title: str
    description: str
    predicted_class: Optional[str] = None
    predicted_team: Optional[str] = None
    similar_replies: list
    top_k: int = 5

class JudgeResponse(BaseModel):
    scores: list
    used: Optional[str] = None
    votes_applied: Optional[int] = None

class RetrieveRequest(BaseModel):
    title: str
    description: str
    top_k: int = 5
    knowledge_base_path: Optional[str] = None
    # Optional predicted context to improve feedback-aware reranking
    predicted_class: Optional[str] = None
    predicted_team: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    retrieved_id: str
    label: int  # +1/-1
    predicted_class: Optional[str] = None
    predicted_team: Optional[str] = None
    user_id: Optional[str] = None

class SaveTicketRequest(BaseModel):
    title: str
    description: str
    response: str
    predicted_team: Optional[str] = None
    predicted_classification: Optional[str] = None
    service_name: Optional[str] = None
    service_subcategory: Optional[str] = None
    knowledge_base_path: Optional[str] = None

class RebuildRequest(BaseModel):
    knowledge_base_path: Optional[str] = None

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "resolution-task-api", "version": "1.0.0"}

@app.get("/routes")
def routes() -> Dict[str, Any]:
    return {"routes": [r.path for r in app.router.routes]}

@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {"FEEDBACK_DB_PATH": os.getenv("FEEDBACK_DB_PATH", "feedback.db"), "KNOWLEDGE_BASE_PATH": os.getenv("KNOWLEDGE_BASE_PATH", DEFAULT_KB)}

class SetConfigRequest(BaseModel):
    FEEDBACK_DB_PATH: Optional[str] = None
    KNOWLEDGE_BASE_PATH: Optional[str] = None

@app.post("/config")
def set_config(req: SetConfigRequest) -> Dict[str, Any]:
    changed: Dict[str, Any] = {}
    if req.FEEDBACK_DB_PATH:
        os.environ["FEEDBACK_DB_PATH"] = req.FEEDBACK_DB_PATH
        changed["FEEDBACK_DB_PATH"] = req.FEEDBACK_DB_PATH
    if req.KNOWLEDGE_BASE_PATH:
        os.environ["KNOWLEDGE_BASE_PATH"] = req.KNOWLEDGE_BASE_PATH
        changed["KNOWLEDGE_BASE_PATH"] = req.KNOWLEDGE_BASE_PATH
    return {"changed": changed, "current": {"FEEDBACK_DB_PATH": os.getenv("FEEDBACK_DB_PATH", "feedback.db"), "KNOWLEDGE_BASE_PATH": os.getenv("KNOWLEDGE_BASE_PATH", DEFAULT_KB)}}

@app.get("/feedback_stats")
def feedback_stats(retrieved_id: str, scope_key: Optional[str] = None) -> Dict[str, Any]:
    """Return raw and aggregated feedback counts for a given retrieved_id (and optional scope)."""
    default_db = str(pathlib.Path(__file__).with_name("feedback.db"))
    db_path = os.getenv("FEEDBACK_DB_PATH", default_db)
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback_raw WHERE retrieved_id=?", (retrieved_id,))
        raw_count = cur.fetchone()[0]
        if scope_key:
            cur.execute("SELECT pos,neg FROM feedback_agg WHERE retrieved_id=? AND scope_key=?", (retrieved_id, scope_key))
            agg_rows = cur.fetchall()
        else:
            cur.execute("SELECT scope_key,pos,neg FROM feedback_agg WHERE retrieved_id=?", (retrieved_id,))
            agg_rows = cur.fetchall()
        conn.close()
        return {"db": db_path, "retrieved_id": retrieved_id, "raw_count": raw_count, "agg": agg_rows}
    except Exception as e:
        logger.exception("/feedback_stats error")
        return {"error": str(e)}

@app.get("/generate")
def generate_get(title: str, description: str, top_k: int = 5, knowledge_base_path: Optional[str] = None) -> Dict[str, Any]:
    kb = knowledge_base_path or DEFAULT_KB
    assert process_new_ticket is not None, "backend not available"
    logger.info(f"GET /generate title='{title[:60]}' top_k={top_k} kb={kb}")
    return process_new_ticket(title, description, knowledge_base_path=kb, top_k=top_k)

@app.get("/retrieve")
def retrieve_get(title: str, description: str, top_k: int = 5, knowledge_base_path: Optional[str] = None, predicted_class: Optional[str] = None, predicted_team: Optional[str] = None) -> Dict[str, Any]:
    kb = knowledge_base_path or DEFAULT_KB
    assert retrieve_only is not None, "backend not available"
    # If caller didn't pass predicted context, compute it so retrieval matches the
    # generation pipeline behavior (and so feedback-aware reranking has scope keys).
    pc = predicted_class
    pt = predicted_team
    if (pc is None or pt is None) and resolution_task is not None:
        try:
            ticket_text = f"{title} {description}".strip()
            if pc is None:
                pc, _tpl = resolution_task.classify_ticket(ticket_text)
            if pt is None:
                pt, _conf = resolution_task.classify_team_with_distilbert(ticket_text)
        except Exception as e:
            logger.warning(f"/retrieve pre-classification failed; falling back to caller args: {e}")
            pc = predicted_class
            pt = predicted_team

    logger.info(f"GET /retrieve title='{title[:60]}' top_k={top_k} kb={kb} class={pc} team={pt}")
    return retrieve_only(title, description, knowledge_base_path=kb, top_k=top_k, predicted_class=pc, predicted_team=pt)

@app.post("/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    kb = req.knowledge_base_path or DEFAULT_KB
    assert process_new_ticket is not None, "backend not available"
    logger.info(f"/generate title='{req.title[:60]}' top_k={req.top_k} kb={kb}")
    res = process_new_ticket(req.title, req.description, knowledge_base_path=kb, top_k=req.top_k)
    # Optional auto-judge to boost performance
    if AUTO_JUDGE and 'similar_replies' in res and judge_items and map_scores_to_votes:
        try:
            jr = judge_items(req.title, req.description, res['similar_replies'], req.top_k)
            scores = jr.get('scores', [])
            votes = map_scores_to_votes(scores)
            applied = 0
            for v in votes:
                label = v.get('label', 0)
                if label == 0:
                    continue
                record_feedback(
                    query_id="auto-judge",
                    retrieved_id=str(v.get('retrieved_id')),
                    label=int(label),
                    predicted_class=res.get('predicted_class'),
                    predicted_team=res.get('predicted_team'),
                    user_id="judge"
                )
                applied += 1
            # Re-run retrieval to apply lifts
            rr = retrieve_only(req.title, req.description, knowledge_base_path=kb, top_k=req.top_k, predicted_class=res.get('predicted_class'), predicted_team=res.get('predicted_team'))
            res['similar_replies'] = rr.get('similar_replies', res['similar_replies'])
            res['judge'] = {"scores": scores, "votes_applied": applied}
        except Exception as e:
            logger.warning(f"auto-judge failed: {e}")
    return res

@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
    kb = req.knowledge_base_path or DEFAULT_KB
    assert retrieve_only is not None, "backend not available"
    pc = req.predicted_class
    pt = req.predicted_team
    if (pc is None or pt is None) and resolution_task is not None:
        try:
            ticket_text = f"{req.title} {req.description}".strip()
            if pc is None:
                pc, _tpl = resolution_task.classify_ticket(ticket_text)
            if pt is None:
                pt, _conf = resolution_task.classify_team_with_distilbert(ticket_text)
        except Exception as e:
            logger.warning(f"/retrieve pre-classification failed; falling back to caller args: {e}")
            pc = req.predicted_class
            pt = req.predicted_team

    logger.info(f"/retrieve title='{req.title[:60]}' top_k={req.top_k} kb={kb} class={pc} team={pt}")
    res = retrieve_only(req.title, req.description, knowledge_base_path=kb, top_k=req.top_k, predicted_class=pc, predicted_team=pt)
    if AUTO_JUDGE and 'similar_replies' in res and judge_items and map_scores_to_votes:
        try:
            jr = judge_items(req.title, req.description, res['similar_replies'], req.top_k)
            scores = jr.get('scores', [])
            votes = map_scores_to_votes(scores)
            applied = 0
            for v in votes:
                label = v.get('label', 0)
                if label == 0:
                    continue
                record_feedback(
                    query_id="auto-judge",
                    retrieved_id=str(v.get('retrieved_id')),
                    label=int(label),
                    predicted_class=res.get('predicted_class'),
                    predicted_team=res.get('predicted_team'),
                    user_id="judge"
                )
                applied += 1
            # Re-run retrieval to apply lifts
            res2 = retrieve_only(req.title, req.description, knowledge_base_path=kb, top_k=req.top_k, predicted_class=res.get('predicted_class'), predicted_team=res.get('predicted_team'))
            res['similar_replies'] = res2.get('similar_replies', res['similar_replies'])
            res['judge'] = {"scores": scores, "votes_applied": applied}
        except Exception as e:
            logger.warning(f"auto-judge failed: {e}")
    return res

@app.post("/judge")
def judge(req: JudgeRequest) -> Dict[str, Any]:
    assert judge_items is not None, "judge service unavailable"
    scores = judge_items(req.title, req.description, req.similar_replies, req.top_k)
    return scores

@app.post("/judge_and_retrieve")
def judge_and_retrieve(req: JudgeRequest) -> Dict[str, Any]:
    kb = DEFAULT_KB
    assert judge_items is not None and map_scores_to_votes is not None, "judge service unavailable"
    jr = judge_items(req.title, req.description, req.similar_replies, req.top_k)
    scores = jr.get('scores', [])
    votes = map_scores_to_votes(scores)
    applied = 0
    for v in votes:
        label = v.get('label', 0)
        if label == 0:
            continue
        record_feedback(
            query_id="api-judge",
            retrieved_id=str(v.get('retrieved_id')),
            label=int(label),
            predicted_class=req.predicted_class,
            predicted_team=req.predicted_team,
            user_id="judge"
        )
        applied += 1
    # Re-run retrieve to apply lifts with provided context
    res = retrieve_only(req.title, req.description, knowledge_base_path=kb, top_k=req.top_k, predicted_class=req.predicted_class, predicted_team=req.predicted_team)
    return {"scores": scores, "votes_applied": applied, "retrieved": res}

@app.post("/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    assert record_feedback is not None, "backend not available"
    logger.info(f"/feedback id={req.retrieved_id} label={req.label} class={req.predicted_class} team={req.predicted_team}")
    record_feedback(req.query_id, req.retrieved_id, req.label, req.predicted_class, req.predicted_team, req.user_id)
    return {"ok": True}

@app.post("/save_ticket")
def save_ticket(req: SaveTicketRequest) -> Dict[str, Any]:
    kb = req.knowledge_base_path or DEFAULT_KB
    assert save_resolved_ticket_with_feedback is not None, "backend not available"
    logger.info(f"/save_ticket title='{req.title[:50]}' kb={kb}")
    return save_resolved_ticket_with_feedback(
        ticket_title=req.title,
        ticket_description=req.description,
        edited_response=req.response,
        predicted_team=req.predicted_team,
        predicted_classification=req.predicted_classification,
        service_name=req.service_name,
        service_subcategory=req.service_subcategory,
        knowledge_base_path=kb,
    )

@app.post("/rebuild_embeddings")
def rebuild(req: RebuildRequest) -> Dict[str, Any]:
    kb = req.knowledge_base_path or DEFAULT_KB
    assert rebuild_embeddings is not None, "backend not available"
    logger.info(f"/rebuild_embeddings kb={kb}")
    return rebuild_embeddings(kb)
