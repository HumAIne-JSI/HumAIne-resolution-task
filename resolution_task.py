"""
IT Support Ticket Resolution System

A RAG (Retrieval-Augmented Generation) system for automated IT support ticket 
response generation. Combines semantic search, DistilBERT classification, and 
GPT-3.5-turbo to generate contextually appropriate responses.

Key Features:
    - FAISS-based semantic similarity search
    - DistilBERT team (23 labels) and ticket type (10 classes) classifiers
    - GPT-3.5-turbo for adaptive response generation
    - Incremental learning with feedback loop
    - Streamlit UI for human review

Usage:
    from resolution_task import process_new_ticket, save_resolved_ticket_with_feedback
    
    result = process_new_ticket("VPN access request", "Need VPN for project ABC")
    print(result['response'])

See README_ResolutionTask.md for full documentation.

Author: JSI Team
License: Proprietary
"""

import os
import re
import pickle
import datetime
import logging
import sqlite3
import traceback
from typing import Optional, Dict, List, Tuple

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
    logger_init = logging.getLogger(__name__)
    logger_init.info("Loaded environment variables from .env file")
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass

# Third-party imports
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths
TEAM_CLASSIFIER_PATH = os.getenv("TEAM_CLASSIFIER_PATH", "./perfect_team_classifier")
TICKET_CLASSIFIER_PATH = os.getenv("TICKET_CLASSIFIER_PATH", "./ticket_classifier_model")

# Embedding configuration
SENTENCE_MODEL_NAME = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "embeddings_cache")

# Default parameters
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_KB_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "tickets_large_first_reply_label_copy.csv")

# LLM configuration
LLM_MODEL = "gpt-3.5-turbo"
LLM_MAX_TOKENS = 400  # Reduced from 800 for speed
LLM_TEMPERATURE = 0.2

# Feedback/reranking configuration (Laplace smoothing + stronger lifts)
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", "feedback_refid.db")
FEEDBACK_ENABLED = bool(os.getenv("FEEDBACK_ENABLED", "True").lower() in ['true', '1', 'yes'])  # TOGGLE for A/B testing
FEEDBACK_ALPHA = float(os.getenv("FEEDBACK_ALPHA", "1.0"))  # Laplace alpha
FEEDBACK_BETA = float(os.getenv("FEEDBACK_BETA", "1.0"))    # Laplace beta
# Require more total votes before feedback reaches full strength.
# OPTIMIZED - Lowered from 10 to 5 for better coverage
FEEDBACK_MIN_COUNT = int(os.getenv("FEEDBACK_MIN_COUNT", "2"))  # Lowered to increase % of tickets affected
# Slightly more balanced weights across scopes so no single scope dominates.
# Rebalanced feedback weights:
# Global feedback can become a "popularity prior" and hurt on strong retrieval.
# Prefer class/team-scoped signal when available; keep global as a small backstop.
FEEDBACK_W_GLOBAL = float(os.getenv("FEEDBACK_W_GLOBAL", "0.10"))
FEEDBACK_W_CLASS = float(os.getenv("FEEDBACK_W_CLASS", "0.55"))
FEEDBACK_W_TEAM = float(os.getenv("FEEDBACK_W_TEAM", "0.35"))
# Cap how much feedback can move the score; keep it as a bias, not a full override.
# FIXED: Lowered from 1.5 to 0.5 - feedback was over-valuing popular tickets
FEEDBACK_TOTAL_LIFT_CAP = float(os.getenv("FEEDBACK_TOTAL_LIFT_CAP", "0.5"))  # FIXED: Was 1.5, causing popular tickets to overtake perfect FAISS matches
FEEDBACK_LIFT_MULT = float(os.getenv("FEEDBACK_LIFT_MULT", "1.5"))  # FIXED: Was 4.0, reduced to prevent demoting high FAISS scores
FEEDBACK_POS_BOOST = float(os.getenv("FEEDBACK_POS_BOOST", "0.10"))  # Reduced from 0.10
FEEDBACK_LOG_LEVEL = os.getenv("FEEDBACK_LOG_LEVEL", "INFO").upper()
FEEDBACK_LOG_TOPN = int(os.getenv("FEEDBACK_LOG_TOPN", "5"))
# Pre-FAISS feedback boost: DISABLED - was breaking retrieval with insufficient data
FEEDBACK_PRE_SEARCH = bool(os.getenv("FEEDBACK_PRE_SEARCH", "false").lower() in ['true', '1', 'yes'])

# ============================================================================
# INJECTION CONTROL (NEW)
# ============================================================================
# Allow disabling feedback injection while keeping re-ranking
# Set to 0 to test if injection is hurting performance
FEEDBACK_MAX_INJECT = int(os.getenv("FEEDBACK_MAX_INJECT", "0"))  # 0 = no injection, only re-rank

# ============================================================================
# CONFIDENCE-BASED GATING (CORRECTED 2026-01-30 - LOGIC INVERTED)
# ============================================================================

#
# IMPLEMENTATION:
# - Gate OFF AL when baseline cosine < 0.60 (weak retrieval, 59% of tickets)
# - Activate AL when baseline cosine ≥ 0.60 (strong retrieval, 41% of tickets)
# - Expected improvement: +1.5% over baseline (vs previous -1.6%)
#
FEEDBACK_CONFIDENCE_GATE = bool(os.getenv("FEEDBACK_CONFIDENCE_GATE", "true").lower() in ['true', '1', 'yes'])
FEEDBACK_CONFIDENCE_THRESHOLD = float(os.getenv("FEEDBACK_CONFIDENCE_THRESHOLD", "0.60"))  # CORRECTED: Now gates OFF below threshold

# ============================================================================
# RELEVANCE-FILTERED FEEDBACK (NEW)
# ============================================================================
# To prevent "globally popular but irrelevant" items from degrading AL performance,
# we now filter feedback items by semantic similarity to the current query.
# Only items with cosine(query, feedback_item) >= threshold are injected/boosted.
FEEDBACK_RELEVANCE_THRESHOLD = float(os.getenv("FEEDBACK_RELEVANCE_THRESHOLD", "0.75"))
# Recommended values:
#   0.65 (default): Permissive - allows moderately related items
#   0.70-0.75: Balanced - good signal-to-noise ratio
#   0.80+: Strict - only very similar items

# ============================================================================
# TIERED ACTIVE LEARNING
# ============================================================================
# Tiered Active Learning (AL) policy driven by baseline retrieval quality proxy.
# Goal: avoid harming already-strong retrieval, add small AL influence for mid retrieval,
# and allow full AL for weak retrieval.
AL_TIER_ENABLED = bool(os.getenv("AL_TIER_ENABLED", "true").lower() in ['true', '1', 'yes'])

# Thresholds operate on a retrieval-quality proxy in [0, 1] derived from FAISS inner-product scores.
# (For normalized embeddings, IP ~= cosine similarity.)
# Default tiers tuned from our analysis:
# - AL tends to help when baseline retrieval is weak.
# - AL can hurt when baseline retrieval is already very strong.
# So we only fully disable AL at extremely high similarity, and we
# enable full-strength AL earlier for weak retrieval.
AL_TIER_STRONG_THRESHOLD = float(os.getenv("AL_TIER_STRONG_THRESHOLD", "0.93"))
AL_TIER_WEAK_THRESHOLD = float(os.getenv("AL_TIER_WEAK_THRESHOLD", "0.82"))

# How many top feedback items to inject per tier.
AL_TIER_INJECT_STRONG = int(os.getenv("AL_TIER_INJECT_STRONG", "0"))
AL_TIER_INJECT_MID = int(os.getenv("AL_TIER_INJECT_MID", "3"))
AL_TIER_INJECT_WEAK = int(os.getenv("AL_TIER_INJECT_WEAK", "8"))

# Scale how strongly feedback can move scores per tier (multiplies computed lift and pos_boost).
AL_TIER_LIFT_SCALE_STRONG = float(os.getenv("AL_TIER_LIFT_SCALE_STRONG", "0.0"))
AL_TIER_LIFT_SCALE_MID = float(os.getenv("AL_TIER_LIFT_SCALE_MID", "0.35"))
AL_TIER_LIFT_SCALE_WEAK = float(os.getenv("AL_TIER_LIFT_SCALE_WEAK", "1.0"))


def _al_tier_for_quality(q: float | None) -> tuple[str, float, int]:
    """Return (tier_name, lift_scale, inject_top_n) for a given retrieval quality proxy."""
    # Default to "mid" if missing.
    if q is None or not isinstance(q, (int, float)):
        return "mid", AL_TIER_LIFT_SCALE_MID, AL_TIER_INJECT_MID
    if q >= AL_TIER_STRONG_THRESHOLD:
        return "strong", AL_TIER_LIFT_SCALE_STRONG, AL_TIER_INJECT_STRONG
    if q < AL_TIER_WEAK_THRESHOLD:
        return "weak", AL_TIER_LIFT_SCALE_WEAK, AL_TIER_INJECT_WEAK
    return "mid", AL_TIER_LIFT_SCALE_MID, AL_TIER_INJECT_MID

# Dedicated feedback logger for clearer diagnostics
feedback_logger = logging.getLogger(__name__ + ".feedback")
try:
    feedback_logger.setLevel(getattr(logging, FEEDBACK_LOG_LEVEL, logging.INFO))
except Exception:
    feedback_logger.setLevel(logging.INFO)

def _feedback_init_db(db_path: str = FEEDBACK_DB_PATH) -> None:
    """Create feedback tables if they do not exist."""
    feedback_logger.debug(f"Initializing feedback DB at: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_raw (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                retrieved_id TEXT,
                predicted_class TEXT,
                predicted_team TEXT,
                label INTEGER, -- +1 / -1
                user_id TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_agg (
                retrieved_id TEXT,
                scope_key TEXT,
                pos INTEGER,
                neg INTEGER,
                last_ts DATETIME,
                PRIMARY KEY (retrieved_id, scope_key)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def record_feedback(
    query_id: str,
    retrieved_id: str,
    label: int,
    predicted_class: Optional[str] = None,
    predicted_team: Optional[str] = None,
    user_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Persist a single thumbs up/down event and update aggregates.

    Note: LLM-judge derived feedback is already made softer/conservative at the
    source (via thresholds and a neutral band in judge_service.map_scores_to_votes),
    and overall influence is further limited by FEEDBACK_TOTAL_LIFT_CAP and
    FEEDBACK_MIN_COUNT. Human feedback is treated the same way here but will
    typically be much sparser and therefore carry relatively more signal.
    """
    # Resolve DB path at call time to honor current environment
    db_path = db_path or os.getenv("FEEDBACK_DB_PATH", FEEDBACK_DB_PATH)
    _feedback_init_db(db_path)
    feedback_logger.info(
        f"Feedback received: query_id={query_id} retrieved_id={retrieved_id} label={label} "
        f"class={predicted_class} team={predicted_team} user={user_id}"
    )
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO feedback_raw(query_id, retrieved_id, predicted_class, predicted_team, label, user_id) VALUES (?,?,?,?,?,?)",
            (query_id, retrieved_id, predicted_class, predicted_team, int(label), user_id),
        )
        def _update(scope_key: str) -> None:
            cur.execute(
                "SELECT pos,neg FROM feedback_agg WHERE retrieved_id=? AND scope_key=?",
                (retrieved_id, scope_key),
            )
            row = cur.fetchone()
            pos, neg = (row or (0, 0))
            before = (pos, neg)
            if label > 0:
                pos += 1
            else:
                neg += 1
            cur.execute(
                "REPLACE INTO feedback_agg(retrieved_id, scope_key, pos, neg, last_ts) VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                (retrieved_id, scope_key, pos, neg),
            )
            feedback_logger.debug(
                f"Updated agg for scope={scope_key} id={retrieved_id}: {before} -> (pos={pos}, neg={neg})"
            )
        # Update aggregates by scope.
        # We include class/team scopes to avoid global popularity overriding relevance,
        # but keep global as a small backstop.
        _update("global")
        if predicted_class:
            _update(f"class:{predicted_class}")
        if predicted_team:
            _update(f"team:{predicted_team}")
        conn.commit()
    finally:
        conn.close()

def _get_top_feedback_items(
    predicted_class: Optional[str],
    predicted_team: Optional[str],
    top_n: int = 10,
    db_path: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    knowledge_base: Optional[pd.DataFrame] = None,
    embeddings: Optional[np.ndarray] = None,
    relevance_threshold: float = 0.65,
) -> List[Tuple[str, float]]:
    """Get top-rated items by feedback score, FILTERED by semantic relevance to query.
    
    NEW: Now requires query_embedding + KB to compute relevance of feedback items
    to current query. Only returns items semantically similar to query (cosine > threshold).
    
    This prevents "globally popular but irrelevant" items from being injected.
    
    Args:
        query_embedding: Current query embedding (for relevance filtering)
        knowledge_base: DataFrame with ticket data (for lookup)
        embeddings: All KB embeddings (for relevance computation)
        relevance_threshold: Min cosine similarity to include item (default 0.65)
    
    Returns list of (retrieved_id, feedback_score) tuples for items with positive feedback
    that are semantically relevant to the query.
    """
    db_path = db_path or os.getenv("FEEDBACK_DB_PATH", FEEDBACK_DB_PATH)
    _feedback_init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # CRITICAL FIX: Get ALL feedback from global scope only
        # Do NOT filter by class/team - relevance filtering will handle that
        # This allows feedback from any ticket type to be considered if semantically relevant
        
        cur.execute(
            "SELECT retrieved_id, pos, neg FROM feedback_agg WHERE scope_key='global' AND pos > 0 ORDER BY (pos - neg) DESC LIMIT ?",
            (top_n * 10,)  # Get many candidates for relevance filtering to choose from
        )
        
        item_scores = {}
        for retrieved_id, pos, neg in cur.fetchall():
            # Simple positive ratio as score
            item_scores[retrieved_id] = (pos - neg) / max(1, pos + neg)
        
        # NEW: Filter by semantic relevance to query
        if query_embedding is not None and knowledge_base is not None and embeddings is not None:
            relevance_filtered = []
            for retrieved_id, score in item_scores.items():
                try:
                    # Find item in KB — try Ref first, then Ticket_Reference, then index
                    mask = None
                    if 'Ref' in knowledge_base.columns:
                        mask = knowledge_base['Ref'].astype(str) == str(retrieved_id)
                    if mask is None or not mask.any():
                        if 'Ticket_Reference' in knowledge_base.columns:
                            mask = knowledge_base['Ticket_Reference'] == retrieved_id
                    if mask is None or not mask.any():
                        mask = knowledge_base.index.astype(str) == str(retrieved_id)
                    
                    if mask.any():
                        kb_idx = knowledge_base[mask].index[0]
                        item_embedding = embeddings[kb_idx:kb_idx+1]
                        
                        # Compute cosine similarity
                        similarity = float(np.dot(query_embedding.flatten(), item_embedding.flatten()) / 
                                         (np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)))
                        
                        if similarity >= relevance_threshold:
                            relevance_filtered.append((retrieved_id, score, similarity))
                            feedback_logger.debug(
                                f"Feedback item {retrieved_id}: score={score:.3f}, relevance={similarity:.3f} ✓"
                            )
                        else:
                            feedback_logger.debug(
                                f"Feedback item {retrieved_id}: score={score:.3f}, relevance={similarity:.3f} ✗ (filtered)"
                            )
                except Exception as e:
                    feedback_logger.debug(f"Failed relevance check for {retrieved_id}: {e}")
            
            # Sort by feedback score (relevance already filtered)
            relevance_filtered.sort(key=lambda x: x[1], reverse=True)
            result = [(item_id, score) for item_id, score, _ in relevance_filtered[:top_n]]
            
            if result:
                feedback_logger.info(
                    f"Relevance filtering: {len(item_scores)} candidates → {len(result)} relevant "
                    f"(threshold={relevance_threshold:.2f})"
                )
            return result
        else:
            # Fallback: no relevance filtering (old behavior)
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:top_n]
    finally:
        conn.close()

def _feedback_lift_for(
    retrieved_id: str,
    predicted_class: Optional[str],
    predicted_team: Optional[str],
    db_path: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    knowledge_base: Optional[pd.DataFrame] = None,
    embeddings: Optional[np.ndarray] = None,
    relevance_threshold: float = 0.65,
) -> float:
    """Compute feedback lift ONLY if ticket is semantically relevant to query.
    
    CRITICAL FIX: Now requires query context to check relevance.
    If the feedback ticket isn't semantically similar to the current query (cosine < threshold),
    return 0.0 lift (ignore its votes).
    
    This prevents popular-but-irrelevant tickets from being boosted.
    """
    # Quick exit if feedback is disabled (for baseline comparisons)
    if not FEEDBACK_ENABLED:
        return 0.0
    
    # CRITICAL: Check semantic relevance FIRST before using votes
    if query_embedding is not None and knowledge_base is not None and embeddings is not None:
        try:
            # Find ticket in KB — try Ref first, then Ticket_Reference, then index
            mask = None
            if 'Ref' in knowledge_base.columns:
                mask = knowledge_base['Ref'].astype(str) == str(retrieved_id)
            if mask is None or not mask.any():
                if 'Ticket_Reference' in knowledge_base.columns:
                    mask = knowledge_base['Ticket_Reference'] == retrieved_id
            if mask is None or not mask.any():
                mask = knowledge_base.index.astype(str) == str(retrieved_id)
            
            if mask.any():
                kb_idx = knowledge_base[mask].index[0]
                item_embedding = embeddings[kb_idx:kb_idx+1]
                
                # Compute cosine similarity to query
                similarity = float(np.dot(query_embedding.flatten(), item_embedding.flatten()) / 
                                 (np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)))
                
                # If not relevant, ignore feedback entirely
                if similarity < relevance_threshold:
                    feedback_logger.debug(
                        f"Feedback for {retrieved_id} IGNORED: relevance={similarity:.3f} < threshold={relevance_threshold:.2f}"
                    )
                    return 0.0
                
                feedback_logger.debug(
                    f"Feedback for {retrieved_id} APPLIED: relevance={similarity:.3f} >= threshold={relevance_threshold:.2f}"
                )
            else:
                # Ticket not found in KB - can't check relevance, ignore feedback
                feedback_logger.debug(f"Feedback for {retrieved_id} IGNORED: ticket not found in KB")
                return 0.0
                
        except Exception as e:
            feedback_logger.debug(f"Failed relevance check for {retrieved_id}: {e}, ignoring feedback")
            return 0.0
    else:
        # No query context provided - can't check relevance, ignore feedback for safety
        feedback_logger.debug(f"Feedback for {retrieved_id} IGNORED: no query context for relevance check")
        return 0.0
    
    # Passed relevance check - now look up votes
    db_path = db_path or os.getenv("FEEDBACK_DB_PATH", FEEDBACK_DB_PATH)
    _feedback_init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # ONLY use global scope - semantic relevance already verified above
        cur.execute(
            "SELECT pos,neg FROM feedback_agg WHERE retrieved_id=? AND scope_key='global'",
            (retrieved_id,),
        )
        row = cur.fetchone()
        pos, neg = (row or (0, 0))
        
        # Bayesian smoothing
        p = (pos + FEEDBACK_ALPHA) / (pos + neg + FEEDBACK_ALPHA + FEEDBACK_BETA)
        scale = min(1.0, (pos + neg) / max(1, FEEDBACK_MIN_COUNT))
        lift = (p - 0.5) * scale * FEEDBACK_LIFT_MULT
        
        # Cap individual lift
        if lift > FEEDBACK_TOTAL_LIFT_CAP:
            lift = FEEDBACK_TOTAL_LIFT_CAP
        elif lift < -FEEDBACK_TOTAL_LIFT_CAP:
            lift = -FEEDBACK_TOTAL_LIFT_CAP
        
        feedback_logger.debug(
            f"Feedback lift for {retrieved_id}: pos={pos} neg={neg} p={p:.4f} scale={scale:.4f} lift={lift:.4f} [RELEVANT]"
        )
        return lift
    finally:
        conn.close()

# ============================================================================
# OPENAI CLIENT INITIALIZATION
# ============================================================================
# OpenAI API configuration - set OPENAI_API_KEY environment variable
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_CLIENT_MODE = "unavailable"
_OPENAI_INIT_ERROR = None  # Store initialization error message
# Try to detect and initialize OpenAI client (supports both v0.28 and v1.0+)
try:
    import openai
    
    # Check OpenAI version to determine which API to use
    openai_version = getattr(openai, '__version__', '0.0.0')
    major_version = int(openai_version.split('.')[0])
    
    if major_version >= 1:
        # New SDK style (>=1.0.0)
        from openai import OpenAI as _NewOpenAIClient
        _openai_client = _NewOpenAIClient(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else _NewOpenAIClient()
        _OPENAI_CLIENT_MODE = "new_v1"
        
        def _chat_completion(model: str, messages: list, **kwargs) -> str:
            """Call OpenAI chat completion API (new SDK v1.0+)."""
            resp = _openai_client.chat.completions.create(model=model, messages=messages, **kwargs)
            return resp.choices[0].message.content.strip()
        
        logger.info(f"OpenAI client initialized: v{openai_version} (new API)")
    else:
        # Legacy SDK (<1.0.0)
        if _OPENAI_API_KEY:
            openai.api_key = _OPENAI_API_KEY
        _OPENAI_CLIENT_MODE = "legacy_v0"
        
        def _chat_completion(model: str, messages: list, **kwargs) -> str:
            """Call OpenAI chat completion API (legacy SDK v0.28)."""
            resp = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
            return resp["choices"][0]["message"]["content"].strip()
        
        logger.info(f"OpenAI client initialized: v{openai_version} (legacy API)")
        
except ImportError:
    # OpenAI package not installed
    _OPENAI_CLIENT_MODE = "unavailable"
    _OPENAI_INIT_ERROR = "OpenAI package not installed"
    
    def _chat_completion(model: str, messages: list, **kwargs) -> str:
        """Fallback when OpenAI package not installed."""
        return "[OpenAI client unavailable – install 'openai' package: pip install openai]"
    
    logger.warning("OpenAI package not installed")

except Exception as init_error:
    # Other initialization errors
    _OPENAI_CLIENT_MODE = "error"
    _OPENAI_INIT_ERROR = str(init_error)
    
    def _chat_completion(model: str, messages: list, **kwargs) -> str:
        """Fallback when OpenAI initialization fails."""
        return f"[OpenAI client error: {_OPENAI_INIT_ERROR}]"
    
    logger.error(f"OpenAI client initialization error: {init_error}")

# ============================================================================
# GENERATION CONFIG (env overrides)
# ============================================================================
# Goal: allow the API pipeline to match the legacy (no-feedback) generator behavior
# without importing the old module. Defaults are chosen to be closer to the legacy
# template-style generation (gpt-4 + larger token budget).

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val.strip() if isinstance(val, str) and val.strip() else default


# Model defaults: legacy used gpt-4 for template generation; we expose both paths.
GEN_MODEL_TEMPLATE = _env_str("GEN_MODEL_TEMPLATE", "gpt-4")
GEN_MODEL_PERSONAL = _env_str("GEN_MODEL_PERSONAL", "gpt-4")
GEN_MODEL_SHORT = _env_str("GEN_MODEL_SHORT", "gpt-4")
GEN_MODEL_STATUS = _env_str("GEN_MODEL_STATUS", "gpt-4")

# Token/temperature defaults tuned to be close to legacy.
GEN_MAX_TOKENS_TEMPLATE = _env_int("GEN_MAX_TOKENS_TEMPLATE", 800)
GEN_MAX_TOKENS_PERSONAL = _env_int("GEN_MAX_TOKENS_PERSONAL", 800)
GEN_MAX_TOKENS_SHORT = _env_int("GEN_MAX_TOKENS_SHORT", 150)
GEN_MAX_TOKENS_STATUS = _env_int("GEN_MAX_TOKENS_STATUS", 300)

GEN_TEMPERATURE_TEMPLATE = _env_float("GEN_TEMPERATURE_TEMPLATE", 0.2)
GEN_TEMPERATURE_PERSONAL = _env_float("GEN_TEMPERATURE_PERSONAL", 0.2)
GEN_TEMPERATURE_SHORT = _env_float("GEN_TEMPERATURE_SHORT", 0.7)
GEN_TEMPERATURE_STATUS = _env_float("GEN_TEMPERATURE_STATUS", 0.2)

# LAZY LOADING: Initialize model references to None
# Models will be loaded only when first needed, reducing startup time
_team_classifier = None
_team_tokenizer = None
_device = None

def _get_team_classifier():
    """Lazy load team classifier model on first use.
    
    Returns:
        tuple: (model, tokenizer, device) or (None, None, None) if loading fails
    """
    global _team_classifier, _team_tokenizer, _device
    
    if _team_classifier is not None:
        return _team_classifier, _team_tokenizer, _device
    
    try:
        logger.info(f"Loading team classifier from {TEAM_CLASSIFIER_PATH}")
        _team_tokenizer = DistilBertTokenizer.from_pretrained(TEAM_CLASSIFIER_PATH)
        _team_classifier = DistilBertForSequenceClassification.from_pretrained(TEAM_CLASSIFIER_PATH)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _team_classifier.to(_device)
        _team_classifier.eval()
        
        logger.info(f"Team classifier loaded: {_team_classifier.num_labels} labels, device: {_device}")
        
        return _team_classifier, _team_tokenizer, _device
    except Exception as e:
        logger.error(f"Error loading team classifier: {e}")
        return None, None, None

# Legacy references for backward compatibility (will lazy load on access)
tokenizer = None
classifier = None
device = None

TEAM_MAPPING = {
    0: "(GI-CF) Security & RPA",
    1: "(GI-CyberSec) Security Operation Center",
    2: "(GI-IaaS) Admin - License & Asset Management",
    3: "(GI-IaaS) Admin - Local IT purchase",
    4: "(GI-IaaS) Backend Application Srv. & Project Support",
    5: "(GI-IaaS) Development Platform",
    6: "(GI-IaaS) Network On-Prem (LAN,WLAN,WAN 2nd level)",
    7: "(GI-SM) Service Desk",
    8: "(GI-SaaS) SAP & Synertrade",
    9: "(GI-SaaS) Salesforce",
    10: "(GI-UX) Account Management",
    11: "(GI-UX) Application",
    12: "(GI-UX) Group",
    13: "(GI-UX) Network Access",
    14: "(GI-UX) Office365 & MS-Teams",
    15: "(GI-UX) Unified Communication",
    16: "(GI-UX) Windows"
}

def classify_team_with_distilbert(ticket_text, service_category=None, service_subcategory=None):
    """Use trained DistilBERT model to predict which team should handle the ticket"""
    
    # Lazy load the model on first use
    classifier, tokenizer, device = _get_team_classifier()
    
    if classifier is None or tokenizer is None:
        print("⚠️ Team classifier not available, using fallback")
        return "Unknown Team", 0.0
    
    # Parse the input properly
    lines = ticket_text.split('\n', 1) if '\n' in ticket_text else ticket_text.split(' ', 1)
    title = lines[0].strip() if lines else ""
    description = lines[1].strip() if len(lines) > 1 else ticket_text.strip()
    
    # Format input EXACTLY like in the notebook training - this is critical!
    formatted_text = f"[TITLE] {title} [DESCRIPTION] {description}"
    
    # Add service information exactly as in training
    if service_category:
        formatted_text += f" [SERVICE CATEGORY] {service_category}"
    if service_subcategory:
        formatted_text += f" [SERVICE SUBCATEGORY] {service_subcategory}"
    
    print(f"📋 DistilBERT input: {formatted_text[:150]}...")
    
    try:
        # Use EXACT same tokenization parameters as training
        inputs = tokenizer(
            formatted_text, 
            return_tensors="pt", 
            truncation=True, 
            padding='max_length',  # Changed from padding=True
            max_length=512
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = classifier(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities.max().item()
        
        # ALWAYS use label encoder first - this is the most accurate approach
        try:
            with open('./perfect_team_classifier/label_encoder.pkl', 'rb') as f:
                team_label_encoder = pickle.load(f)
            predicted_team = team_label_encoder.inverse_transform([predicted_class_idx])[0]
            print(f"🎯 Predicted team: {predicted_team} (confidence: {confidence:.3f}) [Using CSV label encoder]")
        except Exception as le_error:
            print(f"⚠️ Label encoder error, using TEAM_MAPPING fallback: {le_error}")
            predicted_team = TEAM_MAPPING.get(predicted_class_idx, "(GI-SM) Service Desk")
            print(f"🎯 Predicted team: {predicted_team} (confidence: {confidence:.3f}) [Using TEAM_MAPPING fallback]")
        
        return predicted_team, confidence
        
    except Exception as e:
        print(f"❌ Error in DistilBERT team classification: {e}")
        return "(GI-SM) Service Desk", 0.5  # Default fallback


# Load the knowledge base (tickets_large_first_reply_label_copy.csv)
def load_knowledge_base(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Public_log_anon'])  # Drop rows without public logs

    # Extract the first reply from the Public_log_anon column
    def extract_first_reply(text):
        if pd.isna(text):
            return None
        
        text_str = str(text)
        
        # More aggressive pattern to find timestamp separators and user info
        timestamp_pattern = r'\*{10,}\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
        
        # Also look for patterns like ": servicedesk (0) ************" or ": Name (ID) ************"
        user_pattern = r':\s*[^(]*\([^)]*\)\s*\*{10,}'
        
        # Split by timestamp pattern first
        parts = re.split(timestamp_pattern, text_str)
        
        if len(parts) >= 2:
            first_reply = parts[1].strip()
            
            # Remove user info pattern at the beginning
            first_reply = re.sub(user_pattern, '', first_reply, flags=re.IGNORECASE).strip()
            
            # Find the next timestamp to cut off subsequent replies
            next_timestamp_match = re.search(timestamp_pattern, first_reply)
            if next_timestamp_match:
                first_reply = first_reply[:next_timestamp_match.start()].strip()
            
            # Find next user pattern to cut off subsequent replies
            next_user_match = re.search(user_pattern, first_reply)
            if next_user_match:
                first_reply = first_reply[:next_user_match.start()].strip()
            
            # Clean up common artifacts
            first_reply = re.sub(r'^-+\s*', '', first_reply)  # Remove leading dashes
            first_reply = re.sub(r'\s*-+$', '', first_reply)  # Remove trailing dashes
            first_reply = re.sub(r'^\s*Dear\s+[^,]+,?\s*', '', first_reply, flags=re.IGNORECASE)  # Remove "Dear Name," at start
            
            # Remove lines that are just separators or user info
            lines = first_reply.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Skip lines that are just separators or user info
                if not re.match(r'^-{5,}$', line) and not re.match(r'^\s*:\s*[^(]*\([^)]*\)', line):
                    cleaned_lines.append(line)
            
            first_reply = '\n'.join(cleaned_lines).strip()
            
            # Return only if it's substantial (not just whitespace or artifacts)
            return first_reply if len(first_reply) > 50 else None
        
        return text_str[:500] if len(text_str) > 50 else None

    df['first_reply'] = df['Public_log_anon'].apply(extract_first_reply)
    df = df.dropna(subset=['first_reply'])  # Drop rows without first replies

    return df

# Build a SentenceTransformer-based RAG system with FAISS
class RAGSystem:
    def __init__(self, knowledge_base, sentence_model_name="all-MiniLM-L6-v2", kb_path: str | None = None):
        self.knowledge_base = knowledge_base
        self.sentence_model_name = sentence_model_name
        self.sentence_model = SentenceTransformer(sentence_model_name)
        self.index = None
        self.embeddings = None
        self.title_embeddings = None
        self.description_embeddings = None
        self.category_index = {}
        self.kb_path = kb_path

    def build_index(self, kb_path: str | None = None, kb_mtime: float | None = None):
        """Build or load cached embeddings + FAISS index. 

        Caching strategy:
        - Cache file stored under embeddings_cache/<basename>_<model>_<mtime>.npz
        - If cache exists, load arrays and rebuild FAISS index (fast)
        - If DISABLE_EMBEDDING_CACHE env var is set to '1', force rebuild
        """
        if self.index is not None and self.embeddings is not None:
            return  # Already built in-memory

        kb_path = kb_path or self.kb_path
        cache_disabled = os.getenv("DISABLE_EMBEDDING_CACHE") == "1"
        cache_dir = "embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = None
        if kb_path and kb_mtime:
            safe_model = self.sentence_model_name.replace('/', '_')
            cache_key = f"{os.path.basename(kb_path)}_{safe_model}_{int(kb_mtime)}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

        # Try loading cache
        if cache_file and not cache_disabled and os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=False)
                self.title_embeddings = data['title_embeddings']
                self.description_embeddings = data['description_embeddings']
                self.embeddings = data['embeddings']
                # Rebuild FAISS index
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings)
                self._build_category_index()
                print(f"⚡ Loaded embeddings cache from {cache_file}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load embedding cache ({e}), rebuilding...")

        # Build fresh embeddings
        titles = self.knowledge_base['Title_anon'].fillna('').tolist()
        descriptions = self.knowledge_base['Description_anon'].fillna('').tolist()
        print("🔄 Building embeddings (no valid cache)...")
        self.title_embeddings = self.sentence_model.encode(titles, show_progress_bar=True)
        self.description_embeddings = self.sentence_model.encode(descriptions, show_progress_bar=True)
        texts = [f"{title} {desc}".strip() for title, desc in zip(titles, descriptions)]
        self.embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype('float32')
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self._build_category_index()
        print("✅ Enhanced index built successfully")

        # Persist cache
        if cache_file and not cache_disabled:
            try:
                np.savez_compressed(
                    cache_file,
                    title_embeddings=self.title_embeddings,
                    description_embeddings=self.description_embeddings,
                    embeddings=self.embeddings,
                )
                print(f"💾 Saved embeddings cache to {cache_file}")
            except Exception as e:
                print(f"⚠️ Failed to save embeddings cache: {e}")

    def _build_category_index(self):
        """Build index by categories for faster filtering"""
        if 'label_auto' in self.knowledge_base.columns:
            for category in self.knowledge_base['label_auto'].unique():
                if pd.notna(category):
                    mask = self.knowledge_base['label_auto'] == category
                    self.category_index[category] = self.knowledge_base[mask].index.tolist()

    def retrieve_similar_replies(self, query, top_k=5, predicted_category=None, predicted_class: Optional[str] = None, predicted_team: Optional[str] = None):
        """Enhanced retrieval with multi-factor scoring and feedback-aware re-ranking."""
        _feedback_init_db()
        # Get larger candidate set for re-ranking
        search_k = min(max(top_k * 4, top_k), len(self.knowledge_base)) ##### MODIFIED from top_k * 6
        feedback_logger.debug(
            f"Retrieval start: top_k={top_k} search_k={search_k} class={predicted_class} team={predicted_team}"
        )
        
        # Step 0: PRE-FAISS FEEDBACK BOOST - adjust query embedding based on positive feedback
        query_embedding = self.sentence_model.encode([query])

        # NOTE: We select a tier after we have baseline FAISS scores.
        # Pre-search feedback-boost can still be helpful, but we only do it if
        # feedback is enabled and tiering isn't enabled (tiering uses injection + lift scaling instead).
        if FEEDBACK_PRE_SEARCH and FEEDBACK_ENABLED and not AL_TIER_ENABLED:
            top_feedback = _get_top_feedback_items(
                predicted_class, 
                predicted_team, 
                top_n=5,
                query_embedding=query_embedding,
                knowledge_base=self.knowledge_base,
                embeddings=self.embeddings,
                relevance_threshold=FEEDBACK_RELEVANCE_THRESHOLD,
            )
            if top_feedback:
                feedback_logger.info(f"Pre-FAISS boost: incorporating {len(top_feedback)} top-rated items into query")
                # Get embeddings of top-rated items and blend with query
                boost_weight = 0.15  # How much to shift query toward positive examples
                for item_id, score in top_feedback:
                    try:
                        # Find item in knowledge base — try Ref first, then Ticket_Reference, then index
                        mask = None
                        if 'Ref' in self.knowledge_base.columns:
                            mask = self.knowledge_base['Ref'].astype(str) == str(item_id)
                        if mask is None or not mask.any():
                            if 'Ticket_Reference' in self.knowledge_base.columns:
                                mask = self.knowledge_base['Ticket_Reference'] == item_id
                        if mask is None or not mask.any():
                            mask = self.knowledge_base.index.astype(str) == item_id
                        
                        if mask.any():
                            kb_idx = self.knowledge_base[mask].index[0]
                            item_embedding = self.embeddings[kb_idx:kb_idx+1]
                            # Weighted blend: move query slightly toward this positive example
                            query_embedding = query_embedding + (boost_weight * score * item_embedding)
                    except Exception as e:
                        feedback_logger.debug(f"Failed to boost with item {item_id}: {e}")
                
                # Re-normalize after blending
                faiss.normalize_L2(query_embedding)
                feedback_logger.debug("Query embedding adjusted with pre-search feedback boost")
        
        # Step 1: Get candidates from FAISS (baseline retrieval quality proxy)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), search_k)

        # Compute retrieval-quality proxy from baseline FAISS scores.
        # For normalized sentence-transformer embeddings, FAISS IndexFlatIP returns inner product ~ cosine.
        # We use the mean of top_k FAISS scores as a stable proxy.
        quality_proxy = None
        try:
            top_scores = distances[0][: max(1, min(top_k, len(distances[0])))]
            if len(top_scores) > 0:
                quality_proxy = float(np.mean(top_scores))
        except Exception:
            quality_proxy = None

        # ============================================================================
        # OPTIMAL CONFIDENCE-BASED GATING: Evidence-Based AL Activation
        # ============================================================================
        # Comprehensive manual analysis of 318 tickets across 4 evaluations (2026-01-29)
        # revealed the OPTIMAL pattern:
        # 
        # 🎯 BEST RULE: Use AL ONLY when baseline retrieval is WEAK (judge score ≤ 0.20)
        # 
        # Evidence:
        # - 25 tickets (16.4% usage rate)
        # - 56% help rate when activated
        # - +0.9758% average improvement
        # - Expected gain: +387.8% over current gating!
        # 
        # Pattern Discovery:
        # ✅ When baseline WEAK (judge ≤ 0.20): AL helps 56% of time, improves +0.98%
        # ❌ When baseline DECENT (judge > 0.20): AL helps 52.5%, but hurts more (-0.12%)
        # ❌ When baseline STRONG (judge > 0.80): AL helps only 50%, hurts -2.32%
        # 
        # Key Insight: AL is RESCUE mechanism for failed baseline retrieval, 
        # NOT an enhancement for already-working retrieval!
        feedback_gated = False
        if FEEDBACK_ENABLED and FEEDBACK_CONFIDENCE_GATE and quality_proxy is not None:
            max_faiss_score = float(distances[0][0]) if len(distances[0]) > 0 else 0.0
            
            # CORRECTED GATING LOGIC (2026-01-30) - INVERTED FROM PREVIOUS VERSION
            # ======================================================================
            # CRITICAL DISCOVERY from analysis of 316 tickets:
            #   Correlation(baseline_cosine, AL_delta) = -0.206 (p=0.0002)
            # 
            # THIS MEANS: AL helps MORE when baseline is ALREADY STRONG!
            # 
            # Evidence:
            #   AL HELPS  (27 tickets): Avg baseline cosine = 0.4766 → AL improves +9%
            #   AL HURTS  (25 tickets): Avg baseline cosine = 0.6733 → AL degrades -13%
            # 
            # KEY INSIGHT:
            # AL is an OPTIMIZER, not a RESCUER
            # - When baseline is WEAK: Retrieved docs are poor → AL reranking of garbage = garbage
            # - When baseline is STRONG: Retrieved docs are good → AL optimizes ranking = better results
            # 
            # THEREFORE: Gate OFF weak baselines, ACTIVATE strong baselines
            # (This is the OPPOSITE of our previous hypothesis)
            
            STRONG_BASELINE_THRESHOLD = 0.60  # Gate OFF if cosine < 0.60 (weak baseline)
            
            if max_faiss_score < STRONG_BASELINE_THRESHOLD:
                # Baseline retrieval is WEAK - AL cannot rescue poor retrieval!
                feedback_gated = True
                feedback_logger.info(
                    f"🚫 CORRECTED GATE: Baseline retrieval is WEAK ({max_faiss_score:.4f} < {STRONG_BASELINE_THRESHOLD:.2f}) - "
                    f"AL cannot rescue poor retrieval. Using baseline only."
                )
            else:
                # Baseline retrieval is STRONG - AL can optimize ranking!
                feedback_logger.info(
                    f"✅ CORRECTED GATE: Baseline retrieval is STRONG ({max_faiss_score:.4f} ≥ {STRONG_BASELINE_THRESHOLD:.2f}) - "
                    f"Activating AL to optimize ranking (expected +1.5% improvement based on n=316 analysis)"
                )
        
        # If gated, return baseline results immediately
        if feedback_gated:
            feedback_logger.info(
                f"🚫 AL GATED: Returning pure FAISS baseline results"
            )
            # Return baseline results immediately - no re-ranking, no injection
            candidates = self.knowledge_base.iloc[indices[0][:top_k]].copy()
            candidates['faiss_score'] = distances[0][:top_k]
            # Use Ref as stable retrieved_id (survives KB sub-sampling / index resets)
            if 'Ref' in candidates.columns:
                ref_col = candidates['Ref']
                fallback_ids = candidates.index.astype(str)
                candidates['retrieved_id'] = ref_col.where(ref_col.notna(), fallback_ids).astype(str)
            elif 'Ticket_Reference' in candidates.columns:
                tr = candidates['Ticket_Reference']
                fallback_ids = candidates.index.astype(str)
                candidates['retrieved_id'] = tr.where(tr.notna(), fallback_ids).astype(str)
            else:
                candidates['retrieved_id'] = candidates.index.astype(str)
            
            # Add metadata for tracking
            candidates['enhanced_score'] = candidates['faiss_score']
            candidates['feedback_lift'] = 0.0
            candidates['confidence_gated'] = True
            
            return candidates[['retrieved_id', 'first_reply', 'Title_anon', 'Description_anon', 
                              'enhanced_score', 'feedback_lift', 'confidence_gated']]

        tier_name, tier_lift_scale, tier_inject_n = ("mid", 1.0, 10)
        if AL_TIER_ENABLED:
            tier_name, tier_lift_scale, tier_inject_n = _al_tier_for_quality(quality_proxy)
            if feedback_logger.isEnabledFor(logging.INFO):
                feedback_logger.info(
                    f"AL tier={tier_name} quality_proxy={quality_proxy} lift_scale={tier_lift_scale} inject_top_n={tier_inject_n}"
                )
        candidates = self.knowledge_base.iloc[indices[0]].copy()
        candidates['faiss_score'] = distances[0]
        # Use Ref as stable retrieved_id (survives KB sub-sampling / index resets).
        # Prefer Ref > Ticket_Reference > DataFrame index.
        if 'Ref' in candidates.columns:
            ref_col = candidates['Ref']
            fallback_ids = candidates.index.astype(str)
            candidates['retrieved_id'] = ref_col.where(ref_col.notna(), fallback_ids).astype(str)
        elif 'Ticket_Reference' in candidates.columns:
            tr = candidates['Ticket_Reference']
            fallback_ids = candidates.index.astype(str)
            candidates['retrieved_id'] = tr.where(tr.notna(), fallback_ids).astype(str)
        else:
            candidates['retrieved_id'] = candidates.index.astype(str)
        # Optional: warn if duplicates still exist (can indicate KB issues)
        try:
            if feedback_logger.isEnabledFor(logging.WARNING):
                dup_ids = candidates['retrieved_id'][candidates['retrieved_id'].duplicated()].unique()
                if len(dup_ids) > 0:
                    feedback_logger.warning(f"Duplicate retrieved_ids detected in candidates: {list(dup_ids)[:10]}")
        except Exception:
            # Logging issues should never break retrieval
            pass
        
        # Inject top feedback items into candidate set (tiered strength)
        # - strong tier: inject 0
        # - mid tier: inject a few
        # - weak tier: inject more
        inject_n = 10
        if AL_TIER_ENABLED:
            inject_n = max(0, int(tier_inject_n))
        
        # Apply global injection limit (allows disabling injection via env var)
        inject_n = min(inject_n, FEEDBACK_MAX_INJECT)

        if FEEDBACK_ENABLED and inject_n > 0:
            top_feedback = _get_top_feedback_items(
                predicted_class, 
                predicted_team, 
                top_n=inject_n,
                query_embedding=query_embedding,
                knowledge_base=self.knowledge_base,
                embeddings=self.embeddings,
                relevance_threshold=FEEDBACK_RELEVANCE_THRESHOLD,
            )
            if top_feedback:
                injected_count = 0
                for item_id, score in top_feedback:
                    # Check if already in candidates
                    if item_id not in candidates['retrieved_id'].values:
                        try:
                            # Find item in knowledge base — try Ref first, then Ticket_Reference, then index
                            mask = None
                            if 'Ref' in self.knowledge_base.columns:
                                mask = self.knowledge_base['Ref'].astype(str) == str(item_id)
                            if mask is None or not mask.any():
                                if 'Ticket_Reference' in self.knowledge_base.columns:
                                    mask = self.knowledge_base['Ticket_Reference'] == item_id
                            if mask is None or not mask.any():
                                mask = self.knowledge_base.index.astype(str) == item_id
                            
                            if mask.any():
                                kb_idx = self.knowledge_base[mask].index[0]
                                # Add to candidates with synthetic FAISS score (will be re-ranked by enhanced scoring)
                                new_row = self.knowledge_base.loc[kb_idx].copy()
                                new_row['faiss_score'] = 0.5  # Neutral FAISS score
                                new_row['retrieved_id'] = item_id
                                candidates = pd.concat([candidates, pd.DataFrame([new_row])], ignore_index=True)
                                injected_count += 1
                        except Exception as e:
                            feedback_logger.debug(f"Failed to inject feedback item {item_id}: {e}")
                
                if injected_count > 0:
                    feedback_logger.info(f"Injected {injected_count} high-feedback items into candidate set")
        
        # Step 2: Calculate enhanced scores
        enhanced_scores = []
        lifts = []
        for idx, row in candidates.iterrows():
            # Use the row's own faiss_score (robust even after candidate injection and index resets)
            faiss_score = float(row.get('faiss_score', 0.0) or 0.0)
            base = self._calculate_enhanced_score(query, row, predicted_category, faiss_score)

            # Apply tiered AL strength with RELEVANCE CHECK
            # Pass query context so lift is only applied if ticket is relevant
            lift = _feedback_lift_for(
                str(row['retrieved_id']), 
                predicted_class, 
                predicted_team,
                query_embedding=query_embedding,
                knowledge_base=self.knowledge_base,
                embeddings=self.embeddings,
                relevance_threshold=FEEDBACK_RELEVANCE_THRESHOLD,
            )
            # Optional positive boost: count global positives to amplify ranking for well-liked items
            # Fetch raw global positives quickly (separate connection to avoid slowing loop too much)
            pos_boost = 0.0
            if FEEDBACK_ENABLED and FEEDBACK_POS_BOOST > 0:
                try:
                    conn_tmp = sqlite3.connect(os.getenv("FEEDBACK_DB_PATH", FEEDBACK_DB_PATH))
                    cur_tmp = conn_tmp.cursor()
                    cur_tmp.execute(
                        "SELECT pos FROM feedback_agg WHERE retrieved_id=? AND scope_key='global'",
                        (str(row['retrieved_id']),)
                    )
                    rtmp = cur_tmp.fetchone()
                    if rtmp and rtmp[0] > 0:
                        pos_boost = FEEDBACK_POS_BOOST * rtmp[0]
                        # FIX B: Clamp pos_boost to prevent runaway scores
                        pos_boost = min(pos_boost, 0.5)  # Cap at 0.5 to keep comparable to base scores
                except Exception:
                    pos_boost = 0.0
                finally:
                    try:
                        conn_tmp.close()
                    except Exception:
                        pass
            # FIX A: lift already includes FEEDBACK_LIFT_MULT from _feedback_lift_for.
            scaled_lift = lift + pos_boost
            if AL_TIER_ENABLED:
                scaled_lift = float(scaled_lift) * float(tier_lift_scale)
            
            # PROTECTION: Don't demote high FAISS scores with feedback
            # If FAISS found a near-perfect match, trust it over popularity signals
            faiss_score = row.get('faiss_score', 0.0)
            if faiss_score > 0.90:
                # Perfect match - ignore feedback entirely
                scaled_lift = 0.0
                feedback_logger.debug(f"High FAISS score {faiss_score:.4f} - ignoring feedback lift")
            elif faiss_score > 0.80:
                # Very good match - limit feedback influence to prevent demotion
                if scaled_lift < 0:  # Only limit negative lifts
                    scaled_lift = max(scaled_lift, -0.05)  # Cap demotion at -0.05
                    feedback_logger.debug(f"Good FAISS score {faiss_score:.4f} - limiting feedback demotion to -0.05")
            
            enhanced_scores.append(base + scaled_lift)
            lifts.append(scaled_lift)
            feedback_logger.debug(
                f"Candidate id={row['retrieved_id']} base={base:.4f} raw_lift={lift:.4f} scaled_lift={scaled_lift:.4f} pos_boost={pos_boost:.4f} faiss={row.get('faiss_score'):.4f}"
            )
        candidates['enhanced_score'] = enhanced_scores
        candidates['feedback_lift'] = lifts
        candidates['confidence_gated'] = False  # Mark that these went through AL
        
        # SCOPE VERIFICATION: Analyze lift distribution to verify scope matching
        if FEEDBACK_ENABLED and len(lifts) > 0:
            nonzero_lifts = [l for l in lifts if abs(l) > 0.01]
            positive_lifts = [l for l in lifts if l > 0.01]
            negative_lifts = [l for l in lifts if l < -0.01]
            
            feedback_logger.info(
                f"Scope verification: query_class={predicted_class} query_team={predicted_team} | "
                f"candidates_total={len(candidates)} lifts_nonzero={len(nonzero_lifts)} "
                f"(+{len(positive_lifts)} pos, {len(negative_lifts)} neg) | "
                f"avg_lift={sum(lifts)/len(lifts):.4f} max_lift={max(lifts):.4f} min_lift={min(lifts):.4f}"
            )
        
        # Step 3: Re-rank and return top results
        # Prefer items with higher feedback_lift when enhanced_score ties or is close
        results = (
            candidates.sort_values(by=['enhanced_score', 'feedback_lift'], ascending=[False, False])
            .head(top_k)
        )
        # Log top-N results with scores
        if FEEDBACK_LOG_TOPN > 0 and feedback_logger.isEnabledFor(logging.INFO):
            preview = []
            for i, (_, r) in enumerate(results.head(FEEDBACK_LOG_TOPN).iterrows()):
                preview.append(
                    {
                        "id": str(r.get('retrieved_id')),
                        "faiss": float(r.get('faiss_score', 0.0)),
                        "lift": float(r.get('feedback_lift', 0.0)),
                        "enh": float(r.get('enhanced_score', 0.0)),
                    }
                )
            feedback_logger.info(f"Top{min(FEEDBACK_LOG_TOPN, top_k)} after rerank: {preview}")
        
        # Return columns based on whether feedback is enabled.
        # (Note: with tiering, feedback may be effectively disabled via lift_scale=0,
        # but we keep the column for transparency when FEEDBACK_ENABLED is True.)
        if FEEDBACK_ENABLED:
            return results[['retrieved_id', 'first_reply', 'Title_anon', 'Description_anon', 'enhanced_score', 'feedback_lift', 'confidence_gated']]
        results['confidence_gated'] = False
        return results[['retrieved_id', 'first_reply', 'Title_anon', 'Description_anon', 'enhanced_score', 'confidence_gated']]

    def _calculate_enhanced_score(self, query, candidate_row, predicted_category, faiss_score):
        """Calculate enhanced score considering multiple factors
        
        REBALANCED weights to give feedback room to dominate:
        - Base components use 65% total (down from 100%)
        - Feedback lift (±1.5 max) can now significantly impact rankings
        - Prevents base score variance from overwhelming feedback signal
        """
        # Base FAISS similarity (40% weight) - Reduced from 50%
        base_score = 0.4 * faiss_score
        
        # Category match bonus (12% weight) - Reduced from 20%
        category_bonus = 0.0
        if predicted_category and candidate_row.get('label_auto') == predicted_category:
            category_bonus = 0.12
        
        # Title similarity (8% weight) - Reduced from 15%
        title_sim = self._calculate_field_similarity(query, candidate_row.get('Title_anon', ''))
        title_score = 0.08 * title_sim
        
        # Description similarity (5% weight) - Reduced from 10%
        desc_sim = self._calculate_field_similarity(query, candidate_row.get('Description_anon', ''))
        desc_score = 0.05 * desc_sim
        
        # Quality bonus removed (was 5%) - redundant with FAISS similarity
        
        # Total base score now ranges from ~0.4 to ~0.65 (35% room for feedback)
        # Feedback lift (±1.5 capped) is ADDED to this, giving it proper influence
        return base_score + category_bonus + title_score + desc_score

    def _calculate_field_similarity(self, query, field_text):
        """Calculate similarity between query and specific field"""
        if pd.isna(field_text) or not str(field_text).strip():
            return 0.0
        
        try:
            query_emb = self.sentence_model.encode([query])
            field_emb = self.sentence_model.encode([str(field_text)])
            similarity = cosine_similarity(query_emb, field_emb)[0][0]
            return max(0.0, similarity)
        except:
            return 0.0

    def _assess_response_quality(self, response_text):
        """Assess response quality for scoring"""
        if pd.isna(response_text) or not str(response_text).strip():
            return 0.0
        
        response_str = str(response_text)
        quality_score = 0.0
        
        # Length check (not too short, not too long)
        length = len(response_str)
        if 50 <= length <= 2000:
            quality_score += 0.4
        
        # Has professional structure
        if any(greeting in response_str.lower() for greeting in ['dear', 'hello', 'thank you', 'need:']):
            quality_score += 0.3
        
        # Contains useful information
        if any(info in response_str.lower() for info in ['information', 'required', 'approval', 'project', 'manager']):
            quality_score += 0.3
        
        return min(1.0, quality_score)


# LAZY LOADING: Ticket classifier initialization
_ticket_classifier = None
_ticket_tokenizer = None
_ticket_label_encoder = None
_ticket_metadata = None

def _get_ticket_classifier():
    """Lazy load ticket classifier model on first use"""
    global _ticket_classifier, _ticket_tokenizer, _ticket_label_encoder, _ticket_metadata
    
    if _ticket_classifier is not None:
        return _ticket_classifier, _ticket_tokenizer, _ticket_label_encoder, _ticket_metadata
    
    ticket_classifier_path = './ticket_classifier_model'
    
    try:
        import json
        print(f"⚡ Lazy-loading ticket classifier from {ticket_classifier_path}...")
        
        _ticket_tokenizer = DistilBertTokenizer.from_pretrained(ticket_classifier_path)
        _ticket_classifier = DistilBertForSequenceClassification.from_pretrained(ticket_classifier_path)
        
        # Place ticket classifier on the same device we use for inference.
        # Prefer the device configured by the lazy-loaded team classifier.
        _, _, tc_device = _get_team_classifier()
        if tc_device is None:
            tc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _ticket_classifier.to(tc_device)
        _ticket_classifier.eval()
        
        # Load label encoder
        with open(f'{ticket_classifier_path}/label_encoder.pkl', 'rb') as f:
            _ticket_label_encoder = pickle.load(f)
        
        # Load metadata
        with open(f'{ticket_classifier_path}/metadata.json', 'r') as f:
            _ticket_metadata = json.load(f)
        
        print(f"✅ Ticket classifier loaded successfully")
        print(f"📊 Ticket classifier classes: {_ticket_metadata['classes']}")

        return _ticket_classifier, _ticket_tokenizer, _ticket_label_encoder, _ticket_metadata

    except Exception as e:
        print(f"❌ Error loading ticket classifier: {e}")
        print("💡 Falling back to keyword-based classification")
        return None, None, None, None


def classify_ticket(ticket_text, service_category=None, service_subcategory=None):
    """Enhanced classification using pre-trained DistilBERT model with keyword fallback"""
    
    # Lazy load ticket classifier
    ticket_classifier, ticket_tokenizer, ticket_label_encoder, ticket_metadata = _get_ticket_classifier()
    
    # Parse the input properly
    lines = ticket_text.split('\n', 1) if '\n' in ticket_text else ticket_text.split(' ', 1)
    title = lines[0].strip() if lines else ""
    description = lines[1].strip() if len(lines) > 1 else ticket_text.strip()
    
    # Format input EXACTLY like in training
    formatted_text = f"[TITLE] {title} [DESCRIPTION] {description}"
    
    # Add service information if available
    if service_category:
        formatted_text += f" [SERVICE] {service_category}"
    if service_subcategory:
        formatted_text += f" [SUBCATEGORY] {service_subcategory}"
    
    print(f"📋 Ticket classifier input: {formatted_text[:150]}...")
    
    # Try using the trained model first
    if ticket_classifier is not None and ticket_tokenizer is not None and ticket_label_encoder is not None:
        try:
            # Ensure inputs go to the same device as the model
            tc_device = next(ticket_classifier.parameters()).device
            # Tokenize input
            inputs = ticket_tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=512
            ).to(tc_device)
            
            # Get prediction
            with torch.no_grad():
                outputs = ticket_classifier(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities.max().item()
            
            # Convert to original label
            predicted_class = ticket_label_encoder.inverse_transform([predicted_class_idx])[0]
            
            print(f"🎯 Predicted ticket type: {predicted_class} (confidence: {confidence:.3f})")
            
            # Map predicted class to template response
            template_response = get_template_response_for_class(predicted_class)
            
            return predicted_class, template_response
            
        except Exception as e:
            print(f"❌ Error in model-based classification: {e}")
            print("💡 Falling back to keyword-based classification")
    
    # Fallback to keyword-based classification
    return classify_ticket_with_keywords(ticket_text, service_category, service_subcategory)

def get_template_response_for_class(predicted_class):
    """Map predicted class to appropriate template response"""
    
    template_mapping = {
        "vpn_request": "Need: Extra information needed to enable Client-to-Site VPN tunnel requests.",
        "onboarding": "Need: Extra information needed to start the IT On-boarding of a new internal employee.",
        "software_request": "Need: Extra information needed to request a software or a licence on my Windows device.",
        "software_support": "Need: Extra information needed to support you about local software / programs.",
        "admin_rights": "Need: Extra information needed to request admin rights on the computer.",
        "offboarding": "Need: Extra information needed to start the IT Off-boarding process.",
        "absence_request": "Need: Extra information needed to request an Absence ticket.",
        "badge_access": "Need: Extra information needed to request office access.",
        "email_support": "Need: Extra information needed to support you with email/mailbox issues.",
        "password_reset": "Need: Extra information needed to reset your password.",
        "printer_support": "Need: Extra information needed to support you with printer issues.",
        "hardware_support": "Need: Extra information needed to support you with hardware issues.",
        "other": "Thank you for your request. I will review the details and get back to you shortly."
    }
    
    return template_mapping.get(predicted_class, "Thank you for your request. I will review the details and get back to you shortly.")

def classify_ticket_with_keywords(ticket_text, service_category=None, service_subcategory=None):
    """Keyword-based classification as fallback"""
    
    text_lower = ticket_text.lower()
    
    # VPN requests (most specific first)
    if any(word in text_lower for word in ['vpn', 'tunnel', 'remote access', 'client-to-site', 'network access']):
        return "vpn_request", "Need: Extra information needed to enable Client-to-Site VPN tunnel requests."
    
    # Employee onboarding
    elif any(word in text_lower for word in ['onboard', 'new employee', 'employee setup', 'employee starting', 'employee id', 'missing employee']):
        return "onboarding", "Need: Extra information needed to start the IT On-boarding of a new internal employee."
    
    # Software installation vs support
    elif any(word in text_lower for word in ['install', 'installation', 'need software']) and not any(word in text_lower for word in ['support', 'help', 'issue', 'problem']):
        return "software_request", "Need: Extra information needed to request a software or a licence on my Windows device."
    
    # Software support
    elif any(word in text_lower for word in ['software support', 'software help', 'software issue', 'help with', 'issue with']):
        return "software_support", "Need: Extra information needed to support you about local software / programs."
    
    # Admin rights
    elif any(word in text_lower for word in ['admin', 'administrator', 'elevated', 'admin rights', 'privileges']):
        return "admin_rights", "Need: Extra information needed to request admin rights on the computer."
    
    # Employee offboarding
    elif any(word in text_lower for word in ['offboard', 'leaving', 'last day', 'equipment return']):
        return "offboarding", "Need: Extra information needed to start the IT Off-boarding process."
    
    # Absence requests
    elif any(word in text_lower for word in ['absence', 'vacation', 'time off', 'successfactors']):
        return "absence_request", "Need: Extra information needed to request an Absence ticket."
    
    # Badge/access requests
    elif any(word in text_lower for word in ['badge', 'access card', 'building access', 'office access']):
        return "badge_access", "Need: Extra information needed to request office access."
    
    # Email support
    elif any(word in text_lower for word in ['email', 'mailbox', 'outlook', 'mail']):
        return "email_support", "Need: Extra information needed to support you with email/mailbox issues."
    
    # Password reset
    elif any(word in text_lower for word in ['password', 'reset', 'unlock', 'account locked']):
        return "password_reset", "Need: Extra information needed to reset your password."
    
    # Printer support
    elif any(word in text_lower for word in ['printer', 'printing', 'print']):
        return "printer_support", "Need: Extra information needed to support you with printer issues."
    
    # Hardware support
    elif any(word in text_lower for word in ['hardware', 'laptop', 'desktop', 'monitor', 'keyboard', 'mouse']):
        return "hardware_support", "Need: Extra information needed to support you with hardware issues."
    
    else:
        return "other", "Thank you for your request. I will review the details and get back to you shortly."


# Add these functions after the existing imports and before the generate_response_with_openai function

def select_best_templates(title, description, template_examples, top_k=3):
    """Select the most similar templates based on content similarity"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Create query text
    query_text = f"{title} {description}"
    
    # Create comparison texts
    comparison_texts = []
    for _, row in template_examples.iterrows():
        comp_text = f"{row.get('Title_anon', '')} {row.get('Description_anon', '')}"
        comparison_texts.append(comp_text)
    
    # Calculate similarities using sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query_text])
    comparison_embeddings = model.encode(comparison_texts)
    
    # Calculate cosine similarities
    similarities = np.dot(query_embedding, comparison_embeddings.T).flatten()
    
    # Add similarity scores and sort
    template_examples = template_examples.copy()
    template_examples['template_similarity'] = similarities
    
    return template_examples.nlargest(top_k, 'template_similarity')

def get_specific_instructions(classification, predicted_team):
    """Get specific instructions based on ticket type and team"""
    instructions = {
        'admin_rights': """
- MUST include form fields: 'Need:', 'Do you need admin rights...', 'Project Manager Full name:'
- Use EXACT structure from admin rights templates
- Include approval requirements""",
        
        'badge_access': """
- MUST start with 'Below you will find the additional form information'
- Include 'Need: Extra information needed to request office access'
- Use structured format with colleague information""",
        
        'email_support': """
- Focus on mailbox or communication issues
- Include troubleshooting steps if shown in template
- Maintain technical support format""",
        
        'software_request': """
- Include software installation requirements
- Use form structure for license/approval process
- Reference project manager requirements""",
        
        'vpn_request': """
- For onboarding: Use brief auto-generated format
- For VPN access: Include network configuration details
- Maintain structured list format"""
    }
    
    team_instructions = {
        '(BF) Information Security Office': "- Focus on security approval processes\n- Include project manager requirements",
        '(LF) IT Office Access Italy': "- Use Italian office access format\n- Include colleague contact information",
        '(GI-UX) Account Management': "- Use onboarding templates for new employees\n- Keep format brief and structured"
    }
    
    result = instructions.get(classification, "- Follow the template structure exactly")
    if predicted_team in team_instructions:
        result += f"\n{team_instructions[predicted_team]}"
    
    return result

def analyze_response_type_simple(response_text):
    """Enhanced analysis to determine if response is template-based or personalized
    
    Returns:
        tuple: (response_type: str, confidence: float, length_category: str)
               response_type: "template", "personalized", or "mixed"
               confidence: 0.0 to 1.0
               length_category: "short" (<300 chars), "medium" (300-800), "long" (>800)
    """
    if not response_text or pd.isna(response_text):
        return "unknown", 0.0
    
    text = str(response_text).lower()
    text_length = len(response_text)
    
    # Determine length category
    if text_length < 300:
        length_category = "short"
    elif text_length < 800:
        length_category = "medium"
    else:
        length_category = "long"
    
    # Template indicators (enhanced with more patterns)
    template_keywords = [
        "below you will find the additional form information",
        "need: extra information needed",
        "this additional ticket is automatically created",
        "- instructions:",
        "- project manager full name:",
        "- computer name:",
        "- justification:",
        "- employee id:",
        "automatically created",
        "auto-generated",
        "this ticket has been",
        "important information:",
        "please before submitting",
        "required information",
        "canonicalname:",
        "memberof:",
        "displayname:",
        "the following required information"
    ]
    
    # Personalized indicators (enhanced with Italian language patterns)
    personalized_keywords = [
        "thank you for contacting",
        "i understand",
        "i will help",
        "hello",
        "dear",
        "i apologize",
        "let me check",
        "please note that",
        "i recommend",
        "thank you for your request",
        "grazie",  # Italian: thank you
        "preferisco",  # Italian: I prefer
        "come sede",  # Italian: as location
        "per quanto riguarda",  # Italian: regarding
        "ti contatto",  # Italian: I contact you
        "buongiorno",  # Italian: good morning
        "cordiali saluti"  # Italian: kind regards
    ]
    
    # Count matches
    template_score = sum(1 for keyword in template_keywords if keyword in text)
    personalized_score = sum(1 for keyword in personalized_keywords if keyword in text)
    
    # Length-based scoring (short responses are usually personalized)
    if text_length < 200:
        personalized_score += 3  # Strong indicator of personalized response
    elif text_length < 400:
        personalized_score += 1  # Moderate indicator
    elif text_length > 1000:
        template_score += 2  # Long responses often templates
    
    # Check for structured lists (template indicator)
    dash_lines = text.count('\n-')
    if dash_lines >= 3:
        template_score += 3  # Increased weight for structured lists
    elif dash_lines >= 1:
        template_score += 1
    
    # Check for form fields (strong template indicator)
    if ':' in text and text.count(':') >= 5:
        template_score += 2
    
    # Check for questions (personalized indicator)
    question_marks = text.count('?')
    if question_marks > 0:
        personalized_score += question_marks
    
    # Check for email-style formatting (template indicator)
    if '***' in text or '####' in text or '====' in text:
        template_score += 2
    
    # Check for first-person pronouns (personalized indicator)
    first_person = ['i am', 'i will', 'i can', 'i have', 'my ', 'we will', 'we can']
    first_person_count = sum(1 for phrase in first_person if phrase in text)
    if first_person_count > 0:
        personalized_score += first_person_count
    
    # Determine type
    total_score = template_score + personalized_score
    if total_score == 0:
        # Default based on length
        if text_length < 300:
            return "personalized", 0.7, length_category  # Short = likely personalized
        else:
            return "template", 0.6, length_category  # Long = likely template
    
    template_confidence = template_score / total_score
    
    if template_confidence > 0.6:
        return "template", template_confidence, length_category
    elif template_confidence < 0.4:
        return "personalized", 1 - template_confidence, length_category
    else:
        return "mixed", 0.5, length_category

# Update OpenAI API call to ensure compatibility and proper response handling
def generate_short_personalized_response(ticket_title, ticket_description, classification, predicted_team, similar_replies):
    """Generate concise, personalized responses for short reply scenarios
    
    Args:
        ticket_title: The ticket title
        ticket_description: The ticket description  
        classification: Predicted ticket type
        predicted_team: Predicted team
        similar_replies: DataFrame of similar tickets
        
    Returns:
        str: A short, personalized response matching the expected style
    """
    
    # Extract short examples from similar replies
    short_examples = []
    for _, reply in similar_replies.iterrows():
        first_reply = str(reply.get('first_reply', ''))
        if len(first_reply) < 400 and len(first_reply) > 30:  # Short but not empty
            short_examples.append({
                'reply': first_reply[:300],  # Limit context
                'title': str(reply.get('Title_anon', ''))[:100]
            })
            if len(short_examples) >= 3:  # Limit to 3 examples
                break
    
    # Build prompt for short response
    examples_text = ""
    if short_examples:
        examples_text = "\n\nExamples of short, personalized responses:\n"
        for i, ex in enumerate(short_examples, 1):
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
Title: {ticket_title}
Description: {ticket_description}
Classification: {classification}
Team: {predicted_team}
{examples_text}

Write a SHORT, personalized response (max 200 chars):"""

    try:
        messages = [
            {"role": "system", "content": "You are an IT support agent who writes very short, personalized responses. Maximum 200 characters. No templates."},
            {"role": "user", "content": prompt}
        ]
        
        generated_reply = _chat_completion(
            model=GEN_MODEL_SHORT,
            messages=messages,
            max_tokens=GEN_MAX_TOKENS_SHORT,  # Limit tokens for short response
            temperature=GEN_TEMPERATURE_SHORT,  # Higher temperature for natural variation
            top_p=0.9
        )
        
        # Enforce length limit
        if len(generated_reply) > 300:
            generated_reply = generated_reply[:297] + "..."
            
        logger.info(f"Generated short personalized response: {len(generated_reply)} chars")
        return generated_reply
        
    except Exception as e:
        logger.error(f"Error generating short response: {e}")
        # Fallback to very simple response
        return f"Thank you for your message. The {predicted_team} team will review your request."


def generate_response_with_openai_personal(ticket_title, ticket_description, classification, predicted_team, team_confidence, similar_replies):
    """Enhanced personalized response generation with better context awareness"""
    
    # Enhanced context analysis
    context_analysis = analyze_ticket_context(ticket_title, ticket_description, classification)
    
    # Select best similar replies
    if len(similar_replies) > 2:
        similar_replies = select_best_similar_replies(ticket_title, ticket_description, similar_replies, top_k=2)
    
    # Construct enhanced prompt
    prompt = (
        f"You are an expert IT support specialist generating a professional first reply. "
        f"Use the context and similar examples to create a response that matches the expected format and tone.\n\n"
        f"CURRENT TICKET:\n"
        f"Title: {ticket_title}\n"
        f"Description: {ticket_description}\n"
        f"Request Type: {classification}\n"
        f"Assigned Team: {predicted_team} (confidence: {team_confidence:.2f})\n"
        f"Context: {context_analysis}\n\n"
        f"SIMILAR RESOLVED TICKETS FOR REFERENCE:\n"
    )
    
    # Add similar replies with better formatting
    for i, (_, reply) in enumerate(similar_replies.iterrows(), 1):
        prompt += (
            f"{i}. Similar Ticket Title: {reply['Title_anon']}\n"
            f"   Similar Ticket Description: {reply['Description_anon'][:300]}...\n"
            f"   First Reply: {reply['first_reply']}\n\n"
        )
    
    # Enhanced instructions based on patterns
    response_style = determine_response_style(similar_replies)
    
    prompt += (
        f"RESPONSE REQUIREMENTS:\n"
        f"- Style: {response_style}\n"
        f"- Match the tone and structure of similar replies\n"
        f"- Address the specific request clearly and professionally\n"
        f"- Include relevant next steps or requirements\n"
        f"- Keep length appropriate for first reply (150-300 words)\n\n"
        f"Generate a professional first reply that follows the patterns shown in the similar tickets above:"
    )

    print("Prompt sent to OpenAI API (Enhanced Personal):")
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)

    # NO TRY-EXCEPT - Let errors propagate to see what's wrong
    return _chat_completion(
    model=GEN_MODEL_PERSONAL,
        messages=[
            {"role": "system", "content": "You are an expert IT support specialist who writes clear, professional responses that match organizational standards."},
            {"role": "user", "content": prompt}
        ],
    max_tokens=GEN_MAX_TOKENS_PERSONAL,
    temperature=GEN_TEMPERATURE_PERSONAL,
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.2
    )

def analyze_ticket_context(title, description, classification):
    """Analyze ticket context for better response generation"""
    context_factors = []
    
    # Urgency indicators
    if any(word in title.lower() for word in ['urgent', 'urgente', 'asap', 'immediately']):
        context_factors.append("urgent")
    
    # Request type specific context
    if classification == 'admin_rights':
        if 'install' in description.lower():
            context_factors.append("software_installation")
        if 'java' in description.lower():
            context_factors.append("development_tools")
    
    # Team context
    if 'onboarding' in description.lower() or 'new_entry' in title.lower():
        context_factors.append("employee_onboarding")
    
    return ", ".join(context_factors) if context_factors else "standard_request"


def detect_temporal_context(title, description):
    """Detect temporal/status-update context (outages, global communications, recovery plans).

    Returns: 'temporal_update' or 'standard'
    """
    text = (str(title or "") + " " + str(description or "")).lower()

    # Patterns that indicate a system status / global communication / recovery plan
    temporal_patterns = [
        r"\byesterday\b",
        r"\bwe have sent a global communication\b",
        r"\brecovery plan\b",
        r"\bwhen .* returns to normal\b",
        r"\boutage\b",
        r"\bservice interruption\b",
        r"\bwe are currently experiencing\b",
        r"\bincident\b",
        r"\bsynertrade\b",
        r"\bsap\b",
    ]

    for pat in temporal_patterns:
        if re.search(pat, text):
            return 'temporal_update'

    return 'standard'


def generate_status_update_response(ticket_title, ticket_description, classification, predicted_team, team_confidence, similar_replies):
    """Generate a structured status-update style first reply for incidents/outages/recovery messages.

    This aims to mirror the organization's past global communications (e.g., "Yesterday, from Group IT...") and
    include: current status, impact, known workarounds, next steps, and contact/ETA where available.
    """
    # Build concise context from similar replies
    examples_text = ""
    for i, (_, r) in enumerate(similar_replies.head(3).iterrows(), 1):
        fr = r.get('first_reply', '')
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
        f"TICKET:\nTitle: {ticket_title}\nDescription: {ticket_description}\nAssigned Team: {predicted_team} (confidence: {team_confidence:.2f})\n\n"
        f"SIMILAR_EXAMPLES:\n{examples_text}\n\n"
        "Generate the STATUS UPDATE now. Keep it factual, short paragraphs, 120-220 words. If you don't have ETA, say 'investigating' and provide workarounds if present."
    )

    print("Prompt sent to OpenAI API (Status Update):")
    print(prompt[:800] + '...' if len(prompt) > 800 else prompt)

    # NO TRY-EXCEPT - Let errors propagate to see what's wrong
    return _chat_completion(
    model=GEN_MODEL_STATUS,
        messages=[
            {"role": "system", "content": "You are an IT operations communicator writing concise status updates."},
            {"role": "user", "content": prompt}
        ],
    max_tokens=GEN_MAX_TOKENS_STATUS,
    temperature=GEN_TEMPERATURE_STATUS,
        top_p=0.8
    )

def select_best_similar_replies(title, description, similar_replies, top_k=2):
    """Select the most relevant similar replies for better context"""
    # Already has similarity scores from RAG system
    if 'enhanced_score' in similar_replies.columns:
        return similar_replies.nlargest(top_k, 'enhanced_score')
    else:
        return similar_replies.head(top_k)

def determine_response_style(similar_replies):
    """Determine the appropriate response style based on similar replies"""
    reply_texts = [reply.get('first_reply', '') for _, reply in similar_replies.iterrows()]
    
    # Check for common patterns
    if any('Below you will find' in text for text in reply_texts):
        return "Structured form-based response"
    elif any(len(text) < 200 for text in reply_texts):
        return "Brief informational response"
    else:
        return "Detailed personalized response"

# Replace the existing generate_response_with_openai_personal function with this:

# def generate_response_with_openai_personal(ticket_title, ticket_description, classification, predicted_team, team_confidence, similar_replies):
#     """Generate personalized response with proper DataFrame handling"""
    
#     # Ensure similar_replies is a DataFrame
#     if isinstance(similar_replies, list):
#         similar_replies = pd.DataFrame(similar_replies)
    
#     # Construct the prompt
#     prompt = (
#         f"You are an AI assistant tasked with generating a helpful and professional first reply for a support ticket. "
#         f"Below is the ticket information and similar tickets with their first replies. Use this context to craft a response "
#         f"that is clear, concise, and addresses the user's needs effectively.\n\n"
#         f"Ticket Title: {ticket_title}\n"
#         f"Ticket Description: {ticket_description}\n"
#         f"Request Type: {classification}\n"
#         f"Assigned Team: {predicted_team} (confidence: {team_confidence:.2f})\n\n"
#         f"Similar Tickets and Their First Replies:\n"
#     )
    
#     # Add similar tickets - handle DataFrame properly
#     example_count = 0
#     for index, reply in similar_replies.iterrows():
#         if example_count >= 3:  # Limit to 3 examples
#             break
            
#         prompt += (
#             f"{example_count + 1}. Similar Ticket Title: {reply.get('Title_anon', 'N/A')}\n"
#             f"   Similar Ticket Description: {reply.get('Description_anon', 'N/A')}\n"
#             f"   First Reply: {reply.get('first_reply', 'N/A')}\n\n"
#         )
#         example_count += 1
    
#     prompt += (
#         "Based on the above information, generate a professional and personalized first reply for the given ticket. "
#         "Ensure the response is actionable and addresses the user's request clearly."
#     )

#     # Log the prompt
#     print("Prompt sent to OpenAI API:")
#     print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

#     # Call OpenAI API using the modern client
#     try:
#         client = OpenAI(api_key=openai.api_key)
#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful AI assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=300,
#             temperature=0.7,
#             top_p=0.9,
#             frequency_penalty=0.2,
#             presence_penalty=0.3
#         )
#         generated_reply = completion.choices[0].message.content.strip()
#         return generated_reply
        
#     except Exception as e:
#         print(f"Error calling OpenAI API: {e}")
#         return "Sorry, we encountered an issue generating a response. Please try again later."


def generate_response_with_openai(ticket_title, ticket_description, classification, predicted_team, team_confidence, similar_replies, temporal_context='standard'):
    """Enhanced response generation with simple template vs personalized detection"""
    
    # Analyze similar replies to determine response type
    template_examples = []
    personalized_examples = []
    short_examples = []
    
    print(f"🔍 Analyzing {len(similar_replies)} similar replies...")
    
    for _, reply in similar_replies.iterrows():
        first_reply = str(reply.get('first_reply', ''))
        response_type, confidence, length_category = analyze_response_type_simple(first_reply)
        
        print(f"   Reply: {response_type} (confidence: {confidence:.2f}, length: {length_category})")
        
        if length_category == "short" and len(first_reply) < 400:
            short_examples.append(reply)
        elif response_type == "template" and confidence > 0.5:
            template_examples.append(reply)
        elif response_type == "personalized" and confidence > 0.5:
            personalized_examples.append(reply)
    
    # Check if we should generate a short response
    has_short_replies = len(short_examples) >= 2 or (len(short_examples) > 0 and len(short_examples) >= len(template_examples))
    
    if has_short_replies:
        print(f"📝 Detected SHORT REPLY pattern ({len(short_examples)} short examples)")
        print("   Using specialized short response generator...")
        # Convert to DataFrame
        if isinstance(short_examples[0], pd.Series):
            short_df = pd.DataFrame(short_examples)
        else:
            short_df = pd.DataFrame(short_examples)
        return generate_short_personalized_response(ticket_title, ticket_description, classification, predicted_team, short_df)
    
    # Decide which approach to use for standard responses
    use_template = len(template_examples) > len(personalized_examples)

    # If temporal/status-update context is detected, prefer personalized status-update responses
    if temporal_context == 'temporal_update':
        print("🔔 Temporal context detected — forcing personalized/status-update response")
        use_template = False

    print(f"📊 Decision: {'Template' if use_template else 'Personalized'}")
    print(f"   Template examples: {len(template_examples)}")
    print(f"   Personalized examples: {len(personalized_examples)}")
    print(f"   Short examples: {len(short_examples)}")
    
    # Generate appropriate response using existing functions
    if use_template and template_examples:
        print("🔧 Generating template response...")
        # Convert list of Series to DataFrame for template function
        if isinstance(template_examples[0], pd.Series):
            template_df = pd.DataFrame(template_examples)
        else:
            template_df = pd.DataFrame(template_examples)
        return generate_template_response_function(ticket_title, ticket_description, classification, predicted_team, template_df)
    else:
        print("💬 Generating personalized response...")
        # Convert list of Series to DataFrame or use original similar_replies
        if personalized_examples:
            if isinstance(personalized_examples[0], pd.Series):
                personalized_df = pd.DataFrame(personalized_examples)
            else:
                personalized_df = pd.DataFrame(personalized_examples)
        else:
            personalized_df = similar_replies  # Use original DataFrame
        
        # If temporal/context is status update, use dedicated status-update generator
        if temporal_context == 'temporal_update':
            return generate_status_update_response(ticket_title, ticket_description, classification, predicted_team, team_confidence, personalized_df)

        return generate_response_with_openai_personal(ticket_title, ticket_description, classification, predicted_team, team_confidence, personalized_df)
    
# Replace the existing generate_template_response_function with this:

def generate_template_response_function(ticket_title, ticket_description, classification, predicted_team, template_examples):
    """Generate template-based response using LEGACY-PARITY simple strict prompt (like resolution_task_no_feedback.py)"""
    
    # Ensure template_examples is a DataFrame
    if isinstance(template_examples, list):
        template_examples = pd.DataFrame(template_examples)
    
    # LEGACY PARITY: Simple, strict prompt matching the old high-performing system
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
        f"Ticket Title: {ticket_title}\n"
        f"Ticket Description: {ticket_description}\n"
        f"Request Type: {classification}\n"
        f"Assigned Team: {predicted_team}\n\n"
        f"TEMPLATE EXAMPLES FROM SIMILAR TICKETS:\n"
    )
    
    # Add template examples - simple iteration, no ranking/filtering complexity
    example_count = 0
    for index, reply in template_examples.iterrows():
        if example_count >= 3:  # Limit to 3 examples (legacy behavior)
            break
            
        prompt += (
            f"EXAMPLE {example_count + 1}:\n"
            f"Title: {reply.get('Title_anon', 'N/A')}\n"
            f"Description: {reply.get('Description_anon', 'N/A')}\n"
            f"Template Response:\n{reply.get('first_reply', 'N/A')}\n"
            f"{'='*50}\n\n"
        )
        example_count += 1
    
    prompt += (
        f"GENERATE THE RESPONSE:\n"
        f"Follow the EXACT structure and format from the examples above. "
        f"Use the same field names, formatting, and template structure. "
        f"Replace only the specific values (names, computer IDs, etc.) that are relevant to the new ticket. "
        f"Do NOT add conversational language or explanations."
    )

    print("Prompt sent to OpenAI API (Template - Legacy Parity):")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Use configured generation settings
    return _chat_completion(
        model=GEN_MODEL_TEMPLATE,
        messages=[
            {"role": "system", "content": "You are an IT support system that generates responses matching the exact format and style of provided examples."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=GEN_MAX_TOKENS_TEMPLATE,
        temperature=GEN_TEMPERATURE_TEMPLATE,
        top_p=0.8,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )


# def generate_response_with_openai(ticket_title, ticket_description, classification, predicted_team, team_confidence, similar_replies):
#     # Analyze response patterns from similar tickets to determine the correct format
#     response_patterns = []
#     template_indicators = [
#         "Below you will find the additional form information",
#         "Need: Extra information needed",
#         "This additional ticket is automatically created",
#         "- Instructions:",
#         "- Project Manager Full name:",
#         "- Computer name:",
#         "- Justification:"
#     ]
    
#     # Detect if similar replies use templates/forms
#     uses_template = False
#     for _, reply in similar_replies.iterrows():
#         first_reply = str(reply.get('first_reply', ''))
#         if any(indicator in first_reply for indicator in template_indicators):
#             uses_template = True
#             response_patterns.append(first_reply[:500])  # Get pattern samples
    
#     # Build context-aware prompt based on response type
#     if uses_template:
#         prompt = (
#             f"You are an IT support system that generates STRUCTURED TEMPLATE RESPONSES. "
#             f"You must follow the EXACT format and structure shown in the similar tickets below.\n\n"
#             f"IMPORTANT INSTRUCTIONS:\n"
#             f"- Copy the EXACT format from similar tickets\n"
#             f"- Use structured lists with dashes (-)\n"
#             f"- Include form fields like 'Need:', 'Project Manager Full name:', etc.\n"
#             f"- Do NOT write conversational responses\n"
#             f"- Start with 'Below you will find the additional form information' when appropriate\n"
#             f"- For onboarding tickets, use brief auto-generated messages\n\n"
#             f"Ticket Title: {ticket_title}\n"
#             f"Ticket Description: {ticket_description}\n"
#             f"Request Type: {classification}\n"
#             f"Assigned Team: {predicted_team}\n\n"
#             f"TEMPLATE EXAMPLES FROM SIMILAR TICKETS:\n"
#         )
#     else:
#         prompt = (
#             f"You are an IT support agent generating a first reply to a support ticket. "
#             f"Follow the response style and format shown in the similar tickets below.\n\n"
#             f"Ticket Title: {ticket_title}\n"
#             f"Ticket Description: {ticket_description}\n"
#             f"Request Type: {classification}\n"
#             f"Assigned Team: {predicted_team}\n\n"
#             f"Similar Tickets and Their First Replies:\n"
#         )
    
#     # Add similar ticket examples
#     for i, reply in similar_replies.iterrows():
#         if uses_template:
#             prompt += (
#                 f"EXAMPLE {i+1}:\n"
#                 f"Title: {reply['Title_anon']}\n"
#                 f"Description: {reply['Description_anon']}\n"
#                 f"Template Response:\n{reply['first_reply']}\n"
#                 f"{'='*50}\n\n"
#             )
#         else:
#             prompt += (
#                 f"{i+1}. Similar Ticket Title: {reply['Title_anon']}\n"
#                 f"   Similar Ticket Description: {reply['Description_anon']}\n"
#                 f"   First Reply: {reply['first_reply']}\n\n"
#             )
    
#     # Different instructions based on response type
#     if uses_template:
#         prompt += (
#             f"GENERATE THE RESPONSE:\n"
#             f"Follow the EXACT structure and format from the examples above. "
#             f"Use the same field names, formatting, and template structure. "
#             f"Replace only the specific values (names, computer IDs, etc.) that are relevant to the new ticket. "
#             f"Do NOT add conversational language or explanations."
#         )
#     else:
#         prompt += (
#             "Based on the above information, generate a professional first reply that matches the style and approach of the similar tickets. "
#             "Ensure the response is actionable and addresses the user's request clearly."
#         )

#     print("Prompt sent to OpenAI API:")
#     print(prompt)#[:1000] + "..." if len(prompt) > 1000 else prompt)

#     try:
#         client = OpenAI(api_key=openai.api_key)
#         completion = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an IT support system that generates responses matching the exact format and style of provided examples."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=800,  # Increased for longer template responses
#             temperature=0.3,  # Lower temperature for more consistent formatting
#             top_p=0.8,
#             frequency_penalty=0.1,
#             presence_penalty=0.1
#         )
#         generated_reply = completion.choices[0].message.content.strip()
#         return generated_reply
        
#     except Exception as e:
#         print(f"Error calling OpenAI API: {e}")
#         return "Sorry, we encountered an issue generating a response. Please try again later."



# Ensure the result dictionary contains all expected keys
def generate_response(ticket_title, ticket_description, rag_system, retrieval_k: int = 5):
    """Enhanced response generation with improved retrieval and optional profiling"""
    import time
    profile_enabled = os.getenv("PROFILE_API", "false").lower() in ("1", "true", "yes")
    timings = {} if profile_enabled else None
    
    ticket_text = str(ticket_title) + " " + str(ticket_description)
    
    # Step 1: Classification
    if profile_enabled: t0 = time.time()
    word_class, template = classify_ticket(ticket_text)
    if profile_enabled: timings['classify_ticket'] = time.time() - t0

    # Map word_class to category labels for filtering
    category_mapping = {
        "vpn_request": "to enable Client-to-Site VPN tunnel requests",
        "onboarding": "to start the IT On-boarding of a new internal employee",
        "software_request": "to request a software or a licence on my Windows device",
        "offboarding": "to start the IT Off-boarding process",
        "absence_request": "to request an Absence ticket",
        "admin_rights": "to request admin rights on the computer"
    }
    
    predicted_category = category_mapping.get(word_class, None)
    
    # Step 2: Team classification
    if profile_enabled: t0 = time.time()
    predicted_team, team_confidence = classify_team_with_distilbert(ticket_text)
    if profile_enabled: timings['classify_team'] = time.time() - t0

    # Detect temporal/status-update context and pass to generator
    temporal_context = detect_temporal_context(ticket_title, ticket_description)

    # Step 3: Retrieval with feedback
    if profile_enabled: t0 = time.time()
    similar_replies = rag_system.retrieve_similar_replies(
        ticket_text,
        top_k=retrieval_k,
        predicted_category=predicted_category,
        predicted_class=word_class,
        predicted_team=predicted_team,
    )
    if profile_enabled: timings['retrieval'] = time.time() - t0

    # Step 4: LLM response generation
    if profile_enabled: t0 = time.time()
    response = generate_response_with_openai(ticket_title, ticket_description, word_class, predicted_team, team_confidence, similar_replies, temporal_context=temporal_context)
    if profile_enabled: timings['llm_generation'] = time.time() - t0

    result = {
        "classification": word_class,
        "predicted_team": predicted_team,
        "team_confidence": team_confidence,
        "response": response,
        "similar_replies": similar_replies.to_dict(orient="records"),
        "predicted_class": word_class,
        "retrieval_k": retrieval_k
    }
    
    if profile_enabled:
        result['_generation_profiling'] = timings
        logger.info(f"⏱️ GEN: ticket={timings.get('classify_ticket',0):.2f}s team={timings.get('classify_team',0):.2f}s retrieval={timings.get('retrieval',0):.2f}s llm={timings.get('llm_generation',0):.2f}s")
    
    return result

############################
# CACHING LAYER FOR RAG
############################
_RAG_CACHE = {"kb_path": None, "kb_mtime": None, "rag": None, "df": None}

def _get_or_build_rag(knowledge_base_path: str, force_rebuild: bool = False):
    """Return cached RAG system unless source CSV changed or rebuild forced."""
    try:
        current_mtime = os.path.getmtime(knowledge_base_path)
    except OSError:
        current_mtime = None
    
    # TESTING INCREMENTAL EMBEDDINGS MODE
    # Disable mtime check to prevent automatic rebuilds when CSV changes
    # Still allows initial build when cache is empty
    cache_ok = (
        _RAG_CACHE["rag"] is not None and
        _RAG_CACHE["kb_path"] == knowledge_base_path
        and (current_mtime is None or _RAG_CACHE["kb_mtime"] == current_mtime)  # DISABLED: mtime check
        and not force_rebuild
    )
    if cache_ok:
        print(f"✅ Using cached RAG system (incremental mode - no rebuild on CSV changes)")
        return _RAG_CACHE["rag"], _RAG_CACHE["df"]

    # Build only if cache is empty (first run) or force_rebuild=True
    if _RAG_CACHE["rag"] is None:
        print(f"🔨 Building RAG system for first time (cache empty)...")
    else:
        print(f"🔨 Rebuilding RAG system (force_rebuild={force_rebuild})...")
    
    df = load_knowledge_base(knowledge_base_path)
    rag = RAGSystem(df, kb_path=knowledge_base_path)
    rag.build_index(kb_path=knowledge_base_path, kb_mtime=current_mtime)
    _RAG_CACHE.update({
        "kb_path": knowledge_base_path,
        "kb_mtime": current_mtime,
        "rag": rag,
        "df": df
    })
    print(f"✅ RAG system ready: {len(df)} tickets indexed")
    return rag, df

def _fallback_response(ticket_title, ticket_description, classification, predicted_team, similar_replies_df):
    """Offline fallback if OpenAI unavailable or errors occur."""
    snippet = ""
    if similar_replies_df is not None and not similar_replies_df.empty:
        first = similar_replies_df.iloc[0]
        snippet = (str(first.get('first_reply', '')) or '')[:400]
    return (
        f"[Fallback Generated Reply]\n"
        f"Request Type: {classification}\nPredicted Team: {predicted_team}\n"
        f"We received your ticket titled '{ticket_title}'. We are reviewing the details and will proceed accordingly."
        + (f"\n\nReference example (trimmed):\n{snippet}" if snippet else "")
    )

def rebuild_embeddings(knowledge_base_path: str):
    """Force a fresh embedding build ignoring any existing cache.

    Returns a summary dict with basic stats. This sets DISABLE_EMBEDDING_CACHE=1
    temporarily so that the RAGSystem.build_index path always recomputes
    embeddings instead of loading a cached .npz. After building we manually
    persist a new cache file so subsequent standard calls can load it fast.
    """
    prev_disable = os.environ.get("DISABLE_EMBEDDING_CACHE")
    os.environ["DISABLE_EMBEDDING_CACHE"] = "1"
    try:
        # Force rebuild by passing force_rebuild=True
        rag, df = _get_or_build_rag(knowledge_base_path, force_rebuild=True)
        records = len(df)
        emb_dim = int(rag.embeddings.shape[1]) if rag.embeddings is not None else None

        # Manually save a fresh cache (mirrors logic in build_index)
        cache_file = None
        saved_cache = False
        try:
            if rag.kb_path and rag.embeddings is not None:
                kb_mtime = os.path.getmtime(rag.kb_path)
                safe_model = rag.sentence_model_name.replace('/', '_')
                cache_key = f"{os.path.basename(rag.kb_path)}_{safe_model}_{int(kb_mtime)}"
                cache_dir = "embeddings_cache"
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
                np.savez_compressed(
                    cache_file,
                    title_embeddings=rag.title_embeddings,
                    description_embeddings=rag.description_embeddings,
                    embeddings=rag.embeddings,
                )
                saved_cache = True
        except Exception as e:
            print(f"⚠️ Failed to persist rebuilt embeddings cache: {e}")

        return {
            "rebuilt": True,
            "records": records,
            "embedding_dim": emb_dim,
            "cache_file": cache_file,
            "cache_saved": saved_cache,
        }
    finally:
        # Restore prior setting
        if prev_disable is None:
            os.environ.pop("DISABLE_EMBEDDING_CACHE", None)
        else:
            os.environ["DISABLE_EMBEDDING_CACHE"] = prev_disable

def save_resolved_ticket_with_feedback(
    ticket_title: str,
    ticket_description: str,
    edited_response: str,
    predicted_team: str = None,
    predicted_classification: str = None,
    service_name: str = None,
    service_subcategory: str = None,
    knowledge_base_path: str = "tickets_large_first_reply_label.csv"
):
    """
    Save a resolved ticket with user-edited response to the knowledge base for future learning.
    
    This function enables continuous improvement by adding approved responses to the training data.
    The new ticket will be used by the RAG system for future similar requests.
    
    Args:
        ticket_title: Original ticket title
        ticket_description: Original ticket description
        edited_response: User-approved/edited response (final version sent to user)
        predicted_team: Team that handled the ticket (optional - will auto-classify if not provided)
        predicted_classification: Ticket classification (optional - will auto-classify if not provided)
        service_name: Service category (optional)
        service_subcategory: Service subcategory (optional)
        knowledge_base_path: Path to the CSV knowledge base
    
    Returns:
        dict: Status with keys 'success', 'message', 'ticket_ref', 'new_kb_size', 'embedding_invalidated'
    
    Example API usage:
        # Minimal usage - auto-classifies team and type
        result = save_resolved_ticket_with_feedback(
            ticket_title="VPN access needed",
            ticket_description="Need VPN for project ABC",
            edited_response="Need: VPN access approved for project ABC..."
        )
        
        # Or with explicit values
        result = save_resolved_ticket_with_feedback(
            ticket_title="VPN access needed",
            ticket_description="Need VPN for project ABC",
            edited_response="Need: VPN access approved for project ABC...",
            predicted_team="(GI-UX) Network Access",
            predicted_classification="vpn_request",
            service_name="Network",
            service_subcategory="VPN"
        )
    """
    global _RAG_CACHE  # Declare global at the start of the function
    
    try:
        # Auto-classify if not provided
        if predicted_team is None or predicted_classification is None:
            print("🔄 Auto-classifying ticket (team/classification not provided)...")
            ticket_text = f"{ticket_title} {ticket_description}"
            
            # Get classification if not provided
            if predicted_classification is None:
                predicted_classification, _ = classify_ticket(
                    ticket_text, 
                    service_category=service_name, 
                    service_subcategory=service_subcategory
                )
                print(f"📊 Auto-classified as: {predicted_classification}")
            
            # Get team if not provided
            if predicted_team is None:
                predicted_team, team_conf = classify_team_with_distilbert(
                    ticket_text,
                    service_category=service_name,
                    service_subcategory=service_subcategory
                )
                print(f"👥 Auto-assigned to team: {predicted_team} (confidence: {team_conf:.3f})")
        
        # Use cached knowledge base if available, otherwise load from disk
        # This ensures we work with the same data that's currently in the RAG system
        if _RAG_CACHE.get("df") is not None:
            df = _RAG_CACHE["df"].copy()
            print(f"📊 Using cached knowledge base: {len(df)} tickets")
        else:
            df = pd.read_csv(knowledge_base_path)
            print(f"📊 Loaded knowledge base from disk: {len(df)} tickets")
        
        # Generate a unique ticket reference
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ticket_ref = f"FEEDBACK_{timestamp}"
        
        # Format the response as it would appear in Public_log_anon
        # Simulate the format: timestamp separator + user info + response
        formatted_public_log = (
            f"********** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} **********\n"
            f": servicedesk (0) **********\n"
            f"{edited_response}\n"
        )
        
        # Create new row with all necessary columns
        new_row = {
            'Ticket_Reference': ticket_ref,
            'Title_anon': ticket_title,
            'Description_anon': ticket_description,
            'Public_log_anon': formatted_public_log,
            'first_reply': edited_response,  # Extracted first reply
            'label_auto': predicted_classification,
            'Team': predicted_team,
            'Service_Name': service_name or 'Unknown',
            'Service_Subcategory': service_subcategory or 'Unknown',
            'Status': 'Resolved',
            'Priority': 'Normal',
            'Source': 'API_Feedback'
        }
        
        # Add any missing columns with default values
        for col in df.columns:
            if col not in new_row:
                new_row[col] = None
        
        # Append new row to dataframe
        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save updated knowledge base
        new_df.to_csv(knowledge_base_path, index=False)
        
        # INCREMENTAL EMBEDDING: Add single embedding to existing FAISS index
        embedding_added = False
        
        if _RAG_CACHE.get("rag") is not None:
            try:
                print(f"⚡ Adding single embedding for new ticket (incremental update)...")
                rag_system = _RAG_CACHE["rag"]
                
                # Verify RAG system state before modification
                old_index_size = rag_system.index.ntotal
                old_kb_size = len(rag_system.knowledge_base)
                print(f"📊 Current state: {old_index_size} embeddings, {old_kb_size} tickets in KB")
                
                # Generate embedding for the new ticket (combined text)
                new_text = f"{ticket_title} {ticket_description}".strip()
                new_combined_emb = rag_system.sentence_model.encode([new_text], convert_to_numpy=True)
                new_combined_emb = np.array(new_combined_emb, dtype='float32')
                
                # Normalize the embedding for FAISS (cosine similarity)
                faiss.normalize_L2(new_combined_emb)
                
                # Generate individual title and description embeddings
                new_title_emb = rag_system.sentence_model.encode([ticket_title], convert_to_numpy=True)
                new_title_emb = np.array(new_title_emb, dtype='float32')
                
                new_desc_emb = rag_system.sentence_model.encode([ticket_description], convert_to_numpy=True)
                new_desc_emb = np.array(new_desc_emb, dtype='float32')
                
                # Add to FAISS index
                rag_system.index.add(new_combined_emb)
                print(f"✅ Added to FAISS index: {old_index_size} -> {rag_system.index.ntotal}")
                
                # Update all embedding arrays in memory
                if rag_system.embeddings is not None:
                    rag_system.embeddings = np.vstack([rag_system.embeddings, new_combined_emb])
                    print(f"✅ Updated embeddings array: {rag_system.embeddings.shape}")
                
                if rag_system.title_embeddings is not None:
                    rag_system.title_embeddings = np.vstack([rag_system.title_embeddings, new_title_emb])
                    print(f"✅ Updated title_embeddings: {rag_system.title_embeddings.shape}")
                
                if rag_system.description_embeddings is not None:
                    rag_system.description_embeddings = np.vstack([rag_system.description_embeddings, new_desc_emb])
                    print(f"✅ Updated description_embeddings: {rag_system.description_embeddings.shape}")
                
                # Update the knowledge base DataFrame in RAG system
                rag_system.knowledge_base = new_df.copy()  # Use copy to ensure clean state
                print(f"✅ Updated knowledge_base DataFrame: {old_kb_size} -> {len(rag_system.knowledge_base)} tickets")
                
                # Rebuild category index for filtering
                rag_system._build_category_index()
                print(f"✅ Rebuilt category index")
                
                # Update cache metadata
                _RAG_CACHE["df"] = new_df
                _RAG_CACHE["kb_mtime"] = os.path.getmtime(knowledge_base_path)
                _RAG_CACHE["rag"] = rag_system  # Ensure updated object is cached
                
                # Verify final state
                final_index_size = rag_system.index.ntotal
                final_kb_size = len(rag_system.knowledge_base)
                
                if final_index_size == old_index_size + 1 and final_kb_size == old_kb_size + 1:
                    embedding_added = True
                    print(f"✅ Verification passed: Index {final_index_size}, KB {final_kb_size}")
                    print(f"💡 New ticket immediately available for retrieval!")
                else:
                    print(f"⚠️ Verification failed: Expected +1, got index +{final_index_size - old_index_size}, KB +{final_kb_size - old_kb_size}")
                   
                    raise Exception("Incremental update verification failed")
                
            except Exception as e:
                import traceback
                print(f"⚠️ Incremental embedding failed: {e}")
                print(f"📋 Traceback: {traceback.format_exc()}")
                print(f"🔄 Cache will be invalidated - full rebuild on next request")
                _RAG_CACHE = {"kb_path": None, "kb_mtime": None, "rag": None, "df": None}
                _RAG_CACHE = {"kb_path": None, "kb_mtime": None, "rag": None, "df": None}
        else:
            # No active RAG system in cache, will rebuild on next request
            print(f"💡 No active RAG system - embedding will be built on next request")
        
        print(f"✅ Feedback saved successfully: {ticket_ref}")
        print(f"📊 Knowledge base size: {len(df)} -> {len(new_df)} tickets")
        
        return {
            "success": True,
            "message": "Ticket feedback saved successfully (incremental embedding added)" if embedding_added else "Ticket feedback saved successfully",
            "ticket_ref": ticket_ref,
            "new_kb_size": len(new_df),
            "embedding_added_incrementally": embedding_added,
            "embedding_invalidated": not embedding_added  # Only invalidated if incremental add failed
        }
        
    except Exception as e:
        print(f"❌ Error saving feedback: {e}")
        return {
            "success": False,
            "message": f"Failed to save feedback: {str(e)}",
            "ticket_ref": None,
            "new_kb_size": None,
            "embedding_invalidated": False
        }


# Main function to process a new ticket (now cached)
def process_new_ticket(ticket_title, ticket_description, knowledge_base_path="tickets_large_first_reply_label_copy.csv", force_rebuild=False, top_k: int = 5):
    """Process a new ticket with optional performance profiling."""
    import time
    profile_enabled = os.getenv("PROFILE_API", "false").lower() in ("1", "true", "yes")
    
    if profile_enabled:
        timings = {}
        start_total = time.time()
        
        # Step 1: Get RAG system
        t0 = time.time()
        rag_system, df = _get_or_build_rag(knowledge_base_path, force_rebuild=force_rebuild)
        timings['rag_load'] = time.time() - t0
        
        # Step 2: Generate response
        t0 = time.time()
        result = generate_response(ticket_title, ticket_description, rag_system, retrieval_k=top_k)
        timings['generation'] = time.time() - t0
        
        timings['total'] = time.time() - start_total
        result['_profiling'] = timings
        
        logger.info(f"⏱️ PROFILING: total={timings['total']:.2f}s rag={timings['rag_load']:.2f}s gen={timings['generation']:.2f}s")
        return result
    else:
        rag_system, df = _get_or_build_rag(knowledge_base_path, force_rebuild=force_rebuild)
        result = generate_response(ticket_title, ticket_description, rag_system, retrieval_k=top_k)
        return result

def retrieve_only(
    ticket_title: str,
    ticket_description: str,
    knowledge_base_path: str = DEFAULT_KB_PATH,
    top_k: int = 5,
    predicted_class: Optional[str] = None,
    predicted_team: Optional[str] = None,
):
    """Return only retrieval results with feedback-aware reranking, no LLM generation.

    This computes the same classification and team prediction used by generate_response
    to ensure retrieval filtering context matches, then returns only the retrieved items
    as a list of dicts plus basic metadata.
    """
    rag_system, df = _get_or_build_rag(knowledge_base_path, force_rebuild=False)

    ticket_text = f"{ticket_title} {ticket_description}"
    # Use provided predictions when available to keep scope consistent across reranks
    if predicted_class is None:
        predicted_class, _ = classify_ticket(ticket_text)
    else:
        word_class = predicted_class
    if predicted_team is None:
        predicted_team, team_confidence = classify_team_with_distilbert(ticket_text)
    else:
        team_confidence = None

    category_mapping = {
        "vpn_request": "to enable Client-to-Site VPN tunnel requests",
        "onboarding": "to start the IT On-boarding of a new internal employee",
        "software_request": "to request a software or a licence on my Windows device",
        "offboarding": "to start the IT Off-boarding process",
        "absence_request": "to request an Absence ticket",
        "admin_rights": "to request admin rights on the computer",
    }
    predicted_category = category_mapping.get(predicted_class or word_class, None)

    # Ensure feedback functions in RAG use the current DB path via env
    similar_replies = rag_system.retrieve_similar_replies(
        ticket_text,
        top_k=top_k,
        predicted_category=predicted_category,
    predicted_class=predicted_class or word_class,
        predicted_team=predicted_team,
    )

    return {
    "predicted_class": predicted_class or word_class,
        "predicted_team": predicted_team,
        "team_confidence": team_confidence,
        "similar_replies": similar_replies.to_dict(orient="records"),
        "retrieval_k": top_k,
    }

# Example usage
if __name__ == "__main__":
    knowledge_base_path = "tickets_large_first_reply_label_copy.csv"#"tickets_large_first_reply_label.csv"
    ticket_title = "VPN access request"
    ticket_description = "Requesting VPN access for project KE-123456. User: Jane Smith."

    result = process_new_ticket(ticket_title, ticket_description, knowledge_base_path)
    print("*"*50)
    print("Classification:", result["classification"])
    print("Predicted Team:", result["predicted_team"])
    print("Generated Response:", result["response"])
    print("Similar Replies:", result["similar_replies"])
    # save_resolved_ticket_with_feedback(ticket_title, ticket_description, result["response"])