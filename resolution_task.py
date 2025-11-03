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
LLM_MAX_TOKENS = 800
LLM_TEMPERATURE = 0.2

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


# Load the knowledge base (tickets_large_first_reply_label.csv)
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

    def retrieve_similar_replies(self, query, top_k=5, predicted_category=None):
        """Enhanced retrieval with multi-factor scoring and category awareness"""
        
        # Get larger candidate set for re-ranking
        search_k = min(top_k * 4, len(self.knowledge_base))
        
        # Step 1: Get candidates from FAISS
        query_embedding = self.sentence_model.encode([query])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), search_k)
        
        candidates = self.knowledge_base.iloc[indices[0]].copy()
        candidates['faiss_score'] = distances[0]
        
        # Step 2: Calculate enhanced scores
        enhanced_scores = []
        for idx, row in candidates.iterrows():
            score = self._calculate_enhanced_score(query, row, predicted_category, distances[0][candidates.index.get_loc(idx)])
            enhanced_scores.append(score)
        
        candidates['enhanced_score'] = enhanced_scores
        
        # Step 3: Re-rank and return top results
        results = candidates.nlargest(top_k, 'enhanced_score')
        return results[['first_reply', 'Title_anon', 'Description_anon', 'enhanced_score']]

    def _calculate_enhanced_score(self, query, candidate_row, predicted_category, faiss_score):
        """Calculate enhanced score considering multiple factors"""
        # Base FAISS similarity (50%)
        base_score = 0.5 * faiss_score
        
        # Category match bonus (20%)
        category_bonus = 0.0
        if predicted_category and candidate_row.get('label_auto') == predicted_category:
            category_bonus = 0.2
        
        # Title similarity (15%)
        title_sim = self._calculate_field_similarity(query, candidate_row.get('Title_anon', ''))
        title_score = 0.15 * title_sim
        
        # Description similarity (10%)
        desc_sim = self._calculate_field_similarity(query, candidate_row.get('Description_anon', ''))
        desc_score = 0.1 * desc_sim
        
        # Response quality bonus (5%)
        quality_bonus = 0.05 * self._assess_response_quality(candidate_row.get('first_reply', ''))
        
        return base_score + category_bonus + title_score + desc_score + quality_bonus

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
        
        # Get device from team classifier or detect
        _, _, device = _get_team_classifier()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        _ticket_classifier.to(device)
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
            # Tokenize input
            inputs = ticket_tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=512
            ).to(device)
            
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
    """Simple analysis to determine if response is template-based or personalized"""
    if not response_text or pd.isna(response_text):
        return "unknown", 0.0
    
    text = str(response_text).lower()
    
    # Template indicators (simple keyword detection)
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
        "this ticket has been"
    ]
    
    # Personalized indicators
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
        "thank you for your request"
    ]
    
    # Count matches
    template_score = sum(1 for keyword in template_keywords if keyword in text)
    personalized_score = sum(1 for keyword in personalized_keywords if keyword in text)
    
    # Check for structured lists (template indicator)
    dash_lines = text.count('\n-')
    if dash_lines >= 3:
        template_score += 2
    
    # Check for questions (personalized indicator)
    question_marks = text.count('?')
    if question_marks > 0:
        personalized_score += 1
    
    # Determine type
    total_score = template_score + personalized_score
    if total_score == 0:
        return "unknown", 0.0
    
    template_confidence = template_score / total_score
    
    if template_confidence > 0.6:
        return "template", template_confidence
    elif template_confidence < 0.4:
        return "personalized", 1 - template_confidence
    else:
        return "mixed", 0.5

# Update OpenAI API call to ensure compatibility and proper response handling
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert IT support specialist who writes clear, professional responses that match organizational standards."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.3,
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an IT operations communicator writing concise status updates."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2,
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
    
    print(f"🔍 Analyzing {len(similar_replies)} similar replies...")
    
    for _, reply in similar_replies.iterrows():
        first_reply = str(reply.get('first_reply', ''))
        response_type, confidence = analyze_response_type_simple(first_reply)
        
        print(f"   Reply: {response_type} (confidence: {confidence:.2f})")
        
        if response_type == "template" and confidence > 0.5:
            template_examples.append(reply)
        elif response_type == "personalized" and confidence > 0.5:
            personalized_examples.append(reply)
    
    # Decide which approach to use
    use_template = len(template_examples) > len(personalized_examples)

    # If temporal/status-update context is detected, prefer personalized status-update responses
    if temporal_context == 'temporal_update':
        print("🔔 Temporal context detected — forcing personalized/status-update response")
        use_template = False

    print(f"📊 Decision: {'Template' if use_template else 'Personalized'}")
    print(f"   Template examples: {len(template_examples)}")
    print(f"   Personalized examples: {len(personalized_examples)}")
    
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
    """Generate template-based response with enhanced similarity matching"""
    
    # Ensure template_examples is a DataFrame
    if isinstance(template_examples, list):
        template_examples = pd.DataFrame(template_examples)
    
    # Enhanced template selection - find the MOST similar template
    if len(template_examples) > 1:
        template_examples = select_best_templates(ticket_title, ticket_description, template_examples, top_k=3)
    
    # Create more specific prompts based on ticket type
    specific_instructions = get_specific_instructions(classification, predicted_team)
    
    prompt = (
        f"You are an IT support system that generates STRUCTURED TEMPLATE RESPONSES. "
        f"You must follow the EXACT format and structure shown in the examples below.\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- Copy the EXACT format from the most similar ticket\n"
        f"- Use IDENTICAL structure, field names, and formatting\n"
        f"- Replace ONLY the specific values (names, IDs, etc.)\n"
        f"- Keep ALL punctuation, spacing, and line breaks\n"
        f"- Start with 'Below you will find' when appropriate\n"
        f"- Use bullet points (-) exactly as shown\n"
        f"{specific_instructions}\n\n"
        f"Current Ticket:\n"
        f"Title: {ticket_title}\n"
        f"Description: {ticket_description}\n"
        f"Request Type: {classification}\n"
        f"Assigned Team: {predicted_team}\n\n"
        f"TEMPLATE EXAMPLES (RANKED BY SIMILARITY):\n"
    )
    
    # Add template examples with enhanced context
    example_count = 0
    for index, reply in template_examples.iterrows():
        if example_count >= 2:  # Reduced to 2 best examples for focus
            break
            
        similarity_note = f"[SIMILARITY: {reply.get('enhanced_score', 0.0):.3f}]" if 'enhanced_score' in reply else ""
        prompt += (
            f"EXAMPLE {example_count + 1} {similarity_note}:\n"
            f"Title: {reply.get('Title_anon', 'N/A')}\n"
            f"Description: {reply.get('Description_anon', 'N/A')[:200]}...\n"
            f"EXACT TEMPLATE TO FOLLOW:\n{reply.get('first_reply', 'N/A')}\n"
            f"{'='*60}\n\n"
        )
        example_count += 1
    
    prompt += (
        f"GENERATE RESPONSE:\n"
        f"Use the EXACT structure from the most similar example above. "
        f"Copy the format precisely - same field names, same punctuation, same layout. "
        f"Change ONLY the specific details relevant to the current ticket. "
        f"Do NOT add extra text or modify the template structure."
    )

    print("Prompt sent to OpenAI API (Enhanced Template):")
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)

    # NO TRY-EXCEPT - Let errors propagate to see what's wrong
    return _chat_completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an IT support system that generates responses matching the exact format and style of provided examples."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.2,
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
    """Enhanced response generation with improved retrieval"""
    ticket_text = ticket_title + " " + ticket_description
    word_class, template = classify_ticket(ticket_text)

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
    predicted_team, team_confidence = classify_team_with_distilbert(ticket_text)

    # Detect temporal/status-update context and pass to generator
    temporal_context = detect_temporal_context(ticket_title, ticket_description)

    # Use enhanced retrieval with category awareness
    similar_replies = rag_system.retrieve_similar_replies(
        ticket_text,
        top_k=retrieval_k,
        predicted_category=predicted_category
    )

    # Generate response with better context
    response = generate_response_with_openai(ticket_title, ticket_description, word_class, predicted_team, team_confidence, similar_replies, temporal_context=temporal_context)

    return {
        "classification": word_class,
    "predicted_team": predicted_team,
    "team_confidence": team_confidence,
        "response": response,
        "similar_replies": similar_replies.to_dict(orient="records"),
    "predicted_class": word_class,
    "retrieval_k": retrieval_k
    }

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
    rag_system, df = _get_or_build_rag(knowledge_base_path, force_rebuild=force_rebuild)
    result = generate_response(ticket_title, ticket_description, rag_system, retrieval_k=top_k)

    # NO FALLBACK - Let any errors propagate so we can see what's wrong
    return result

# Example usage
if __name__ == "__main__":
    knowledge_base_path = "tickets_large_first_reply_label.csv"#"tickets_large_first_reply_label.csv"
    ticket_title = "VPN access request"
    ticket_description = "Requesting VPN access for project KE-123456. User: Jane Smith."

    result = process_new_ticket(ticket_title, ticket_description, knowledge_base_path)
    print("*"*50)
    print("Classification:", result["classification"])
    print("Predicted Team:", result["predicted_team"])
    print("Generated Response:", result["response"])
    print("Similar Replies:", result["similar_replies"])
    # save_resolved_ticket_with_feedback(ticket_title, ticket_description, result["response"])   