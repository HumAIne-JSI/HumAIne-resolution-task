# IT Support Ticket Resolution System

An intelligent IT support system that uses Retrieval-Augmented Generation (RAG) to automatically generate contextually appropriate responses to support tickets.

## Features

- **Semantic Similarity Search**: FAISS-based vector similarity for finding relevant historical tickets
- **Intelligent Classification**: DistilBERT models for team assignment (23 teams) and ticket categorization (10 classes)
- **Adaptive Response Generation**: GPT-3.5-turbo with template vs personalized decision policy
- **Incremental Learning**: Feedback loop that adds approved responses without full index rebuild
- **Human-in-the-Loop**: Streamlit UI for response review and editing
- **Performance Optimized**: Lazy model loading, embedding caching, efficient incremental updates

## Architecture

```
Input Ticket
    ↓
Classification (Team + Type)
    ↓
Embedding Generation (384-dim)
    ↓
FAISS Retrieval (top-k similar)
    ↓
Decision Policy (Template vs Personalized)
    ↓
GPT-3.5-turbo Generation
    ↓
Human Review (Streamlit UI)
    ↓
Feedback Loop (Incremental Update)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `sentence-transformers>=2.2.0`
- `faiss-cpu>=1.7.0` (or `faiss-gpu` for GPU support)
- `transformers>=4.30.0`
- `torch>=2.0.0`
- `openai>=1.0.0`
- `pandas>=1.5.0`
- `streamlit>=1.20.0`

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your OpenAI API key:
```bash
OPENAI_API_KEY=sk-...your_key_here...
```

3. Ensure model directories exist:
- `./perfect_team_classifier/` - DistilBERT team classifier
- `./ticket_classifier_model/` - DistilBERT ticket type classifier

4. Prepare knowledge base CSV with required columns:
- `Title_anon`: Ticket title
- `Description_anon`: Ticket description
- `Public_log_anon`: Historical conversation log
- `first_reply`: Extracted first response
- `Team`: Assigned team
- `label_auto`: Ticket classification

## Usage

### Command Line

```python
from resolution_task import process_new_ticket, save_resolved_ticket_with_feedback

# Process a new ticket
result = process_new_ticket(
    ticket_title="VPN access request",
    ticket_description="Need VPN access for project KE-123456",
    knowledge_base_path="tickets_large_first_reply_label_copy.csv"
)

print(f"Classification: {result['classification']}")
print(f"Team: {result['predicted_team']}")
print(f"Response: {result['response']}")

# Save feedback after human review
feedback = save_resolved_ticket_with_feedback(
    ticket_title="VPN access request",
    ticket_description="Need VPN access for project KE-123456",
    edited_response=reviewed_response,  # Human-approved version
    predicted_team=result['predicted_team'],
    predicted_classification=result['classification']
)

print(f"Feedback saved: {feedback['ticket_ref']}")
print(f"KB size: {feedback['new_kb_size']}")
```

### Streamlit UI

```bash
streamlit run streamlit_resolution_ui.py
```

## Performance Metrics

- **Startup time**: ~0s (lazy loading)
- **Embedding generation**: 50-200ms per ticket
- **FAISS retrieval**: <10ms (top-5)
- **GPT-3.5-turbo latency**: 3-8ms
- **Incremental update**: 50-200ms (vs 30-60s full rebuild)

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB (models + embeddings cache)

### Recommended
- CPU: 8+ cores or GPU (CUDA 11.0+)
- RAM: 16GB
- Storage: 10GB

## API Reference

### `process_new_ticket()`

Process a new support ticket and generate a response.

**Parameters:**
- `ticket_title` (str): Ticket title
- `ticket_description` (str): Detailed ticket description
- `knowledge_base_path` (str): Path to CSV knowledge base
- `force_rebuild` (bool): Force full embedding rebuild (default: False)
- `top_k` (int): Number of similar tickets to retrieve (default: 5)

**Returns:**
```python
{
    "classification": str,           # Ticket type (e.g., "vpn_request")
    "predicted_team": str,           # Assigned team
    "team_confidence": float,        # Confidence score (0-1)
    "response": str,                 # Generated response
    "similar_replies": list[dict],   # Retrieved similar tickets
    "retrieval_k": int              # Number of retrievals used
}
```

### `save_resolved_ticket_with_feedback()`

Save a resolved ticket with edited response to knowledge base.

**Parameters:**
- `ticket_title` (str): Original ticket title
- `ticket_description` (str): Original description
- `edited_response` (str): Human-approved final response
- `predicted_team` (str, optional): Team assignment
- `predicted_classification` (str, optional): Ticket type
- `service_name` (str, optional): Service category
- `service_subcategory` (str, optional): Service subcategory
- `knowledge_base_path` (str): Path to CSV knowledge base

**Returns:**
```python
{
    "success": bool,
    "message": str,
    "ticket_ref": str,               # Unique reference ID
    "new_kb_size": int,              # Updated KB size
    "embedding_added_incrementally": bool,
    "embedding_invalidated": bool
}
```

## Models

### Team Classifier
- **Architecture**: DistilBERT
- **Labels**: 23 teams (e.g., "(GI-UX) Network Access")
- **Input format**: `[TITLE] {title} [DESCRIPTION] {description}`

### Ticket Type Classifier
- **Architecture**: DistilBERT
- **Classes**: 10 types (vpn_request, onboarding, admin_rights, etc.)
- **Input format**: `[TITLE] {title} [DESCRIPTION] {description}`

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Index**: FAISS IndexFlatIP (cosine similarity)

## Troubleshooting

### "OpenAI client unavailable"
- Ensure `OPENAI_API_KEY` is set in `.env` or environment
- Install OpenAI package: `pip install openai>=1.0.0`

### "Team/Ticket classifier not found"
- Download pre-trained models to correct directories
- Check paths in `.env` configuration

### "BatchEncoding cast warning"
- Harmless tokenizer warning, does not affect results
- Can be ignored

### Slow performance
- Enable embedding cache (default)
- Use GPU if available (install `faiss-gpu`)
- Reduce `top_k` parameter for faster retrieval

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Proprietary - Internal use only

## Authors

- JSI Team
- Data Engineering Team (GFT)

## Acknowledgments

- Sentence Transformers library
- Hugging Face Transformers
- OpenAI GPT-3.5-turbo
- FAISS vector search library
