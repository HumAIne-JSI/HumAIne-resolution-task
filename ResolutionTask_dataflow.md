![][image1]

| Feature vector composition                                                                                                                                                       |                                                                                            |                                                  |                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :----------------------------------------------- | :---------------------------------------------------------- |
| **Title**<br><em>VPN access request</em>                                                                                                                                         | **Description**<br><em>Requesting VPN access for project KE-123456. User: Jane Smith.</em> | **Service**<br><em>Network</em>                  | **Service Subcategory**<br><em>VPN Access</em>              |
| **Text composed for embedding**<br><strong>Joint: Title + Description</strong><br><em>"VPN access request - Requesting VPN access for project KE-123456. User: Jane Smith."</em> | **Separate embeddings**<br>Title: <em>No</em><br>Description: <em>No</em>                  | **Service (embedding)**<br><em>Not included</em> | **Subcategory (embedding)**<br><em>Not included</em>        |
| **Embedding vector (used in FAISS)**<br><em>-0.0642, 0.0158, -0.0327, ...</em>                                                                                                   | &nbsp;                                                                                     | &nbsp;                                           | &nbsp;                                                      |
| &nbsp;                                                                                                                                                                           | &nbsp;                                                                                     | &nbsp;                                           | &nbsp;                                                      |
| **Title + Description** -> 384 dims (all-MiniLM-L6-v2)                                                                                                                           | &nbsp;                                                                                     | **Service** -> <em>metadata (not embedded)</em>  | **Service Subcategory** -> <em>metadata (not embedded)</em> |

**Feature vector**

- **Total length:** 384
- **Included features:**
  - **Title** — short summary of the helpdesk ticket
  - **Description** — detailed ticket body
  - **Service** — high-level category (e.g., Network, End-User Computing) — used for filtering/analytics (not embedded)
  - **Service Subcategory** — subcategory (e.g., VPN Access) — used for filtering/analytics (not embedded)
  - **Embedding generation mode** — Joint (Title + Description). Separate title/description embeddings are not used for retrieval.
- **Processing steps:**
  - Concatenate Title and Description into one string.
  - Compute a 384-dim sentence embedding using Sentence Transformers (all-MiniLM-L6-v2).
  - L2-normalize embedding and index in FAISS IndexFlatIP for cosine similarity.
  - Optionally filter candidates by Service/Service Subcategory at retrieval time.
  - Retrieve top-k=5 similar tickets and generate response via GPT-3.5-turbo.
  - User reviews/edits response in Streamlit UI and saves feedback.
  - Append feedback to CSV knowledge base and incrementally add its embedding to FAISS (no full rebuild).

| Data processing pipeline           |                                                                                                                                                             |                        |                            |                            |
| ---------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------- | :------------------------- | :------------------------- |
| **Process**                        | **Description**                                                                                                                                             | **Location**           | **Responsible Actor**      | **Format**                 |
| Data Collection                    | Ticket data stored/exported to CSV knowledge base.                                                                                                          | Ticketing system / CSV | IT Ops (GFT)               | CSV                        |
| Data anonymization                 | Sensitive fields pseudonymized/removed (names, emails, IDs).                                                                                                | Anonymization pipeline | Data engineer (GFT)        | CSV                        |
| Data cleaning                      | Remove invalid rows, enforce required fields (Title, Description), normalize whitespace.                                                                    | Preprocessing pipeline | Automated scripts (JSI)    | CSV                        |
| Embedding & indexing               | Compute 384-dim embeddings and build FAISS index (IndexFlatIP, L2-normalized).                                                                              | Notebooks / RAG module | Automated scripts (JSI)    | Numpy arrays + FAISS index |
| Retrieval                          | Query embedding vs FAISS to get top-k similar tickets (k configurable, e.g., 5).                                                                            | RAG module             | Application (JSI)          | In-memory results          |
| LLM response generation            | Compose structured prompt with retrieved snippets; generate reply via GPT-3.5-turbo. Decision policy: Template vs Personalized based on retrieved examples. | RAG module             | Application (JSI)          | Text response              |
| Human review & editing             | Analyst reviews and edits AI response in Streamlit UI.                                                                                                      | Streamlit app          | Service desk analyst (GFT) | UI text                    |
| Feedback save & incremental update | Append edited reply to CSV, generate embedding, add 1 vector to FAISS without full rebuild.                                                                 | RAG module             | Application (JSI)          | CSV + FAISS updated        |
| Optional model training            | DistilBERT team classifier and ticket-type classifier retrained on curated data.                                                                            | Training module        | ML engineer (JSI)          | Model artifacts            |
| Evaluation                         | Retrieval P@k, response acceptance rate, classifier F1; periodic reports.                                                                                   | Evaluation module      | Application (JSI)          | Metrics JSON/CSV           |

**Runtime snapshot (example)**

- RAG index: 1597 tickets indexed; device: CPU
- Team classifier: DistilBERT (23 labels) — predicted (GI-UX) Network Access (0.752)
- Ticket classifier: DistilBERT (10 classes) — predicted vpn_request (0.998)
- Retrieval: top-k = 5 similar replies (all templates in this example)
- Decision: Template
- Note: harmless tokenizer warning may appear (BatchEncoding cast); does not affect results.

  -

[image1]: MANUAL_DRAW_THIS_DIAGRAM "Manual: Recreate as a pipeline diagram with labeled boxes and arrows. Boxes (left→right): Data Collection → Anonymization → Cleaning → Embedding (384-dim) → FAISS Index → Retrieval (top-k) → LLM (gpt-3.5-turbo) → Human Review (Streamlit) → Feedback Save (CSV) → Incremental Add (FAISS). Use 16x16 layout, titles 32pt, headers 20pt, body 18pt, arrow linewidth=3. Save as notebooks/public/ResolutionTask_dataflow.png and update this link to the final image path."
