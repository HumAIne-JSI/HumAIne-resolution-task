"""Streamlit frontend for the resolution_task pipeline.

Features:
- Input ticket title + description (and optional service category fields, kept generic)
- Run generation via process_new_ticket from resolution_task without modifying core code
- Display retrieved similar tickets (if available) and their first replies
- Editable generated response area
- Simple quality / similarity placeholder (if the backend returns similarities)
- Download / copy convenience + a stub "Send" action

This frontend is defensive: resolution_task.py contains unfinished stubs; we
therefore:
- Wrap imports & calls in try/except
- Tolerate missing keys in the result dict
- Provide fallback messaging if retrieval/generation not implemented yet

Run:
    streamlit run notebooks/streamlit_resolution_ui.py

Prereqs:
    pip install streamlit pandas sentence-transformers scikit-learn

Environment (recommended):
    Set OPENAI_API_KEY instead of using any hard-coded key in backend.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import base64
import os
import json
import streamlit as st

# ------------------------------ Branding Helpers --------------------------- #
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), 'public')

def _asset_path(name: str) -> str:
    p = os.path.join(PUBLIC_DIR, name)
    return p if os.path.exists(p) else ''

def _img_base64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

LOGO_LIGHT = _asset_path('logo_light.png')
LOGO_DARK = _asset_path('logo_dark.png')
FAVICON = _asset_path('favicon.png')
BG_IMAGE = _asset_path('humaine-vector-img-scaled.jpg')

BG_B64 = _img_base64(BG_IMAGE)
LOGO_B64 = _img_base64(LOGO_LIGHT) or _img_base64(LOGO_DARK)

# Inject global CSS for "humaine" vibe (gradient teal/navy, soft cards)
bg_overlay_css = ""
if BG_B64:
        bg_overlay_css = f"""
        .stApp::before {{
                content: '';
                position: fixed;
                inset:0;
                background: url(data:image/png;base64,{BG_B64}) center/cover no-repeat;
                opacity:1;
                pointer-events:none;
                z-index:0;
        }}
        """.strip()

custom_css = f"""
<style>
body {{
    background: linear-gradient(135deg, #f5f9fa 0%, #eef6f9 40%, #e8f3f6 100%) !important;
}}
section.main > div {{ background: transparent !important; }}
{bg_overlay_css}
h1, h2, h3, h4 {{ font-family:'Segoe UI','Inter',sans-serif; font-weight:600; letter-spacing:0.5px; }}
p, div, span, label {{ font-family:'Segoe UI','Inter',sans-serif; }}
div.block-container {{ padding-top:1.2rem; }}
.st-expander, .stTextArea, .stTextInput, .stSelectbox, .stNumberInput, .stDownloadButton button, .stButton button {{ border-radius:10px !important; }}
.small-metric {{ background:rgba(255,255,255,0.65); backdrop-filter:blur(6px); padding:0.75rem 0.9rem; border:1px solid #d9e5eb; border-radius:12px; box-shadow:0 2px 4px rgba(0,0,0,0.04); font-size:0.8rem; line-height:1.2rem; color:#1d2d35; }}
.small-metric strong {{ color:#1d4252; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; }}
.stButton button {{ background:linear-gradient(90deg,#1d4252 0%, #2d7486 60%, #35a6b3 100%); color:#fff; font-weight:600; border:0; box-shadow:0 2px 6px rgba(0,0,0,0.15); }}
.stButton button:hover {{ filter:brightness(1.08); }}
details {{ border-radius:14px !important; overflow:hidden; }}
summary {{ font-weight:600; }}
pre, code {{ font-size:0.80rem !important; }}
section[data-testid='stSidebar'] > div {{ background:linear-gradient(180deg,#11252e 0%, #1d4252 60%, #2d7486 100%); color:#fff; }}
section[data-testid='stSidebar'] h1, section[data-testid='stSidebar'] h2, section[data-testid='stSidebar'] h3, section[data-testid='stSidebar'] p, section[data-testid='stSidebar'] label {{ color:#f2f8fa !important; }}
section[data-testid='stSidebar'] .stTextInput input, section[data-testid='stSidebar'] .stSlider, section[data-testid='stSidebar'] .stCheckbox div {{ color:#102229 !important; }}
footer {{ visibility:hidden; }}
::-webkit-scrollbar {{ width:10px; }}
::-webkit-scrollbar-track {{ background:rgba(0,0,0,0.05); }}
::-webkit-scrollbar-thumb {{ background:#2d7486; border-radius:5px; }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

if LOGO_B64:
    # Extra CSS to make logo stand out
    st.sidebar.markdown(
        """
        <style>
        .sidebar-logo-card {position:relative; text-align:center; margin:0.4rem 0 1.2rem; padding:14px 12px 18px; border-radius:20px; 
            background:linear-gradient(145deg,rgba(255,255,255,0.18),rgba(255,255,255,0.05));
            border:1px solid rgba(255,255,255,0.28); box-shadow:0 8px 22px -10px rgba(0,0,0,0.55), 0 2px 6px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.08) inset; backdrop-filter:blur(10px);}
        .sidebar-logo-card::before {content:''; position:absolute; inset:0; padding:1px; border-radius:inherit; background:linear-gradient(130deg,#35a6b3,#2d7486,#1d4252); -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0); -webkit-mask-composite:xor; mask-composite:exclude; opacity:0.65;}
        .sidebar-logo-card img {max-width:170px; width:100%; filter:drop-shadow(0 8px 14px rgba(0,0,0,0.55)); transition:transform .45s cubic-bezier(.19,1,.22,1), filter .45s ease;}
        .sidebar-logo-card:hover img {transform:scale(1.06) translateY(-2px) rotate(-1deg); filter:drop-shadow(0 12px 24px rgba(0,0,0,0.55));}
        .sidebar-logo-sub {font-size:0.68rem; letter-spacing:0.18em; text-transform:uppercase; color:#d3e8ed; margin-top:0.65rem; font-weight:600;}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f"""
        <div class='sidebar-logo-card'>
            <a href='https://humaine-horizon.eu/' target='_blank' rel='noopener noreferrer' style='text-decoration:none; display:block;'>
                <img src='data:image/png;base64,{LOGO_B64}' alt='Project Humaine Logo'>
                <div class='sidebar-logo-sub'>Project Humaine</div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

import streamlit as st  # re-import safe (already imported above)

# Lazy / guarded import of resolution_task
try:
    import resolution_task  # noqa: F401
    from resolution_task import process_new_ticket  # type: ignore
    from resolution_task import save_resolved_ticket_with_feedback  # type: ignore
    try:
        from resolution_task import rebuild_embeddings  # type: ignore
    except Exception:
        rebuild_embeddings = None  # type: ignore
except Exception as e:  # pragma: no cover
    process_new_ticket = None  # type: ignore
    save_resolved_ticket_with_feedback = None  # type: ignore
    rebuild_embeddings = None  # type: ignore
    _IMPORT_ERR = str(e)
else:
    _IMPORT_ERR = None

DEFAULT_KNOWLEDGE_BASE = "tickets_large_first_reply_label_copy.csv"

st.set_page_config(
    page_title="Humaine Resolution Assistant",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Sidebar ------------------------------------ #
with st.sidebar:
    st.title("⚙️ Settings")
    knowledge_base_path = st.text_input(
        "Knowledge Base CSV", value=DEFAULT_KNOWLEDGE_BASE, help="Path relative to repo root/notebooks"
    )
    top_k = st.slider("Top-k similar replies", 1, 15, 5)
    if rebuild_embeddings:
        if st.button("🔄 Generate / Rebuild Embeddings", use_container_width=True):
            with st.spinner("Rebuilding embeddings (forcing fresh computation)..."):
                try:
                    stats = rebuild_embeddings(knowledge_base_path)
                    st.success(f"Rebuilt: {stats.get('records')} rows | dim {stats.get('embedding_dim')} | cache: {'saved' if stats.get('cache_saved') else 'not saved'}")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")
    else:
        st.caption("(Rebuild function not available in backend)")
    show_raw_json = st.checkbox("Show raw backend JSON", value=False)
    st.markdown("---")
    st.markdown("### ℹ️ Notes")
    st.caption(
        "If functions in resolution_task are incomplete, you'll see fallback messages."
    )
    if _IMPORT_ERR:
        st.error(f"Failed to import resolution_task: {_IMPORT_ERR}")

# ------------------------------ Main Input --------------------------------- #
st.title("Humaine Ticket Resolution Assistant")

with st.expander("Ticket Input", expanded=True):
    col_a, col_b = st.columns([1, 3])
    with col_a:
        ticket_title = st.text_input("Title", placeholder="e.g. Need admin rights for software install")
        service_category = st.text_input("Service Category (optional)")
        service_subcategory = st.text_input("Service Subcategory (optional)")
    with col_b:
        ticket_description = st.text_area(
            "Description",
            height=200,
            placeholder="Describe the issue / request...",
        )

    run_button = st.button("🚀 Generate First Reply", type="primary", use_container_width=True)

# ------------------------------ Helper funcs -------------------------------- #

def safe_process_ticket(title: str, desc: str, k: int) -> Dict[str, Any]:
    if not process_new_ticket:
        return {"error": "process_new_ticket not available – backend import failed."}
    try:
        result = process_new_ticket(
            ticket_title=title,
            ticket_description=desc,
            knowledge_base_path=knowledge_base_path,
            top_k=k,
        )
        if not isinstance(result, dict):  # Normalize
            return {"warning": "Backend returned non-dict result", "raw": str(result)}
        return result
    except Exception as e:  # pragma: no cover
        return {"error": f"Exception during generation: {e}"}

# ------------------------------ Run pipeline -------------------------------- #
result: Dict[str, Any] | None = None
if run_button:
    if not ticket_title.strip():
        st.warning("Please provide a title.")
    elif not ticket_description.strip():
        st.warning("Please provide a description.")
    else:
        with st.spinner("Running resolution pipeline..."):
            result = safe_process_ticket(ticket_title, ticket_description, top_k)

# ------------------------------ Output Sections ---------------------------- #
if result:
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("Generation completed.")
        st.caption("Inference powered by internal models + GPT backend.")

    # Core metadata
    meta_cols = st.columns(4)
    def get_field(key: str, default: str = "–"):
        val = result.get(key)
        return default if val in (None, "") else val

    team_conf_raw = result.get("team_confidence")
    if isinstance(team_conf_raw, (int, float)):
        team_conf_display = f"{team_conf_raw:.2f}"
    else:
        team_conf_display = "–"

    metric_style = "<style>.small-metric div[data-testid='stMetricValue'] {font-size: 0.9rem;}</style>"
    st.markdown(metric_style, unsafe_allow_html=True)

    with meta_cols[0]:
        st.container().markdown(f"<div class='small-metric'><strong>Predicted Team</strong><br>{get_field('predicted_team')}</div>", unsafe_allow_html=True)
    with meta_cols[1]:
        st.container().markdown(f"<div class='small-metric'><strong>Classification</strong><br>{get_field('classification')}</div>", unsafe_allow_html=True)
    with meta_cols[2]:
        st.container().markdown(f"<div class='small-metric'><strong>Team Confidence</strong><br>{team_conf_display}</div>", unsafe_allow_html=True)
    with meta_cols[3]:
        st.container().markdown(f"<div class='small-metric'><strong>Retrieval k</strong><br>{get_field('retrieval_k', str(top_k))}</div>", unsafe_allow_html=True)

    # Retrieved similar replies
    st.subheader("🔍 Retrieved Similar Tickets")
    similar = result.get("similar_replies") or result.get("similar") or []

    if isinstance(similar, list) and len(similar) > 0:
        # Limit / slice to UI selection of top_k
        display_similar = similar[:top_k]
        for idx, row in enumerate(display_similar, 1):
            with st.expander(f"Similar #{idx}"):
                title_val = row.get("Title_anon") if isinstance(row, dict) else row
                desc_val = row.get("Description_anon") if isinstance(row, dict) else ""
                first_reply = row.get("first_reply") if isinstance(row, dict) else ""
                st.markdown(f"**Title:** {title_val}")
                if desc_val:
                    st.markdown(f"**Description:**\n\n{desc_val}")
                if first_reply:
                    st.markdown("**First Reply:**")
                    st.code(first_reply[:4000])
    elif isinstance(similar, list) and len(similar) == 0:
        st.info("No similar tickets returned.")
    else:
        st.warning("Similar replies structure not recognized.")

    # Generated response (editable)
    st.subheader("✍️ Generated First Reply")
    generated_text = result.get("response") or result.get("generated_response") or "(No response returned)"
    edited_response = st.text_area(
        "Edit before sending (this will not overwrite backend unless you build a save endpoint)",
        value=generated_text,
        height=300,
    )

    colx, coly, colz = st.columns([1,1,2])
    with colx:
        download_name = f"ticket_reply_{ticket_title.replace(' ', '_')[:40]}.txt" or "ticket_reply.txt"
        st.download_button(
            label="💾 Download",
            data=edited_response.encode("utf-8"),
            file_name=download_name,
            mime="text/plain",
        )
    with coly:
        if st.button("📨 Send & Save to KB", use_container_width=True):
            if not save_resolved_ticket_with_feedback:
                st.warning("Feedback save function not available in backend.")
            else:
                # Debug: Show what we're about to save
                st.write("🔍 **Debug Info:**")
                st.write(f"- Title: `{ticket_title}`")
                st.write(f"- Description length: {len(ticket_description)} chars")
                st.write(f"- Response length: {len(edited_response)} chars")
                
                with st.spinner("Saving resolved ticket to knowledge base..."):
                    try:
                        # Extract metadata from result (optional - will auto-classify if not available)
                        predicted_team = result.get("predicted_team") if result else None
                        predicted_classification = result.get("classification") if result else None
                        
                        st.write(f"- Team: `{predicted_team or 'Will auto-classify'}`")
                        st.write(f"- Classification: `{predicted_classification or 'Will auto-classify'}`")
                        st.write(f"- KB Path: `{knowledge_base_path}`")
                        
                        # Save the feedback (team and classification are now optional)
                        feedback_result = save_resolved_ticket_with_feedback(
                            ticket_title=ticket_title,
                            ticket_description=ticket_description,
                            edited_response=edited_response,
                            predicted_team=predicted_team,  # Optional - will auto-classify if None
                            predicted_classification=predicted_classification,  # Optional - will auto-classify if None
                            service_name=service_category if service_category and service_category.strip() else None,
                            service_subcategory=service_subcategory if service_subcategory and service_subcategory.strip() else None,
                            knowledge_base_path=knowledge_base_path
                        )
                        
                        st.write(f"📤 **Feedback result received:**")
                        st.json(feedback_result)
                        
                        if feedback_result.get("success"):
                            st.success(f"✅ {feedback_result['message']}")
                            st.info(
                                f"📊 **Ticket Reference:** {feedback_result['ticket_ref']}\n\n"
                                f"**Knowledge Base Size:** {feedback_result['new_kb_size']} tickets\n\n"
                                f"**Incremental Embedding:** {'✅ Added' if feedback_result.get('embedding_added_incrementally') else '⏳ Will rebuild on next request'}\n\n"
                                "💡 This ticket is now available for future similar requests!"
                            )
                        else:
                            st.error(f"❌ Failed to save: {feedback_result.get('message', 'Unknown error')}")
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Error saving feedback: {str(e)}")
                        st.code(traceback.format_exc())

    # Raw JSON (optional)
    if show_raw_json:
        st.subheader("Raw Backend Output")
        st.code(json.dumps(result, indent=2)[:15000])

# ------------------------------ Footer ------------------------------------- #
st.markdown("---")
st.caption(
    "This UI is a thin layer over resolution_task."
)

