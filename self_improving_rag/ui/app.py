"""
Streamlit User Interface for the Self-Improving RAG system.

Provides:
- Document ingestion (sidebar uploader)
- Interactive Chat with citations
- Feedback collection (thumbs up/down)
- System status and experiment tracking overview
"""

import os
import sys
import uuid
import asyncio
import streamlit as st
from pathlib import Path

# Add root folder to path to allow imports
sys.path.append(str(Path(__file__).parents[2]))

from self_improving_rag.retrieval.ingest import ingest_file
from self_improving_rag.core.pipeline import run_pipeline
from self_improving_rag.feedback.capture import log_feedback
from self_improving_rag.feedback.schema import SignalType
from self_improving_rag.training.experiment_tracker import get_latest_experiments
from self_improving_rag.storage.queries import get_feedback_count

# Page config
st.set_page_config(
    page_title="Self-Improving RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def run_async(coro):
    """Helper to run async code in Streamlit without event loop collisions."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# ──────────────────────────────────────────────
# Sidebar: Ingestion & System Status
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ RAG Management")
    
    st.subheader("📁 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Ingest"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a temp dir for uploads
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_text.text(f"Ingesting {uploaded_file.name}...")
                count = ingest_file(str(temp_path))
                
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
            
            st.success(f"Ingested {len(uploaded_files)} files!")
        else:
            st.warning("Please upload files first.")

    st.divider()
    
    st.subheader("📊 System Stats")
    feedback_count = run_async(get_feedback_count())
    st.metric("Feedback Collected", feedback_count)
    
    if st.button("📉 View Latest Experiments"):
        runs = run_async(get_latest_experiments(5))
        if runs:
            st.table(runs)
        else:
            st.info("No experiment runs found.")

# ──────────────────────────────────────────────
# Main UI: Chat
# ──────────────────────────────────────────────
st.title("💬 Self-Improving RAG")
st.markdown("""
Ask questions about your uploaded documents. 
Your feedback (👍/👎) helps the system fine-tune its reranker!
""")

# Display Chat History
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Sources/Citations if available
        if message["role"] == "assistant" and "cited_chunks" in message:
            with st.expander("References"):
                for chunk in message["cited_chunks"]:
                    st.write(f"**[{chunk.get('display_id', 'Source')}]** from *{chunk.get('source_doc', 'Unknown')}*")
                    st.caption(chunk.get("text", "")[:200] + "...")

        # Feedback Buttons (only for assistant messages)
        if message["role"] == "assistant":
            col1, col2, _ = st.columns([0.1, 0.1, 0.8])
            # We use keys based on session_id and message index to persist feedback
            with col1:
                if st.button("👍", key=f"up_{msg_idx}"):
                    # Log feedback for all cited chunks
                    for chunk in message.get("cited_chunks", []):
                        run_async(log_feedback(
                            message["session_id"], 
                            chunk["chunk_id"], 
                            SignalType.THUMBS_UP, 
                            1.0
                        ))
                    st.toast("Thanks for the positive feedback!")
            with col2:
                if st.button("👎", key=f"down_{msg_idx}"):
                    for chunk in message.get("cited_chunks", []):
                        run_async(log_feedback(
                            message["session_id"], 
                            chunk["chunk_id"], 
                            SignalType.THUMBS_DOWN, 
                            1.0
                        ))
                    st.toast("Feedback recorded. Reranker will learn from this.")

# Chat Input
if prompt := st.chat_input("What is the main topic of these documents?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Searching..."):
            try:
                response = run_async(run_pipeline(prompt, session_id=st.session_state.session_id))
                
                st.markdown(response.answer)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.answer,
                    "cited_chunks": response.cited_chunks,
                    "session_id": response.session_id
                })
                
                st.rerun() # Refresh to show references and feedback buttons

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Failed to generate response: {e}"})
