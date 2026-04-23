# FILE: app.py
"""
MoD-SLM Interactive Interface (Simplified)
-------------------------------------------
A clean, focused interface for interacting with the specialized MoD-SLM.
"""

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import time

# --- Page Config ---
st.set_page_config(
    page_title="MoD-MLM Interface",
    page_icon="ML",
    layout="wide",
)

# --- Basic Dark Theme Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0c1221;
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #060a14 !important;
        border-right: 1px solid #1c2e4a;
    }
    .stChatMessage {
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #1c2e4a;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #060a14 !important;
        border-left: 2px solid #555 !important;
    }
    .stChatInputContainer {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Model Loading (Cached) ---
@st.cache_resource
def load_mo_slm():
    model_dir = "models/hf_fine_tuned"
    import os
    if not os.path.exists(model_dir):
        st.error("Model weights not found. Ensure training is complete.")
        st.stop()
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model     = GPT2LMHeadModel.from_pretrained(model_dir)
    device    = 0 if torch.cuda.is_available() else -1
    
    gen_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=device
    )
    return gen_pipeline, tokenizer

# --- Sidebar ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### Quick Queries")
    if st.button("Role of the CDS"):
        st.session_state.messages.append({"role": "user", "content": "What is the role of the Chief of Defence Staff (CDS)?"})
    if st.button("Agnipath Scheme"):
        st.session_state.messages.append({"role": "user", "content": "Explain the Agnipath scheme."})
    if st.button("DAP 2020 Overview"):
        st.session_state.messages.append({"role": "user", "content": "What is the Defence Acquisition Procedure (DAP) 2020?"})
    
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.clear()
        st.rerun()

# --- Main Interface ---
st.title("MoD-MLM Specialized Knowledge Base")
st.caption("A micro language model specializing in Ministry of Defence (MoD) policy and procedural knowledge.")

# Load Model
with st.spinner("Loading Weights..."):
    generator, tokenizer = load_mo_slm()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am the MoD-MLM. How can I assist you with Ministry of Defence policy or procedural questions today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("Enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Consulting MoD Knowledge Base...")
        
        try:
            # Domain-Specific Constraint ("Fake RAG" Prompt Engineering)
            system_prompt = (
                "You are the MoD-MLM, a specialized assistant for the Indian Ministry of Defence. "
                "Your knowledge is strictly limited to MoD personnel rules, procurement (DAP 2020), "
                "and legal guidelines. Do NOT answer general knowledge questions outside this domain. "
                "If asked about anything else, politely state you are specialized ONLY for MoD data. "
                f"Question: {prompt} \nAnswer:"
            )

            output = generator(
                system_prompt,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            
            full_text = output[0]['generated_text']
            
            # Robust Response Extraction: Find the "Answer:" boundary
            if "Answer:" in full_text:
                response = full_text.split("Answer:")[1].strip()
            else:
                # Fallback to length-slicing if the model removed the anchor
                response = full_text[len(system_prompt):].strip()

            # If the model is completely blank, provide a fallback
            if not response:
                response = "I am specialized only for MoD data. Please rephrase your query related to personnel or procurement."
            
            # Simulated streaming
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Inference error: {e}")
