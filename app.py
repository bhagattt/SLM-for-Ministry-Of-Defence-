# FILE: app.py
"""
MoD-SLM Interactive Explorer (Streamlit Frontend)
-------------------------------------------------
A premium web interface for interacting with the fine-tuned MoD-SLM (GPT-2).
"""

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import time

# --- Page Config ---
st.set_page_config(
    page_title="MoD-SLM Stage 1 | Indian Defence AI",
    page_icon="🛡️",
    layout="wide",
)

# --- Premium Custom Styling (MoD / Indian Defence Aesthetic) ---
st.markdown("""
    <style>
    /* Dark Theme Accents */
    .stApp {
        background-color: #0c1221;
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #060a14 !important;
        border-right: 1px solid #1c2e4a;
    }
    
    /* Chat Aesthetics */
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #1c2e4a;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1c2e4a !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #060a14 !important;
        border-left: 4px solid #f9a825 !important; /* Gold MoD accent */
    }

    /* Header Styling */
    h1, h2, h3 {
        color: #f9a825 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Box */
    .stChatInputContainer {
        border-radius: 20px;
        border: 1px solid #f9a825 !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Model Loading (Cached) ---
@st.cache_resource
def load_mo_slm():
    model_dir = "./hf_mod_model"
    # Check if model exists
    import os
    if not os.path.exists(model_dir):
        st.error("Model not found! Run the training first: `python hf_train.py`")
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

# --- Initialize Sidebar Stats ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ee/Emblem_of_India.svg", width=80)
    st.title("🛡️ MoD-SLM Info")
    st.markdown("---")
    st.info("**Stage 1 Completed**")
    st.success("Target: Standard Laptop GPU")
    
    st.markdown("### Technical Pulse")
    st.metric("Parameters", "124M (GPT-2 FT)")
    st.metric("VRAM Usage", "~2.1 GB")
    st.metric("Training Status", "Converged")
    
    st.markdown("---")
    st.button("Reset Conversation", on_click=lambda: st.session_state.clear())

# --- Main Interaction Logic ---
st.title("🛡️ MoD-SLM Interactive Knowledge Base")
st.caption("Custom Fine-tuned SLM specializing in Indian Ministry of Defence Policies & General Knowledge.")

# Load Model
with st.spinner("Initializing MoD-SLM Brain..."):
    generator, tokenizer = load_mo_slm()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Jai Hind! I am the MoD-SLM. I'm trained on policy circulars, legal text, and audit reports of the Ministry of Defence. How can I assist you today?"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("Ask about MoD policies, Agnipath scheme, or hierarchy..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Consulting MoD Knowledge Base...")
        
        try:
            # Generate
            output = generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            
            full_text = output[0]['generated_text']
            # Simple response cleanup (strip prompt if repeated)
            response = full_text[len(prompt):].strip()
            
            # Simulate "streaming" effect for better UX
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error during inference: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Proprietary Interface for MoD SLM Presentation | Developed with Pure PyTorch & HuggingFace</p>", unsafe_allow_html=True)
