import os
import json
import subprocess
import time
import pandas as pd
import streamlit as st
from generator import generate_text, load_model_and_tokenizer
# Set streamlit page config
st.set_page_config(
    page_title="GPT-2 Fine-Tuning Arena",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load and inject custom CSS stylesheet
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("styles.css")
# Session state initialization
if "training_process" not in st.session_state:
    st.session_state.training_process = None
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None
# Sidebar Configuration
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #a78bfa; font-family: Outfit; font-weight: 700;'>🔮 CONFIGURATION</h2>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
# Pretrained model selection
st.sidebar.subheader("Select Base Model")
model_choice = st.sidebar.selectbox(
    "Base Model Type",
    ["distilgpt2", "gpt2"],
    index=0,
    help="DistilGPT-2 (82M parameters) is highly recommended for faster training on CPU. GPT-2 (124M parameters) is also available but takes longer."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Resources & Diagnostics")
st.sidebar.markdown(
    """
    <div style='background-color: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
        <p style='margin-bottom: 5px; font-size: 0.85rem;'><b>Environment:</b> CPU Mode Only</p>
        <p style='margin-bottom: 5px; font-size: 0.85rem;'><b>PyTorch:</b> v2.12.1+cpu</p>
        <p style='margin-bottom: 0px; font-size: 0.85rem;'><b>Checkpoint Path:</b> <code>checkpoints/latest</code></p>
    </div>
    """,
    unsafe_allow_html=True
)
# Main Page Header
st.markdown("<h1 class='main-title'>GPT-2 Fine-Tuning Arena</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Fine-tune and play with transformer language models directly in your browser</p>", unsafe_allow_html=True)
# Define Tabs
tab1, tab2, tab3 = st.tabs(["📊 Datasets & Exploration", "⚙️ Fine-Tuning Studio", "🎭 Generation Arena"])
# --- TAB 1: DATASETS & EXPLORATION ---
with tab1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Select or Upload Dataset</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        dataset_source = st.radio(
            "Dataset Source",
            ["Sample Datasets", "Upload Custom Text File"],
            index=0
        )
        
        dataset_filepath = None
        
        if dataset_source == "Sample Datasets":
            samples = {
                "Shakespeare's Sonnets (Classical English Poetry)": "data_files/shakespeare.txt",
                "Inspirational & Philosophical Quotes (Short sentences)": "data_files/quotes.txt",
                "Sci-Fi & Cyberpunk Descriptions (Atmospheric snippet list)": "data_files/scifi_descriptions.txt",
                "Detective Noir Stories (Mysterious & moody style)": "data_files/detective_noir.txt"
            }
            sample_choice = st.selectbox("Choose a sample dataset", list(samples.keys()))
            dataset_filepath = samples[sample_choice]
        else:
            uploaded_file = st.file_uploader("Upload custom text dataset (.txt)", type=["txt"])
            if uploaded_file is not None:
                os.makedirs("data_files", exist_ok=True)
                dataset_filepath = os.path.join("data_files", "uploaded_custom.txt")
                with open(dataset_filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Custom dataset uploaded successfully!")
                
    with col2:
        if dataset_filepath and os.path.exists(dataset_filepath):
            with open(dataset_filepath, "r", encoding="utf-8") as f:
                text_content = f.read()
            
            # Simple stats
            char_count = len(text_content)
            word_count = len(text_content.split())
            line_count = len(text_content.splitlines())
            est_tokens = int(char_count / 4) # rough GPT heuristic
            
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Dataset Size</div>
                    <div class='metric-value'>{char_count:,} characters</div>
                </div>
                <div class='metric-card' style='border-left-color: #a78bfa;'>
                    <div class='metric-title'>Word Count</div>
                    <div class='metric-value'>{word_count:,} words</div>
                </div>
                <div class='metric-card' style='border-left-color: #3b82f6;'>
                    <div class='metric-title'>Estimated Tokens</div>
                    <div class='metric-value'>~{est_tokens:,} tokens</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Please select or upload a dataset to view statistics.")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    if dataset_filepath and os.path.exists(dataset_filepath):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Dataset Preview</h3>", unsafe_allow_html=True)
        st.text_area("File Contents Preview", text_content[:2000] + ("..." if len(text_content) > 2000 else ""), height=300, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
# --- TAB 2: FINE-TUNING STUDIO ---
with tab2:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Fine-Tuning Configuration</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Training Epochs", min_value=1, max_value=20, value=3, help="Number of complete passes through the dataset. Keep it small on CPU.")
        batch_size = st.selectbox("Batch Size", [1, 2, 4], index=1, help="Size of batches. CPU training runs best with batch size 1 or 2.")
        lr = st.select_slider("Learning Rate", options=[1e-5, 3e-5, 5e-5, 1e-4, 2e-4], value=5e-5, help="Controls how fast the model updates weights. Default (5e-5) is recommended.")
        
    with col2:
        max_steps = st.number_input("Max Training Steps (-1 for all)", min_value=-1, max_value=1000, value=-1, help="If > 0, halts training after this many steps, overriding Epochs. Highly useful for quick CPU tests.")
        block_size = st.selectbox("Block Size (Tokens)", [64, 128, 256], index=1, help="Context length per training block. Shorter is faster to compute.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process management and UI rendering
    if st.session_state.training_process is not None:
        poll_status = st.session_state.training_process.poll()
        
        if poll_status is None:
            # Training is currently running
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h3>⚡ Training in Progress</h3>", unsafe_allow_html=True)
            
            # Read progress.json
            progress_data = {}
            if os.path.exists("progress.json"):
                try:
                    with open("progress.json", "r") as f:
                        progress_data = json.load(f)
                except Exception:
                    pass
            
            status = progress_data.get("status", "starting")
            pct = progress_data.get("percentage", 0.0)
            step = progress_data.get("step", 0)
            max_steps_data = progress_data.get("max_steps", 100)
            loss = progress_data.get("loss", None)
            epoch = progress_data.get("epoch", 0.0)
            
            # Render status
            if status == "loading_model":
                st.info("Downloading/loading pre-trained model and tokenizer to memory (first run will download files)...")
                st.spinner("Model initialization...")
            elif status == "training":
                st.progress(min(max(pct / 100.0, 0.0), 1.0))
                
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-card'><div class='metric-title'>Step</div><div class='metric-value'>{step} / {max_steps_data}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card' style='border-left-color: #a78bfa;'><div class='metric-title'>Current Loss</div><div class='metric-value'>{f'{loss:.4f}' if loss is not None else 'Evaluating...'}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card' style='border-left-color: #3b82f6;'><div class='metric-title'>Epoch Progress</div><div class='metric-value'>{epoch:.2f}</div></div>", unsafe_allow_html=True)
                
            # Plot Loss Curve
            if os.path.exists("progress_history.json"):
                try:
                    with open("progress_history.json", "r") as f:
                        history = json.load(f)
                    if len(history) > 0:
                        df_history = pd.DataFrame(history)
                        st.markdown("<h4>Training Loss Curve</h4>", unsafe_allow_html=True)
                        st.line_chart(df_history.set_index("step")["loss"])
                except Exception:
                    pass
                    
            if st.button("Cancel Training"):
                st.session_state.training_process.terminate()
                st.session_state.training_process = None
                # Mark progress as failed
                with open("progress.json", "w") as f:
                    json.dump({"status": "failed", "error": "User cancelled the training execution."}, f)
                st.warning("Training process terminated.")
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Auto-reload page every 2 seconds to poll training updates
            time.sleep(2.0)
            st.rerun()
            
        else:
            # Training finished while user was away or on page refresh
            if poll_status == 0:
                st.success("🎉 Fine-Tuning Completed Successfully! Model checkpoint saved in checkpoints/latest")
                st.cache_resource.clear() # clear loaded model caches
            else:
                st.error("❌ Fine-tuning failed.")
                if os.path.exists("progress.json"):
                    try:
                        with open("progress.json", "r") as f:
                            err_data = json.load(f)
                        if "error" in err_data:
                            st.code(err_data["error"], language="bash")
                    except Exception:
                        pass
            
            st.session_state.training_process = None
            st.rerun()
            
    else:
        # Check if previous run exists
        if os.path.exists("progress.json"):
            try:
                with open("progress.json", "r") as f:
                    last_run = json.load(f)
                status = last_run.get("status", "")
                if status == "completed":
                    st.success("Last fine-tuning run completed successfully! Ready to generate text in the Generation Arena.")
                elif status == "failed":
                    st.error(f"Last training run failed. Error: {last_run.get('error', 'Unknown error')}")
            except Exception:
                pass
                
        # Start button
        if dataset_filepath and os.path.exists(dataset_filepath):
            if st.button("Start Fine-Tuning 🚀"):
                # Clean up previous logs
                if os.path.exists("progress.json"):
                    try:
                        os.remove("progress.json")
                    except Exception:
                        pass
                if os.path.exists("progress_history.json"):
                    try:
                        os.remove("progress_history.json")
                    except Exception:
                        pass
                
                # Setup parameters
                cmd = [
                    "python", "trainer.py",
                    "--model", model_choice,
                    "--dataset", dataset_filepath,
                    "--output_dir", "./checkpoints/latest",
                    "--epochs", str(epochs),
                    "--batch_size", str(batch_size),
                    "--lr", str(lr),
                    "--max_steps", str(max_steps),
                    "--block_size", str(block_size),
                    "--progress_file", "progress.json"
                ]
                
                st.session_state.training_process = subprocess.Popen(cmd)
                st.info("Spawning background training process...")
                st.rerun()
        else:
            st.warning("Please upload or select a dataset in Tab 1 first before running training.")
# --- TAB 3: GENERATION ARENA ---
# Cached model loaders
@st.cache_resource
def load_base_cached(name):
    return load_model_and_tokenizer(name)
@st.cache_resource
def load_finetuned_cached(path):
    return load_model_and_tokenizer(path)
with tab3:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Text Generation Settings</h3>", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05, help="Higher values increase creativity/randomness, lower values make output more focused.")
        max_length = st.slider("Max Generated Tokens", min_value=10, max_value=250, value=50, step=5, help="Maximum length of newly generated tokens.")
        
    with col_p2:
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=50, help="Locks generation to top K likely tokens at each step.")
        top_p = st.slider("Top-P (Nucleus Sampling)", min_value=0.0, max_value=1.0, value=0.9, step=0.05, help="Locks generation to cumulative probability threshold.")
        
    with col_p3:
        repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05, help="Discourages the model from repeating words or phrases.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Check if a fine-tuned model exists
    ft_model_exists = os.path.exists("./checkpoints/latest/config.json")
    
    st.markdown("<div class='prompt-container'>", unsafe_allow_html=True)
    prompt = st.text_area("Enter a Prompt to Start Generation", value="Deep in the dark forest, a mystery was waiting", height=80)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Generate Continuation 🔮", key="generate_btn"):
        if not prompt.strip():
            st.error("Please enter a valid text prompt first!")
        else:
            col_b, col_f = st.columns(2)
            
            # Base Model Generation
            with col_b:
                st.markdown("<h4>Base Model Output</h4>", unsafe_allow_html=True)
                with st.spinner("Base model computing..."):
                    try:
                        base_model, base_tokenizer = load_base_cached(model_choice)
                        base_output = generate_text(
                            model=base_model,
                            tokenizer=base_tokenizer,
                            prompt=prompt,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty
                        )[0]
                        st.markdown(f"<div class='generated-text-box-base'>{base_output}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating base output: {str(e)}")
                        
            # Fine-Tuned Model Generation
            with col_f:
                st.markdown("<h4>Fine-Tuned Model Output</h4>", unsafe_allow_html=True)
                if ft_model_exists:
                    with st.spinner("Fine-tuned model computing..."):
                        try:
                            ft_model, ft_tokenizer = load_finetuned_cached("./checkpoints/latest")
                            ft_output = generate_text(
                                model=ft_model,
                                tokenizer=ft_tokenizer,
                                prompt=prompt,
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty
                            )[0]
                            st.markdown(f"<div class='generated-text-box'>{ft_output}</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating fine-tuned output: {str(e)}")
                else:
                    st.warning("⚠️ No fine-tuned model checkpoint found. Go to the Fine-Tuning Studio tab to train a model first, then trigger generation to compare!")