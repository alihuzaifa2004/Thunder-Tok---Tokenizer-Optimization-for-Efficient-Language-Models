import os
import torch
import streamlit as st

# 1. IMMEDIATE FIX FOR PYTHON 3.13 & STREAMLIT
torch.classes.__path__ = []
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# 2. LOCAL IMPORTS
from tokenizer.vocab_builder import build_vocab
from tokenizer.thunder_tok import ThunderTok
from utils.fertility import fertility

# CORRECT IMPORT BASED ON YOUR FOLDER NAME
from llm.llama_wrapper import LLaMAWrapper

# 3. STREAMLIT MODEL CACHE
@st.cache_resource
def load_tiny_llama(vocab_size):
    return LLaMAWrapper(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vocab_size=vocab_size,
        token=None
    )

# 4. STREAMLIT UI
st.set_page_config(page_title="Thunder-Tok + TinyLlama", layout="wide")
st.title("⚡ Thunder-Tok integrated with TinyLlama")

text = st.text_area(
    "Enter input text",
    height=150,
    value="Hello, how does this tokenizer work?"
)

if st.button("Run Thunder-Tok + LLaMA"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    corpus_path = "data/corpus.txt"
    if not os.path.exists(corpus_path):
        st.error(f"File not found: {corpus_path}")
        st.stop()

    # 5. BUILD VOCAB & TOKENIZE
    corpus = open(corpus_path, encoding="utf-8").read().splitlines()
    vocab, probs = build_vocab(corpus, vocab_size=2000)

    tokenizer = ThunderTok(vocab, probs)
    tokens = tokenizer.tokenize(text)

    stoi = {t: i for i, t in enumerate(vocab)}
    input_ids = torch.tensor([[stoi.get(t, 0) for t in tokens]])

    # 6. MODEL GENERATION
    with st.spinner("Loading model & generating output..."):
        llama = load_tiny_llama(len(vocab))
        output_ids = llama.generate(input_ids)

    # 7. SAFE DECODING (NO ERRORS)
    output_ids_list = output_ids.tolist()[0]  # extract batch

    decoded_tokens = [
        vocab[i] if isinstance(i, int) and i < len(vocab) else f"[{i}]"
        for i in output_ids_list
    ]

    output_text = "".join(decoded_tokens).strip()

    # 8. DISPLAY RESULTS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Thunder-Tok Tokens")
        st.write(tokens)

    with col2:
        st.subheader("🔹 Fertility Score")
        st.metric("Score", round(fertility(tokens, text), 3))

    st.subheader("🔹 TinyLlama Output (Experimental)")
    st.success(output_text)
