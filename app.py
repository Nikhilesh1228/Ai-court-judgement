import streamlit as st
import torch
import re
from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration
from lime.lime_text import LimeTextExplainer

# Page config
st.set_page_config(page_title="AI Legal Judgment", page_icon="⚖️", layout="wide")

# 🔥 ULTRA PREMIUM CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    background: -webkit-linear-gradient(#00E5FF, #00FFAA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Input */
textarea {
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00E5FF, #00FFAA);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
}

/* Prediction */
.pred {
    background: #00C853;
    color: black;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}

/* Summary */
.summary {
    background: #1565C0;
    padding: 15px;
    border-radius: 10px;
}

/* Chips */
.chip {
    display: inline-block;
    padding: 6px 12px;
    margin: 6px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: bold;
}

.pos { background-color: #00C853; color: black; }
.neg { background-color: #D50000; color: white; }

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">⚖️ AI Legal Judgment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered classification • summarization • explanation</div>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer

bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer = load_models()

# Input
st.markdown("### 📝 Enter Judgment Text")
text = st.text_area("", height=180, placeholder="Paste legal judgment text here...")

# Button
if st.button("🚀 Analyze", use_container_width=True):

    if text.strip() == "":
        st.warning("⚠️ Please enter text")
    else:
        with st.spinner("Analyzing... ⏳"):

            # Prediction
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)

            pred = torch.argmax(outputs.logits).item()
            result = "✅ Accepted" if pred == 1 else "❌ Rejected"

            # Summary
            t5_input = "summarize: " + text
            t5_inputs = t5_tokenizer.encode(t5_input, return_tensors="pt", truncation=True)
            summary_ids = t5_model.generate(t5_inputs, max_length=60)
            summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # LIME
            def predictor(texts):
                inputs = bert_tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                return probs.numpy()

            explanation = explainer.explain_instance(
                text,
                predictor,
                num_features=6,
                num_samples=100
            )

            # 🔥 Filter stopwords
            stopwords = {"the", "is", "and", "a", "an", "of", "to", "in", "on", "for"}
            filtered_explanation = []

            for word, score in explanation.as_list():
                clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())

                if clean_word not in stopwords and len(clean_word) > 2:
                    filtered_explanation.append((clean_word, score))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Prediction")
            st.markdown(f'<div class="pred">{result}</div>', unsafe_allow_html=True)

            st.markdown("### 📄 Summary")
            st.markdown(f'<div class="summary">{summary}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔍 Key Legal Factors")

            chips_html = ""

            for word, score in filtered_explanation:
                cls = "pos" if score > 0 else "neg"
                chips_html += f'<span class="chip {cls}">{word}</span> '

            st.markdown(f'<div class="card">{chips_html}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("💼 Built by Nikhilesh | AI Legal Tech Project")
