import streamlit as st
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration
from lime.lime_text import LimeTextExplainer

# Page Config
st.set_page_config(page_title="AI Legal Judgment", page_icon="⚖️", layout="wide")

# Custom CSS (Premium Look)
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00E5FF;
}
.subtitle {
    text-align: center;
    color: #B0BEC5;
    margin-bottom: 30px;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}
.result-box {
    background-color: #263238;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="main-title">⚖️ AI Legal Judgment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered classification, summarization & explanation</div>', unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer

bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer = load_models()

# Input Section
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
                num_features=4,
                num_samples=100
            )

        # Layout Output
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Prediction")
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

            st.markdown("### 📄 Summary")
            st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔍 Key Explanation")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(explanation.as_list())
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("💡 Built with Streamlit | AI Legal Tech Project")
