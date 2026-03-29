import streamlit as st
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration
from lime.lime_text import LimeTextExplainer

# Page config
st.set_page_config(page_title="AI Legal Judgment", page_icon="⚖️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #00FFAA;
    }
    .sub {
        text-align: center;
        color: #AAAAAA;
        margin-bottom: 20px;
    }
    .box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">⚖️ AI Legal Judgment System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Analyze legal text with AI (Classification + Summary + Explanation)</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    bert_model.eval()
    t5_model.eval()

    explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer

bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer = load_models()

# Input Section
st.markdown("### 📝 Enter Judgment Text")
text = st.text_area("", height=200)

# Button
if st.button("🚀 Analyze", use_container_width=True):

    with st.spinner("Processing... ⏳"):

        # Prediction
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)

        pred = torch.argmax(outputs.logits).item()
        result = "✅ Accepted" if pred == 1 else "❌ Rejected"

        # Summary
        t5_input = "summarize: " + text
        t5_inputs = t5_tokenizer.encode(t5_input, return_tensors="pt", truncation=True)
        summary_ids = t5_model.generate(t5_inputs, max_length=100)
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # LIME
        def predictor(texts):
            inputs = bert_tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            return probs.numpy()

        explanation = explainer.explain_instance(text, predictor, num_features=6)

    # Output Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Prediction")
        st.success(result)

        st.markdown("### 📄 Summary")
        st.info(summary)

    with col2:
        st.markdown("### 🔍 Explanation")
        st.write(explanation.as_list())
