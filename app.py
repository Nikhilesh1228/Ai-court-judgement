import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration
from lime.lime_text import LimeTextExplainer
import uvicorn

# Instantiate FastAPI application
app = FastAPI()

# Global variables for models and tokenizers
bert_tokenizer = None
bert_model = None
t5_tokenizer = None
t5_model = None
explainer = None

# Define a request body schema
class JudgmentText(BaseModel):
    judgment_text: str

# Function to load models and tokenizers
def load_models():
    global bert_tokenizer, bert_model, t5_tokenizer, t5_model, explainer

    # Load BERT tokenizer and model
    bert_tokenizer = AutoTokenizer.from_pretrained("legal_bert_model")
    bert_model = BertForSequenceClassification.from_pretrained("legal_bert_model")
    bert_model.eval()  # Set BERT model to evaluation mode

    # Load T5 tokenizer and model
    t5_tokenizer = AutoTokenizer.from_pretrained("t5_summarization_model") # Changed to AutoTokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained("t5_summarization_model")
    t5_model.eval() # Set T5 model to evaluation mode

    # Instantiate LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])

    print("Models, tokenizers, and explainer loaded successfully!")

# Define the prediction function for LIME, using the BERT model
def predictor(texts):
    # Tokenize the input texts using the BERT tokenizer
    inputs = bert_tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    bert_model.to(device) # Ensure model is on the correct device

    # Get model outputs (logits) from the BERT model
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Convert probabilities to a NumPy array
    return probabilities.cpu().numpy()

# FastAPI startup event to load models
@app.on_event("startup")
async def startup_event():
    load_models()

# Define the prediction endpoint
@app.post("/predict")
async def predict_judgment(item: JudgmentText):
    # Classification using BERT
    bert_inputs = bert_tokenizer(item.judgment_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_inputs = {k: v.to(device) for k, v in bert_inputs.items()}

    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
    bert_pred_idx = torch.argmax(bert_outputs.logits).item()
    bert_prediction = 'Accepted' if bert_pred_idx == 1 else 'Rejected'

    # Summarization using T5
    t5_input_text = "summarize: " + item.judgment_text
    t5_inputs = t5_tokenizer.encode(t5_input_text, return_tensors="pt", truncation=True)
    t5_inputs = t5_inputs.to(device)

    t5_summary_ids = t5_model.generate(
        t5_inputs,
        max_length=120,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )
    t5_summary = t5_tokenizer.decode(t5_summary_ids[0], skip_special_tokens=True)

    # LIME Explanation
    explanation = explainer.explain_instance(
        item.judgment_text,
        predictor,
        num_features=6, # Number of features to highlight
        num_samples=1000 # Number of perturbations
    )
    lime_explanation = explanation.as_list()

    return {
        "classification": bert_prediction,
        "summary": t5_summary,
        "lime_explanation": lime_explanation
    }

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
