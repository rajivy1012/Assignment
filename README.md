
## 📌 Overview

This repository contains solutions to both **Task 1** and **Task 2** as part of the AI Developer Internship assignment for Vijayi WFH Technologies Pvt Ltd.

- ✅ **Task 1:** Classical ML Pipeline for Support Ticket Classification & Entity Extraction
- ✅ **Task 2:** RAG-Based Semantic Quote Retrieval & QA with Model Fine-Tuning + Streamlit App

## 📁 Directory Structure

```
├── Vijayi_WFH.ipynb                               # Jupyter Notebook with complete implementation
├── data/
│   └── ai_dev_assignment_tickets_complex_1000.xlsx # Input dataset for Task 1
├── models/
│   ├── ticket_classifiers/                        # Trained classical ML models (pickle)
│   └── sentence_transformer/                      # Fine-tuned sentence embedding model
├── streamlit_app.py                               # Streamlit app for Task 2
├── gradio_app.py                                  # Gradio app for Task 1 (optional)
├── requirements.txt                               # Python dependencies
├── README.md                                      # This file
└── demo_video_link.txt                           # Google Drive link to screen recording demo
```

## 🧠 Task 1: Customer Support Ticket Pipeline

### 🎯 Objective
Classify customer support tickets by:
- **Issue Type** (multi-class classification)
- **Urgency Level** (Low, Medium, High)

Additionally, extract entities from tickets:
- Product names
- Dates
- Complaint-related keywords

### 📊 Models Used
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**

### 🔍 Entity Extraction Techniques
- Regex-based product name extraction
- Date patterns using regex
- Keyword lists for complaints (e.g., "broken", "late", "error")

### 📈 Evaluation Metrics
- Accuracy
- F1-Score
- Confusion Matrix
- Classification Report

### 🌐 Gradio Interface (Optional Bonus)
Simple UI to test predictions and extractions interactively.

## 🔍 Task 2: RAG-Based Quote Retrieval System

### 🎯 Objective
Retrieve relevant quotes based on semantic queries using:
- **Fine-Tuned Sentence Embedding Model**
- **FAISS for Vector Indexing**
- **RAG Pipeline for QA**

### 📚 Dataset Used
- `Abirate/english_quotes` from Hugging Face

### 🧠 Model Architecture
- Fine-tuned `sentence-transformers` model on quotes and authors
- Vector similarity search using FAISS
- Integration with language model for enhanced responses

### 🧰 RAG Pipeline
- Uses vector similarity via FAISS for retrieval
- Top matches passed to LLM (GPT or Llama2)
- Returns structured JSON with quote, author, and tags

### ✅ RAG Evaluation
Performed using **RAGAS** framework:
- Precision and recall metrics
- Hallucination detection scores
- Context relevance evaluation

### 💻 Streamlit App
Interactive web UI featuring:
- Input semantic queries (e.g., *"quotes on insanity by Einstein"*)
- Structured outputs: quote, author, tags, and similarity score
- Real-time query processing

## 🎥 Demo Video

A comprehensive walkthrough including:
- Code explanation and methodology
- Jupyter notebook results and analysis
- Live demonstration of Gradio and Streamlit applications

**📎 Demo Video Link:** [Available in demo_video_link.txt]


## ⚙️ How to Run

### ✅ Prerequisites

```bash
pip install -r requirements.txt
```

### 🧪 Run Jupyter Notebook
```bash
jupyter notebook Vijayi_WFH.ipynb
```
*Alternatively, open in Google Colab for cloud execution.*

### 🌐 Run Gradio App (Task 1)
```bash
python gradio_app.py
```

### 🌐 Run Streamlit App (Task 2)
```bash
streamlit run streamlit_app.py
```

## 📦 Dependencies

Key libraries used:
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `streamlit` - Web app framework
- `gradio` - Interactive ML interfaces
- `ragas` - RAG evaluation framework

## ⚠️ Important Notes

- All code was developed independently without LLM code generation
- Only conceptual guidance was sought from AI tools
- Models trained using classical ML algorithms and open-source transformers
- Comprehensive evaluation performed using scikit-learn metrics and RAGAS
- All da
