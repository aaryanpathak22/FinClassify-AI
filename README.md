#  FinClassify-AI – Smart Transaction Categorization
FinClassify-AI is an NLP-based machine learning system designed to automatically categorize raw financial transaction descriptions into structured spending categories.

The project implements a complete ML pipeline including data preprocessing, model training, inference service, and a Streamlit-based user interface for real-time predictions.

---

## Features

- Text normalization and preprocessing pipeline
- Abbreviation expansion for financial transactions
- Supervised ML model training
- Saved model serialization (.pkl)
- Real-time inference service
- Streamlit UI for interactive predictions
- Structured modular project architecture
---

##  Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Pickle (model serialization)**
- **DistilBERT embeddings**
- **Logistic Regression classifier**
- **Streamlit UI**
- **Local processing for privacy**

---

##  Project Structure
<pre> ## Project Structure ``` FinClassify_AI/ │ ├── data/ │ └── transactions.csv │ ├── models/ │ ├── finclassify_model.pkl │ └── finclassify_labels.pkl │ ├── src/ │ ├── data_prep.py │ ├── train_model.py │ └── inference_service.py │ ├── streamlit_app.py ├── README.md └── venv/ ``` </pre>

---

## ⚙️ How It Works

1. Raw transaction text is cleaned and normalized.
2. Abbreviations are expanded using a custom mapping.
3. Text is vectorized using NLP techniques.
4. A supervised ML classifier predicts transaction categories.
5. Predictions are served via a Streamlit web interface.


---

##  How to Run the App

### 1️⃣ Create Virtual Environment
python -m venv venv


### 2️⃣ Activate
**Windows**
venv\Scripts\activate


-> then cd src


### 3️⃣ Install Dependencies
pip install -r requirements.txt


### 4️⃣ Run Streamlit App
python -m streamlit run streamlit_app.py



## ✅ Future Enhancements
- Deployment as REST API
- Cloud hosting (AWS/GCP/Azure)
- Real-time classification
- Bigger multi-category dataset
- Fraud / anomaly detection

---






