#  FinClassify AI

FinClassify AI is a machine-learning powered system that automatically classifies raw bank transaction text (e.g., **"DMART GROCERY"**, **"UBER *TRIP 8877"**) into spending categories such as Groceries, Shopping, Dining, Fuel, Entertainment, and Transport.

It helps users track expenses effortlessly and enables financial apps to automate categorisation.

---

## Features

✅ Classifies raw transaction text  
✅ Shows confidence score  
✅ Visual “Class Scores” bar chart  
✅ Simple explainability (XAI)  
✅ Feedback logging for continuous learning  
✅ Clean Streamlit UI  

---

##  Tech Stack

- **Python**
- **DistilBERT embeddings**
- **Logistic Regression classifier**
- **Streamlit UI**
- **Local processing for privacy**

---

##  Project Structure
FinClassify_AI/
├── data/
│ └── transactions.csv
├── models/
│ ├── finclassify_model.pkl
│ └── finclassify_labels.pkl
├── src/
│ ├── data_prep.py
│ ├── train_model.py
│ ├── inference_service.py
│ └── streamlit_app.py
├── README.md
└── venv/


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






