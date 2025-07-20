# Student Exam Performance Prediction 🎓

**End-to-end Machine Learning project** to predict students’ exam performance using Python. Includes data processing, model training & evaluation, and a Flask + React frontend (thanks to my collaborator!) for realtime predictions.

---

## 🧠 Overview

A complete ML pipeline that:
1. Ingests and preprocesses student data
2. Trains models to predict exam success (classification or regression)
3. Evaluates using multiple metrics & visualizations
4. Exposes a web frontend for interactive predictions

---

## 🔍 Data & Setup

- **Dataset**: Student exam performance CSV (features: demographic info, study habits, test scores)
- **Preprocessing**:
  - Categorical → Numeric mapping
  - Feature scaling (standardization)
  - Train/test split
- **Visualization**: Histograms, boxplots, correlation analysis

---

## ⚙️ Model Training

Contains implementations of several algorithms:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

Each model is:
- Tuned via grid search / parameter sweeps
- Evaluated using:
  - Confusion matrix
  - F1-score
  - ROC curve and AUC

The best-performing model is saved for deployment.

---

## 🚀 Backend & Frontend

- **Backend**: Flask API  
  - Loads saved model & preprocessor
  - Receives POST requests with student data → returns predictions
- **Frontend**: Built by my friend  
  - React interface for inputting student info
  - Displays predicted outcome dynamically

---

## 🧩 Project Structure

- ML_Project_1/
 - ├── data/ # Raw & processed CSVs
 - ├── notebooks/ # EDA and model-training notebooks
 - ├── src/ # Python scripts for pipeline components
 - ├── backend/ # Flask app
 - ├── frontend/ # React UI
 - ├── models/ # Trained model & preprocessor pickle files
 - ├── requirements.txt # Python dependencies
 - └── README.md # This file



---

## 🔧 Installation & Usage

1. **Clone the repo**
   ```bash
   git clone https://github.com/Arshp-svg/ML_Project_1.git
   cd ML_Project_1


2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Run EDA & Train models**
   Launch notebooks in notebooks/
   Scripts in src/ can also be executed directly
4.**Start the API**
    ```bash
    cd backend
    flask run
5.**Run the frontend**
    ```bash
    
    cd frontend
    npm install
    npm start
6.**Make Predictions**
  - Open the web app (usually http://localhost:3000)
  - Enter student details → get performance prediction!
  
# 👥 Collaboration
Backend & ML pipeline: Arshp‑svg

Frontend:https://github.com/shreyash2246/

# 📬 Feedback & Contact
 - Questions, issues or contributions are very welcome! Feel free to open issues or     submit a pull request.


