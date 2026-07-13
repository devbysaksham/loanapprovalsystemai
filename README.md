# 🏦 AI-Based Loan Approval System

## 📌 Project Description
This project is an advanced, production-ready **Machine Learning application** that predicts whether a customer's loan application should be Approved or Rejected based on their demographic and financial information. It is designed and structured as a comprehensive college internship project, demonstrating the complete Machine Learning lifecycle.

## ✨ Features
* **Machine Learning Pipeline**: Complete pipeline including missing value handling, feature engineering, label encoding, and model serialization.
* **Streamlit Glassmorphism UI**: A beautiful, modern web application utilizing glassmorphism cards, responsive CSS, smooth animations, and sidebar navigation.
* **Advanced EDA Notebook**: A highly detailed Jupyter notebook containing comprehensive exploratory data analysis, correlation heatmaps, feature importance, ROC curves, and multi-model evaluations.
* **Modular Architecture**: Separate modules for training (`train_model.py`) and predictions (`predict.py`) for clean, reusable code.

## 🛠 Technology Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Serialization:** Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit

## 📂 Folder Structure
```text
LoanApprovalSystem/
│
├── datasets/
│   └── train_u6lujuX_CVtuZ9i.csv (Kaggle dataset)
│
├── models/
│   ├── loan_model.pkl (Trained ML model)
│   └── encoder.pkl    (Saved categorical encoders)
│
├── notebook/
│   └── Loan_Approval_System.ipynb (Full EDA and evaluation notebook)
│
├── app.py             (Streamlit web application interface)
├── train_model.py     (Automated model training pipeline)
├── predict.py         (Reusable prediction function script)
├── requirements.txt   (Project dependencies)
├── README.md          
├── LICENSE            
├── screenshots/       (Images of the web app)
└── assets/            (Static assets, icons, etc.)
```

## ⚙️ Installation & Running Locally

1. **Clone the repository** (if downloaded from GitHub) or navigate to the project directory:
   ```bash
   cd LoanApprovalSystem
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Train the model** to generate `loan_model.pkl` and `encoder.pkl`:
   ```bash
   python train_model.py
   ```
4. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```
   This will open the app locally at `http://localhost:8501`.

## 🚀 Deployment
This project is fully ready for deployment on **Streamlit Community Cloud** or **GitHub Pages**. Because it utilizes a separated `predict.py` and saved artifact folder (`models/`), it requires absolutely no code changes to run online. Simply connect the repository to Streamlit Cloud and set `app.py` as the entrypoint.

## 🖼 Screenshots
*(You can store your screenshots in the `screenshots/` directory)*
- Add an image of the Home Page, the Glassmorphism cards, and the Success/Rejection animations.

---
*Developed for College Internship Submission.*
