import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    cells = []
    
    # Helper to add markdown
    def add_md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
        
    # Helper to add code
    def add_code(text):
        cells.append(nbf.v4.new_code_cell(text))

    add_md("# 🏦 AI-Based Loan Approval System\n\nThis notebook demonstrates a complete end-to-end Machine Learning classification project to predict loan approvals. It is organized into 20 strict sections.")

    add_md("## 1. Import Libraries")
    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import joblib

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")""")

    add_md("## 2. Load Dataset")
    add_code("df = pd.read_csv('../datasets/train_u6lujuX_CVtuZ9i.csv')\ndf.head()")

    add_md("## 3. Display Dataset Information")
    add_code("df.info()")

    add_md("## 4. Check Dataset Shape")
    add_code("print('Dataset Shape:', df.shape)")

    add_md("## 5. Check Data Types")
    add_code("df.dtypes")

    add_md("## 6. Check Missing Values")
    add_code("df.isnull().sum()")

    add_md("## 7. Handle Missing Values")
    add_code("""# Categorical: Fill with Mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
# Numerical: Fill with Median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

print("Missing values after handling:")
print(df.isnull().sum())""")

    add_md("## 8. Remove Duplicates")
    add_code("print('Duplicates before:', df.duplicated().sum())\ndf.drop_duplicates(inplace=True)")

    add_md("## 9. Encode Categorical Variables\nWe encode variables and save the encoders for later use.")
    add_code("""encoders = {}
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    
df.head()""")

    add_md("## 10. Exploratory Data Analysis\nVisualizing relationships and distributions.")
    add_code("""# Histogram & Distribution Plot
plt.figure(figsize=(10,5))
sns.histplot(df['ApplicantIncome'], bins=30, kde=True, color='blue')
plt.title('Applicant Income Distribution')
plt.show()

# Count Plot
plt.figure(figsize=(8,5))
sns.countplot(x='Loan_Status', data=df, palette='Set2')
plt.title('Loan Status Count')
plt.show()

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
plt.title('Applicant Income vs Loan Status')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()""")

    add_md("## 11. Feature Engineering\nDropping columns like Loan_ID that do not provide predictive value.")
    add_code("""if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)
print("Columns ready for training:", df.columns.tolist())""")

    add_md("## 12. Train Test Split")
    add_code("""X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)""")

    add_md("## 13. Train Multiple Classification Models")
    add_code("""models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully.")""")

    add_md("## 14. Compare Model Performance")
    add_code("""performance = {}
for name, model in trained_models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    performance[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")""")

    add_md("## 15. Select Best Model")
    add_code("""best_model_name = max(performance, key=performance.get)
print(f"🥇 Best Model Selected: {best_model_name}")
best_model = trained_models[best_model_name]""")

    add_md("## 16. Evaluate Model")
    add_code("""y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Feature Importance (if Random Forest or Decision Tree)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances, y=X.columns)
    plt.title('Feature Importance')
    plt.show()
""")

    add_md("## 17. Save Final Model using Joblib")
    add_code("""import os
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/loan_model.pkl')
print("Model saved to '../models/loan_model.pkl'")""")

    add_md("## 18. Save Encoder")
    add_code("""joblib.dump(encoders, '../models/encoder.pkl')
print("Encoders saved to '../models/encoder.pkl'")""")

    add_md("## 19. Show Prediction Examples")
    add_code("""sample = X_test.iloc[[0]]
pred = best_model.predict(sample)
prob = best_model.predict_proba(sample)[0]

print("Sample Input:")
print(sample)
print(f"\\nPrediction: {'Approved' if pred[0] == 1 else 'Rejected'}")
print(f"Confidence: {prob[pred[0]]*100:.2f}%")""")

    add_md("## 20. Conclusion\nThe notebook successfully demonstrated data ingestion, cleaning, EDA, feature encoding, model selection, evaluation using multiple metrics and charts, and serialization of the artifacts (model and encoder) for deployment via Streamlit.")

    nb['cells'] = cells
    
    os.makedirs('notebook', exist_ok=True)
    with open('notebook/Loan_Approval_System.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print("Notebook successfully generated at notebook/Loan_Approval_System.ipynb")

if __name__ == '__main__':
    create_notebook()
