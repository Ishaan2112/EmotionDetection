GoEmotions – Text-Based Emotion Detection Pipeline

This project implements a complete end-to-end emotion detection system using the GoEmotions (Google Emotions Dataset). The pipeline includes data preprocessing, embedding generation using sentence-transformers/all-MiniLM-L6-v2, model training using multiple ML algorithms, evaluation with strong visualizations, and a demo function for inference.

📂 Project Contents
GoEmotions_Emotion_Detection_Clean.ipynb   # Main notebook with full pipeline
xgboost_model.pkl                          # Saved XGBoost model
logistic_regression_model.pkl              # Saved Logistic Regression model
README.md                                  # This file

🎯 Task Requirements

This project meets the following assignment requirements:

✅ Use Pandas & NumPy for data handling

✅ Generate embeddings using sentence-transformers/all-MiniLM-L6-v2

✅ Train ML models using scikit-learn / XGBoost

✅ Visualize results using Matplotlib/Seaborn/Plotly

✅ Provide evaluation metrics:

Accuracy

Precision

Recall

F1-score

Confusion matrix

✅ Optionally compare two different models

✅ Provide a demo function: predict_emotion("sample text")

✅ All work completed in a clean, organized Jupyter Notebook

📥 Dataset

Source:
https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset

Download the dataset and place it inside a data/ folder (or update path in the notebook).

🚀 How to Run the Project
Option 1: Run using Jupyter Notebook

Create a virtual environment:

python3 -m venv env
source env/bin/activate   # Mac/Linux
# OR
env\Scripts\activate      # Windows


Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn plotly sentence-transformers xgboost joblib jupyter


Start Jupyter Notebook:

jupyter notebook


Open:

GoEmotions_Emotion_Detection_Clean.ipynb


Run all cells top to bottom.

Option 2: One-Click Run Script

Linux/Mac:

./run.sh


Windows:

run.bat

🧠 Model Training Summary

Two models were trained and compared:

Logistic Regression

XGBoost Classifier

Input features: 384-dimensional embeddings
Output: multi-label emotion predictions

📊 Evaluation

The notebook includes:

Confusion matrix

Classification report

Metrics visualization

Optional ROC/PR curves

📈 Demo Inference Function
predict_emotion("I am really excited today!")


Returns a dictionary with predicted emotion(s) and confidence scores.

🔧 Project Requirements

Python 3.8+

sentence-transformers

scikit-learn

xgboost

matplotlib

seaborn

pandas

numpy

joblib

jupyter