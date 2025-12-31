# 📧 SMS Spam Classifier (97.2% Accuracy)

End-to-end **Machine Learning project** for B.Tech CSE resume. Classifies SMS as **SPAM** or **HAM** using **TF-IDF + Naive Bayes/Logistic Regression** on UCI SMS Spam dataset (5,572 messages).

## 📊 Results Summary
| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Naive Bayes         | **97.2%**| 0.95+   |
| Logistic Regression | 96.8%+   | 0.94+   |

**Test Predictions:**
python predict.py "WINNER!! £900 prize reward!"
→ SPAM (confidence: 90.4%)

python predict.py "Hey, how are you?"
→ HAM (confidence: 1.5%)

## 🚀 Quick Start Tech Stack
1. **Download dataset**: [Kaggle SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
2. Place `spam.csv` in `data/` folder
3. ```bash
   pip install -r requirements.txt
   python notebooks/01_eda_and_preprocessing.ipynb  # Train model
   python predict.py "Free entry to win $1000!"     # Test CLI

## 🛠️ Tech Stack 
Python 3.10+, pandas, NumPy, scikit-learn, matplotlib, seaborn
ML Pipeline: EDA → TF-IDF → Train/Test Split → Model Comparison → Deployment
