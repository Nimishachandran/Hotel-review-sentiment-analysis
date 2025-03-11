# Hotel-review-sentiment-analysis
 **Description**
 This project focuses on sentiment analysis of hotel reviews to enhance personalized guest experiences. By applying machine learning and NLP techniques, the model predicts whether a guest's review is positive or negative, helping hotels improve customer satisfaction. 
# Hotel Review Sentiment Analysis 🏨💬

##  Project Overview
This project focuses on **sentiment analysis of hotel reviews** to enhance personalized guest experiences. By applying **machine learning and NLP techniques**, the model predicts whether a guest's review is **positive or negative**, helping hotels improve customer satisfaction.

##  Features
- **Data Cleaning & Preprocessing** (Pandas, NLTK)
- **Machine Learning Models:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
- **Performance Evaluation** (Accuracy, Precision, Recall, Confusion Matrix)
- **Data Visualization** (Matplotlib, Seaborn)

##  Project Structure
```
hotel-review-sentiment-analysis/
│── README.md
│── requirements.txt
│── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│── notebooks/
│   ├── eda.ipynb
│   ├── training.ipynb
│── data/
│   ├── train.csv.zip
│   ├── test.csv.zip
│── models/
│   ├── trained_model.pkl
│── results/
│   ├── confusion_matrix.png
│   ├── performance_metrics.txt
│── .gitignore
```

##  Installation & Usage
### 1. Clone the Repository
```bash
git clone https://github.com/Nimishachandran/hotel-review-sentiment-analysis.git
cd hotel-review-sentiment-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training
```bash
python src/model_training.py
```

##  Results
###  Model Performance Metrics:
- **Accuracy:** Evaluated for different models.
- **Precision, Recall, and F1-Score:** Included in the classification report.

 ## 🔗 Dataset Source
- https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe/code

##  Contributions
Feel free to contribute! Fork the repo, create a new branch, and submit a pull request.

##  License
This project is licensed under the [MIT License](LICENSE).

---
*Developed as part of the MSc Data Science project for Personalized Guest Experience through Sentiment Analysis.*

