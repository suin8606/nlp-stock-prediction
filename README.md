# NLP Stock Market Prediction: Can Reddit Headlines Predict the DJIA?

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**Can daily Reddit news headlines predict whether the Dow Jones will go up or down?**  
An end-to-end NLP classification pipeline combining text features with market movement data.

---

## 📉 Result: Near-Random Performance — And That's the Point

> Models achieved accuracy close to random chance. Rather than inflating results, this project documents why — and connects it to the **Efficient Market Hypothesis**: if public news already moves prices, it contains little predictive signal by the time it's observable.

| Model | Accuracy |
|-------|----------|
| Random Forest | ~53% |
| Logistic Regression | ~52% |
| Naive Bayes | ~51% |
| Baseline (majority class) | ~50% |

---

## 🧠 The Business Question

Financial news moves markets — but does it move them *predictably*? I tested whether headlines scraped from Reddit's r/worldnews could classify next-day DJIA movement (up or down) better than random chance.

> *If markets are efficient, public information should already be priced in — meaning NLP on public headlines shouldn't outperform a coin flip.*

---

## 🔬 Analysis Pipeline

```
Step  Task                                         
----  ----                                         
1     Load Reddit headlines + DJIA dataset         
2     EDA — headline length, word frequency        
3     Text preprocessing — lowercase, stopwords, punctuation
4     Feature extraction — Bag of Words + TF-IDF   
5     Train/test split (time-based, no leakage)    
6     Model training — Logistic Regression, Random Forest, Naive Bayes
7     Model evaluation — accuracy, confusion matrix
8     Error analysis — where do models fail?       
9     Consolidated model comparison                
10    Honest conclusions tied to EMH               
```

---

## 📐 Methods Reference

| Method | Purpose |
|--------|---------|
| TF-IDF | Weight rare but informative words over common ones |
| Bag of Words | Baseline text representation |
| Logistic Regression | Linear baseline classifier |
| Random Forest | Ensemble method, handles non-linear patterns |
| Naive Bayes | Fast probabilistic classifier for text |
| Time-based split | Prevent data leakage — future data never trains the model |
| Confusion matrix | Understand where models fail, not just overall accuracy |

---

## 💡 What I Would Do Next

1. **Sentiment scoring** — instead of raw word counts, use VADER or FinBERT to extract sentiment polarity from headlines
2. **Intraday data** — daily close-to-close movement is noisy; intraday reactions to specific news events may have stronger signal
3. **Entity recognition** — headlines mentioning specific companies or sectors may have more targeted predictive power
4. **Longer context window** — aggregate headlines over 3–5 days rather than single-day snapshots

---

## 📂 Folder Structure

```
nlp-stock-prediction/
│
├── README.md
├── nlp_stock_prediction.ipynb
├── requirements.txt
└── .gitignore
```

---

## ▶️ How to Run

```bash
# 1. clone
git clone https://github.com/suin8606/nlp-stock-prediction.git
cd nlp-stock-prediction

# 2. install dependencies
pip install -r requirements.txt

# 3. run notebook
jupyter notebook nlp_stock_prediction.ipynb
```

---

## 🗂️ Dataset

[Daily News for Stock Market Prediction — Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews)

| File | Description |
|------|-------------|
| Combined_News_DJIA.csv | Reddit top 25 headlines per day + DJIA movement label (2008–2016) |

---

## 🏷️ Skills Demonstrated

`Python` `NLP` `Text Preprocessing` `TF-IDF` `Bag of Words`  
`Logistic Regression` `Random Forest` `Naive Bayes` `Model Evaluation`  
`EDA` `Data Visualization` `Financial Data` `Scikit-learn`

---

## 👤 Author

**Suin Kim** — M.S. Statistics & Data Science, Baruch College  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/suinkim29)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:suin.kim29@gmail.com)
