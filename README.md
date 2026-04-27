# Stock Market Movement Prediction from News Headlines
### Can daily news predict whether the DJIA goes up or down?

A full NLP pipeline using Reddit news headlines (2008–2016) to predict daily DJIA stock market movement. I built this to explore how well text-based features capture financial market signals — and to be honest about where they fall short.

---

## The Business Question

> *If today's top news headlines are overwhelmingly negative — war, recession, scandal — does the market go down? Can we quantify that relationship and build a predictor?*

This is a classic financial NLP problem. I used the Kaggle Stocknews dataset (top 25 Reddit headlines per day + DJIA labels) to test three modeling approaches and document what works, what doesn't, and why.

---

## Result: Near-Random Performance — And Why That's an Honest Finding

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest (tuned) | see notebook | see notebook |
| Logistic Regression (L1/L2) | see notebook | see notebook |
| Naive Bayes (NER features) | see notebook | — |
| Random baseline | 50% | 0.500 |

> Run the notebook to see exact numbers — they depend on GridSearchCV tuning runs.

**Why performance is near random chance — and why that's expected:**

1. **Efficient Market Hypothesis**: By the time Reddit headlines are posted, markets have already priced in the news. Public information is reflected in prices almost instantly.
2. **Signal-to-noise**: 25 headlines per day cover politics, sports, science, crime — most have no market relevance. Bag-of-words loses the context that makes a headline financially meaningful.
3. **VADER limitations**: VADER was trained on social media, not financial text. "The Fed raised rates" reads as neutral to VADER but is deeply market-moving depending on context.
4. **Missing features**: Market movement is driven by volume, options flow, earnings surprises, and macro data — none of which are in this text-only dataset.

This isn't a failed project — it's an honest one. Documenting *why* a model underperforms is as valuable as reporting a suspiciously high accuracy.

---

## Key Charts

### Target Label Distribution and DJIA Trend
> Slight class imbalance (more up days than down). I use SMOTE + class_weight='balanced' to handle this.



---

### Most Common Words in Headlines
> Dominated by political and economic terms — "government", "bank", "police", "year". Mostly noise from a market-prediction standpoint.



---

### VADER Sentiment vs Market Direction
> Sentiment scores show weak separation between up and down days — the distributions largely overlap. This is the first signal that text features alone won't be enough.



---

### Named Entity Word Clouds
> Top organizations, countries, and people mentioned across 8 years of headlines. The Fed, China, Obama, and Goldman Sachs dominate.



---

### Feature Importance — Random Forest
> Sentiment scores dominate feature importance, with a few TF-IDF terms mixed in. This confirms sentiment is the primary signal the model relies on.



---

### Logistic Regression Coefficients
> Words associated with UP vs DOWN predictions. The coefficient signs are interpretable — negative financial language predicts market decline.



---

### ROC Curves — Model Comparison
> Both models with probability output hover near the random diagonal (AUC ≈ 0.5), confirming the fundamental difficulty of this prediction task.



---

### Error Analysis — Where the Model Fails
> Even high-confidence predictions are frequently wrong. The model's uncertainty (predictions clustered near 0.5) reflects genuine difficulty, not a tuning problem.



---

## How to Export Charts From the Notebook

After each `plt.show()` call, add:
```python
plt.savefig('img/01_label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```
Then commit the `img/` folder and images will render in this README automatically.

---

## My Full Pipeline

```
Step  Task                                              Notes
----  ----                                              -----
1     Load data via Kaggle CLI
2     EDA — label balance, DJIA price trend            Day class imbalance identified
3     Text preprocessing                               Combine 25 headlines → clean →
                                                       tokenize → stopwords → lemmatize
4     TF-IDF vectorization                             1,000 features, unigrams + bigrams
5     VADER sentiment scoring                          Compound, pos, neg, neutral scores
6     Lagged features                                  1-day lag on all sentiment scores
7     Named Entity Recognition                         ORG, GPE, PER extraction via NLTK
8     Time-based train/test split                      2008-2014 train | 2015-2016 test
9     SMOTE oversampling                               Balance minority class (Label=0)
10    Random Forest + GridSearchCV                     Tuned on ROC-AUC, cv=3
11    Logistic Regression + L1/L2 tuning               GridSearchCV on C and penalty
12    Naive Bayes with NER features                    Binary entity mention features
13    Model comparison                                 Accuracy, ROC-AUC, F1, ROC curves
14    Error analysis                                   Where/why models fail
15    Conclusions + future improvements
```

---

## NLP Techniques Used

| Technique | Purpose |
|-----------|---------|
| Tokenization (NLTK) | Split text into individual words |
| Stopword removal | Remove high-frequency low-signal words |
| Lemmatization (WordNetLemmatizer) | Reduce words to base form |
| TF-IDF (sklearn) | Weight words by importance to each day's headlines |
| VADER sentiment | Score headline tone (positive/negative/neutral) |
| Lagged features | Capture yesterday's sentiment as a predictor |
| Named Entity Recognition (NLTK NE chunker) | Extract organizations, countries, people |
| Word cloud (wordcloud) | Visualize most common entities |

---

## Models Used

| Model | Why | Key design choice |
|-------|-----|-------------------|
| Random Forest | Handles high-dimensional sparse data, non-linear patterns | GridSearchCV, SMOTE-balanced training |
| Logistic Regression | Strong text classification baseline, interpretable | L1/L2 regularization via GridSearchCV |
| Naive Bayes | Designed for binary text features | NER entity mentions as features |

---

## What Would Actually Improve Performance

1. **FinBERT** — a BERT model fine-tuned on financial text. It understands "The Fed raised rates" in context, not just as a bag of words.
2. **Pre-market news only** — using only headlines published before market open avoids leaking same-day post-market information.
3. **Macro features alongside text** — VIX (fear index), trading volume, sector ETF performance, earnings calendar.
4. **Longer lag windows** — test 2, 3, 5-day lags. Market reactions to news can be delayed, especially for policy changes.
5. **Intraday granularity** — hourly headlines matched to hourly returns instead of daily aggregation.

---

## Folder Structure

```
nlp-stock-prediction/
│
├── README.md
├── nlp_stock_prediction.ipynb
├── requirements.txt
├── .gitignore
└── img/
    ├── 01_label_distribution.png
    ├── 02_word_frequency.png
    ├── 03_sentiment_boxplots.png
    ├── 04_ner_wordclouds.png
    ├── 05_feature_importance.png
    ├── 06_lr_coefficients.png
    ├── 07_roc_curves.png
    └── 08_error_analysis.png
```

---

## How to Run

```bash
# 1. clone
git clone https://github.com/your-username/nlp-stock-prediction.git
cd nlp-stock-prediction

# 2. install dependencies
pip install -r requirements.txt

# 3. set up kaggle credentials
# place kaggle.json in ~/.kaggle/kaggle.json

# 4. create img folder for chart exports
mkdir img

# 5. run notebook
jupyter notebook nlp_stock_prediction.ipynb
```

The notebook automatically downloads the Stocknews dataset via Kaggle CLI on first run.

---

## Dataset

**Stocknews — Daily News for Stock Market Prediction**  
[kaggle.com/datasets/aaron7sun/stocknews](https://www.kaggle.com/datasets/aaron7sun/stocknews)

| File | Description |
|------|-------------|
| Combined_News_DJIA.csv | Top 25 Reddit headlines per day + DJIA up/down label |
| upload_DJIA_table.csv | Daily DJIA open/close/volume prices |
| RedditNews.csv | Raw Reddit headlines (2008–2016) |

---

## Skills Demonstrated

`Python` `NLP` `Text Preprocessing` `TF-IDF` `VADER Sentiment`  
`Named Entity Recognition` `Random Forest` `Logistic Regression` `Naive Bayes`  
`SMOTE` `GridSearchCV` `ROC-AUC` `Time-Series Train/Test Split`  
`Feature Engineering` `Error Analysis`

---

## Author

**Suin Kim**  
M.S. Statistics & Data Science — Baruch College  
[linkedin.com/in/suinkim29](https://linkedin.com/in/suinkim29) · suin.kim29@gmail.com
