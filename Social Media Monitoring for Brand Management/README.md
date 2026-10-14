# Social Media Brand Sentiment Monitoring

I built this project as an end-to-end NLP case study using the **Twitter US Airline Sentiment** dataset. The goal is to clean social-media text, classify tweets as negative, neutral, or positive, and demonstrate how sentiment predictions can be summarized by brand and date.

The project is an academic prototype. The dataset is historical and should not be interpreted as a measure of current airline reputation.

## What I implemented

- Inspected the dataset, its class distribution, and missing values in the fields used by the analysis.
- Explored the dataset's original sentiment labels across airlines and dates.
- Preprocessed tweets by removing URLs, hashtags, mentions, emoticons, and common emoji symbols; normalizing case and punctuation; expanding common contractions; and preserving negation words such as `not`, `no`, `nor`, and `never`.
- Built a reusable scikit-learn text-cleaning transformer and placed it inside a pipeline so training and inference use the same preprocessing.
- Kept negative, neutral, and positive sentiment as separate classes.
- Split the data into training and held-out test sets before hyperparameter tuning.
- Used stratified cross-validation and randomized search to compare word- and character-level TF-IDF features and SVM configurations.
- Evaluated the selected model using accuracy, macro F1, weighted F1, a classification report, a confusion matrix, and test-set error inspection.
- Added functions to predict sentiment for new posts and to produce brand/date summaries with configurable review thresholds.

## Dataset

I use the **Twitter US Airline Sentiment** dataset published on Kaggle by CrowdFlower. It contains 14,640 tweets about six US airlines, collected in February 2015, with existing positive, neutral, or negative sentiment labels.

- Dataset page: <https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment>
- Expected filename: `Tweets.csv`

I do not include the dataset in this repository. Download it from the source above, review its applicable terms, and place `Tweets.csv` in the repository root (beside this README and notebook). The notebook also checks several common Kaggle and local paths; if yours is stored elsewhere, update `DATASET_CANDIDATES` in the notebook.

## Repository layout

```text
social-media-brand-sentiment-monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── main.ipynb
└── Tweets.csv
```

Rename my notebook to `main.ipynb` when you add it to this repository.

## Setup

I recommend Python 3.10 or newer and a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the notebook

1. Download `Tweets.csv` from the dataset source and place it in the repository root, or update `DATASET_CANDIDATES` to point to its location.
2. Start Jupyter:

   ```bash
   jupyter lab
   ```

3. Open `main.ipynb`.
4. Run the notebook from top to bottom. The notebook downloads the NLTK resources it needs (`punkt`, `punkt_tab`, and `stopwords`) during setup; the first run requires internet access for those downloads unless the resources are already installed.

The randomized search evaluates 40 sampled parameter configurations with three-fold stratified cross-validation. This can take a few minutes or longer depending on the computer. I keep the held-out test set out of tuning and use it for final evaluation.

## Results from the saved notebook run

The saved execution of the notebook reported:

| Metric | Baseline | Tuned model |
|---|---:|---:|
| Test accuracy | 79.03% | 79.17% |
| Macro F1 | 0.7170 | 0.7200 |
| Weighted F1 | — | 0.7857 |

The best mean cross-validation accuracy in that run was 79.57%. The improvement on the held-out test set was small, so I describe it as a modest improvement rather than a major gain. Results can vary with library versions, data changes, and execution settings; use the metrics printed by a fresh run as the source of truth.

I report macro F1 alongside accuracy because the dataset is imbalanced. In the saved evaluation, negative tweets were easier to identify than neutral or positive tweets, so the model can still misclassify nuanced feedback.

## Brand monitoring

The notebook includes two different kinds of analysis, which I keep separate:

1. **Historical analysis:** summaries by airline and date using the dataset's original sentiment labels.
2. **Prediction-based monitoring:** a function that scores incoming posts and groups predicted sentiments by brand and UTC date.

The example incoming posts are illustrative examples used to test the workflow. The negative-sentiment percentage and minimum-volume thresholds are demonstration settings, not validated business rules. The monitoring function is a prototype and should be checked against human-labelled, current data before use in an operational setting.

## Limitations

- The tweets were collected in February 2015; they do not represent current airline reputation.
- The model predicts sentiment, not the factual truth or severity of a complaint.
- Negation and contractions are handled with rules, so some ambiguous language may still be normalized imperfectly.
- Removing hashtags removes their words as requested by the case study, which can also remove potentially useful context.
- SVM predictions and aggregate alert percentages inherit classification errors, especially for neutral sentiment.
- The monitoring thresholds are illustrative and have not been statistically calibrated.

## Skills demonstrated

Python, Pandas, exploratory data analysis, regular expressions, NLTK, text normalization, scikit-learn pipelines, TF-IDF, SVM classification, randomized hyperparameter search, cross-validation, model evaluation, error analysis, and time-based sentiment monitoring.
