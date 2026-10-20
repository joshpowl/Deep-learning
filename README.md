# Deep Learning Project: Text Sentiment Classification

## Overview
This project classifies short English reviews as **positive** or **negative** using a PyTorch feed-forward neural network (MLP). Text is tokenized into word unigrams and bigrams and represented using TF-IDF features. Training uses a validation split and early stopping; a separate held-out test set is used for final evaluation.

**Dataset note:** The bundled dataset contains 370 balanced short example reviews curated for this educational demonstration (185 positive and 185 negative). It is self-contained and can run offline. It is not a public benchmark, so the reported metrics should not be interpreted as expected real-world performance.

## Project contents

- `notebooks/Deep_Learning_Sentiment_Classification.ipynb` — project walkthrough with training, results, plots, and inference.
- `data/sentiment_dataset.csv` — labeled example reviews (`text`, `label`).
- `train_sentiment.py` — PyTorch training/evaluation code and artifact generation.
- `predict_sentiment.py` — load the saved model and classify new text.
- `train_distilbert_transfer.py` — optional transfer-learning implementation using pretrained DistilBERT (requires internet on first run and optional dependencies).
- `artifacts/sentiment_mlp.pth` — saved PyTorch model weights/configuration.
- `artifacts/tfidf_vectorizer.joblib` — fitted TF-IDF tokenizer/vectorizer required for matching inference features.
- `artifacts/metrics.json`, `classification_report.txt`, `test_predictions.csv` — evaluation results.
- `artifacts/training_curves.png`, `confusion_matrix.png`, `roc_curve.png`, `model_comparison.png` — plots.
- `docs/Deep_Learning_Sentiment_Project_Report.docx` and `.pdf` — short project report in editable and submission-ready formats.

## Environment setup
Python 3.11 or later is recommended. CPU is supported; a GPU is optional.

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run training and evaluation
Run from the project root (the directory containing `train_sentiment.py`):

```bash
python train_sentiment.py --epochs 60 --batch-size 32 --seed 42 --patience 12
```

This saves/replaces the model and metrics in `artifacts/`. Reported metrics include accuracy, precision, recall, F1-score, and ROC-AUC for both the PyTorch MLP and a Logistic Regression reference. The vocabulary/vectorizer is fitted only to training text; the test split is not used to fit the models.

## Run the notebook
```bash
pip install jupyter
jupyter notebook
```
Open `notebooks/Deep_Learning_Sentiment_Classification.ipynb` and run all cells from top to bottom. Start Jupyter in the project root so the notebook can locate the scripts and dataset.

## Inference using the saved model
After training (or using the included saved artifacts), run:

```bash
python predict_sentiment.py "The delivery was quick and the product works perfectly."
python predict_sentiment.py "The app crashes constantly and support did not help."
```

Output is JSON containing predicted sentiment and class scores. The scores are softmax outputs, not calibrated probabilities; they should not be used for consequential decisions.

## Optional pretrained transfer learning
The main, offline-capable deliverable is the PyTorch MLP. An optional fine-tuning script is included to demonstrate transfer learning with pretrained DistilBERT. In a network-enabled environment, install the optional packages and run:

```bash
pip install "transformers>=4.40" "accelerate>=0.29"
python train_distilbert_transfer.py --epochs 3 --batch-size 8
```

The script downloads `distilbert-base-uncased` on first use and saves the fine-tuned transformer to `artifacts/distilbert_finetuned/`. This optional transformer training was not run as part of the included offline model artifact.

## Results snapshot
On the included random stratified test split of 74 reviews, the stored run recorded:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| PyTorch MLP | 0.6216 | 0.6154 | 0.6486 | 0.6316 | 0.6486 |
| Logistic Regression reference | 0.5541 | 0.5556 | 0.5405 | 0.5479 | 0.6844 |

The MLP has higher accuracy on this split, while the logistic model has a higher ROC-AUC. The small sample makes results uncertain and neither model is production-ready.

## Limitations
This is a small curated educational corpus, not a representative commercial dataset. The model may fail on sarcasm, slang, mixed sentiment, domain-specific terms, or text that differs from the examples. For real use, replace the dataset with a larger, representative, properly licensed corpus and perform further validation, error analysis, and monitoring.

## GitHub publishing
Create an empty GitHub repository and run these commands from this folder:

```bash
git init
git add README.md requirements.txt data train_sentiment.py predict_sentiment.py train_distilbert_transfer.py notebooks artifacts docs project_documentation.md
git commit -m "Add PyTorch sentiment classification project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deep-learning-sentiment-classification.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your GitHub username. The saved model and vectorizer are small enough for a normal repository. Do not publish data you do not have permission to share.
