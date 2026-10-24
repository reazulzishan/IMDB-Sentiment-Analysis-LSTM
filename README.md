# IMDb Sentiment Analysis (Bidirectional LSTM)

Classifies IMDb movie reviews as positive or negative using a bidirectional LSTM — a network that reads each review forwards *and* backwards, so context that arrives late in a sentence still counts.

## Why bidirectional

A standard LSTM reads left to right, which means it commits to an interpretation before it has seen the end of the sentence. Reviews break that assumption constantly:

```python
predict_sentiment("The visual effects were stunning, but the plot was nonexistent.")
# → Negative
```

Everything before the comma is positive. The verdict is in the second half.

## Results

| Metric | Value |
|---|---|
| Test accuracy | To be updated after the re-run of the revised notebook |
| Test set | 25,000 unseen reviews, evaluated once at the end |
| Training set | 20,000 reviews for training, 5,000 held out for validation |

The earlier ~85% figure came from a run that used the test set for validation during training, so it was not a clean held-out result. The revised notebook keeps the two separate.

Evaluated with accuracy and loss curves, a confusion matrix and a per-class precision/recall/F1 report, which show where the errors actually fall rather than collapsing them into one number.

## Training setup

- **Validation split:** 20% of the training data, so the test set is never seen during training.
- **Early stopping:** training stops when validation loss hasn't improved for 2 epochs, and the best epoch's weights are restored. The earlier run overfitted after epoch 2 (validation loss rose from 0.37 to 0.48).
- **Up to 10 epochs**, Adam optimiser, binary cross-entropy.

## Architecture

```
Embedding (10,000 words) → Bidirectional LSTM (64) → Dense (64) → Dropout → Output
```

**Word embeddings** map the 10,000 most frequent words into a dense vector space, so semantically related words sit near each other instead of being independent one-hot columns.

**Sequence padding** normalises review lengths to 250 tokens. Padding zeros are masked (`mask_zero=True`), so the LSTM doesn't read them. Long reviews are cut from the start (`truncating='pre'`), which keeps the ending, where the verdict usually is.

## Tech stack

TensorFlow · Keras · NumPy · Pandas · Matplotlib · Seaborn

Dataset: IMDb Large Movie Review Dataset (50,000 labelled reviews).

## Running it

```bash
git clone https://github.com/reazulzishan/IMDB-Sentiment-Analysis-LSTM.git
```

Open the notebook in Jupyter or Google Colab, run the training cells, then call `predict_sentiment()` on your own text.

---

**Reazul Akter Zishan** — CSE undergraduate, Daffodil International University
[Portfolio](https://zishan-portfolio-me.netlify.app) · [LinkedIn](https://www.linkedin.com/in/reazul-zishan) · [GitHub](https://github.com/reazulzishan)
