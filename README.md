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
| Validation accuracy | **~85%** |
| Test set | 25,000 unseen reviews |
| Training set | 25,000 reviews |

Evaluated with accuracy and loss curves plus a confusion matrix, which shows where the errors actually fall rather than collapsing them into one number.

## Architecture

```
Embedding (10,000 words) → Bidirectional LSTM (64) → Dense (64) → Dropout → Output
```

**Word embeddings** map the 10,000 most frequent words into a dense vector space, so semantically related words sit near each other instead of being independent one-hot columns.

**Sequence padding** normalises review lengths to a fixed input size.

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
