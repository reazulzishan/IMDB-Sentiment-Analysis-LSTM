# 🎬 Movie Review Sentiment Analysis (Bidirectional LSTM)

An advanced Natural Language Processing (NLP) project that classifies IMDb movie reviews as **Positive** or **Negative** using a **Bidirectional LSTM** neural network.

## 🧠 Project Overview
This project goes beyond simple text classification by utilizing a **Bidirectional Long Short-Term Memory (Bi-LSTM)** architecture. Unlike standard models, this network reads text from both directions (left-to-right and right-to-left), allowing it to capture context, sarcasm, and complex sentence structures more effectively.

## ✨ Key Features
- **Bidirectional LSTM:** Processes text in both directions to understand context better than standard RNNs/LSTMs.
- **Sarcasm Detection:** Capable of identifying misleading sentiments (e.g., "Good acting but garbage story").
- **Word Embedding:** Maps 10,000 frequent words into a dense vector space for semantic understanding.
- **Smart Padding:** Normalizes review lengths to ensure consistent model input.
- **Visual Analytics:** Includes Accuracy/Loss graphs and a Confusion Matrix for error analysis.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Dataset:** IMDb Large Movie Review Dataset (50,000 reviews)

## 📊 Performance & Results
The model was evaluated on 25,000 unseen test reviews.
- **Model Architecture:** Embedding -> Bi-LSTM (64) -> Dense (64) -> Dropout -> Output
- **Accuracy:** ~85% (Validation)
- **Confusion Matrix:** Shows strong capability in distinguishing positive vs. negative sentiment with minimal false positives.

## 🚀 How to Run
1. Clone this repository.
2. Open the `.ipynb` notebook in Google Colab or Jupyter.
3. Run the training cells to build the model.
4. Use the `predict_sentiment()` function to test your own reviews!

### Example Test:
```python
predict_sentiment("The visual effects were stunning, but the plot was nonexistent.")
# Output: Negative Review (Correctly identifies the negative sentiment despite the positive start)
