# Neural Sentiment Classification using PyTorch (FFNN & RNN)

This project implements end-to-end neural pipelines for **5-class sentiment classification** on large-scale Yelp review data using **PyTorch**. Two architectures are built and evaluated: a **Bag-of-Words Feedforward Neural Network (FFNN)** and a **sequence-based Recurrent Neural Network (RNN)** with pretrained word embeddings.

The goal is to compare feature-based vs sequence-based modeling approaches from a **software engineering in ML** perspective, focusing on scalability, reproducibility, and evaluation workflows.

---

## Dataset
- **Yelp Reviews (1–5 star ratings)**
- Training set: **560,000** reviews  
- Validation set: **40,000** reviews  
- Test set: **800** reviews  
- Total: **600K+ reviews**

All text is lowercased and whitespace-tokenized.  
The FFNN uses a Bag-of-Words representation with `<UNK>` handling, while the RNN uses **frozen 50-dimensional pretrained embeddings**.

---

## Models

### Feedforward Neural Network (FFNN)
- Bag-of-Words input representation  
- Linear → ReLU → Linear → LogSoftmax  
- Trained with **NLLLoss** and **AdamW**
- Strong baseline due to global word-signal aggregation

### Recurrent Neural Network (RNN)
- Sequential processing with `nn.RNN`
- Frozen pretrained word embeddings
- Pooling strategies evaluated:
  - Sum pooling
  - Mean pooling
  - Max pooling
  - Last hidden state

---

## Training & Optimization
- Implemented **modular training loops** with reproducible experiment setup
- Fixed random seeds for consistency
- Early stopping (patience = 3)
- Learning rate scheduling using `ReduceLROnPlateau`
- **Hyperparameter tuning with Optuna (30 trials)**:
  - Hidden dimensions
  - Learning rates
  - Batch sizes
  - Dropout

---

## Results

| Model | Validation Accuracy | Test Accuracy |
|------|---------------------|---------------|
| FFNN (optimized) | **64.75%** | **62.5%** |
| RNN (best) | 56.13% | 54.75% |

- Optuna tuning improved FFNN validation accuracy from **59.2% → 64.75% (+5.55%)**
- FFNN outperformed RNN due to strong global feature aggregation and lower sequence overhead
- RNN showed improved behavior on **longer, sentiment-rich reviews**

---

## Error Analysis
- Common misclassifications occurred between **4-star and 5-star reviews**
- Ambiguous sentiment phrases (e.g., positive content with minor complaints)
- Frozen embeddings acted as a regularizer, limiting overfitting but reducing expressiveness

---

## Engineering Takeaways
- Feature-based models can outperform sequence models at scale when sentiment signals are additive
- Hyperparameter optimization and ablation studies provide measurable gains with minimal architectural changes
- Reproducible experiment design is critical for reliable model comparison

---

## Tech Stack
- **Python, PyTorch**
- Optuna
- NumPy
- scikit-learn
- Matplotlib, Seaborn

---

## How to Run
```bash
pip install torch optuna scikit-learn numpy matplotlib seaborn tqdm
python ffnn.py
python rnn.py
