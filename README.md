# AG News Text Classification

**Note**: This project was completed as part of the AIML332 course at Te Herenga Waka — Victoria University of Wellington.

## Objective
Classify news articles from the AG News Dataset using both traditional feature-based models (TF and TF-IDF with Naive Bayes) and deep learning approaches (CNNs with trainable or pre-trained GloVe embeddings, and fine-tuned DistilBERT). Compare performance across these approaches to assess the impact of model complexity and embedding quality.

## Data and Pre-trained Embeddings
This project requires the following files:
- *Included in this repo:*
  - `test.csv` - Test set for AG News.
- *To download externally:*
  - `train.csv` - Full AG News training dataset.
    - Download from: https://huggingface.co/datasets/cherac/ag_news_classification_data/resolve/main/train.csv 
  - `glove.6B.100d.txt` - Pre-trained GloVe embeddings.
    - Download from: https://huggingface.co/datasets/cherac/ag_news_classification_data/resolve/main/glove.6B.100d.txt
   
## Structure
- `test.csv` - Test set for AG News.
- `traditional_vs_cnn_text_classification.ipynb` - Traditional models and CNN approaches.
- `distilbert_text_classification.ipynb` - Fine-tuned DistilBERT.
- `requirements.txt` - Python dependencies.

**Instructions:** Place all files in the *same folder as the notebooks* before running them.

## Methods
- **Traditional / CNN notebook:**
  - Preprocess text data.
  - Train TF and TF-IDF with Naive Bayes; record training and test accuracy.
  - Train CNNs with random embeddings and fixed GloVe embeddings; record training/validation and test accuracy.
- **DistilBERT notebook:**
  - Preprocess and clean text (lowercasing, removing punctuation/special characters).
  - Tokenise text using DistilBERT tokeniser.
  - Fine-tune pretrained DistilBERT with HuggingFace Trainer and evaluate on the test set.
 
## Key Results
| Model                          | Test Accuracy |
|--------------------------------|---------------|
| TF + Naive Bayes               | 90%           |
| TF-IDF + Naive Bayes           | 90.5%         |
| CNN (random embeddings)        | 90.9%         |
| CNN (fixed GloVe embeddings)   | 91.3%         |
| DistilBERT (fine-tuned)        | 93.5%         |

**Observations:**
- Accuracy improves as model progress from traditional bag-of-words approachs to deep learning models.
- Using pre-trained embeddings (GloVe) slightly improves CNN performance.
- Fine-tuned DistilBERT yields the highest test accuracy, highlighting the effectiveness of large-scale pre-trained models for text classification

## How to Run:
Python version: 3.10+
```bash
pip install -r requirements.txt
jupyter notebook
# Then open traditional_vs_cnn_text_classification.ipynb or distilbert_text_classification.ipynb in the browser, and run the cells in order
```

## Summary
This project demonstrates that deep learning architectures outperform traditional feature-based methods for text classification. Leveraging pretrained embeddings (GloVe) and fine-tuning transformer models (DistilBERT) further improves accuracy, highlighting the value of richer model architectures and semantic representations.

## Reproducibility / Notes
- Random seeds are fixed where applicable to ensure consistent results.
- Both notebooks can be run end-to-end.


