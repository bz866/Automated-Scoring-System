# Automated-Scoring-System
Automated Scoring System - DS-1011 Natural Language Processing

* LR_Model.ipynb 
Linear Regression baseline model

* BiLSTM folder 
Bi-directional lstm model
  * converted score to range 0-10
  * pretrained GloVe 6b.50d.txt
  * word embeddings leave fine-tuned in training process

* aes folder
Attention based cnn/lstm stacked model
  * see hi_LSTM-CNN.py for model training
  * see hier_networks.py/build_hrcnn_model() for model structure
  * see softattention.py for attention() function.
