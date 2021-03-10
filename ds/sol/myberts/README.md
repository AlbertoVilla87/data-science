# My BERTS

Create your own BERTS models for Topic Modelling, Sentiment and Emotional Analysis in a multilingual way.

## Dependencies

### Pretrained model

Bert trained model is used to create the desired model from this one. We use the pretrained multilingual BERT model:

https://github.com/google-research/bert/blob/master/multilingual.md

## Configuration file

`local.ini` contains the parameters configuration and the paths of the pipeline

```python
[LOGS]
Path = logs

[INPUTS]
data = data/data_talkbot/raw/chat.txt                                   # Whatsapp chat example
data_labeled = data/data_talkbot/raw/data_label.csv                     # Training data set

[OUTPUTS]
proc = data/data_talkbot/processed/chat_processed.csv                   # Whatsapp processed
pred = data/data_talkbot/predictions/out_data.csv                       # Final output with predictions

[MODELS]
bert_model = data/data_talkbot/models                                   # Pretrained BERT model
bert_tokenizer = data/data_talkbot/models/vocab_tokenizer.txt           # BERT tokenizer vocabulary
batch_size = 10                                                         # Number of batchs
epochs = 10                                                             # Number of epochs/models

[ENCODING]
spa = latin-1

[OUT_MODELS]
dir = data/data_talkbot/out_models                                      # Models created dirs
model_selected = data/data_talkbot/out_models/BERT_ft_epoch4.model      # Model selected for prediction
```

## Scripts

The scripts are enumarated as following:

* `_bert_train.py`
* `_bert_predict.py`
* `_process_wasap.py (out of scope)`

### About _bert_train.py

#### Description

The script does:
1. Create multiple BERT models (set by the number of epochs) from `data_labeled`.
2. Save the models in `[OUT_MODELS][dir]` folder.

The desired labels are fixed in `corpus.py` from utils. See DICT_LABEL variable. These labels must be consistent with the labels provided by `data_labeled`.

### About _bert_predict.py

#### Description

The script does:
1. Predict the labels for new texts using the model selected in model_selected
2. Save the prediction in `[OUT_MODELS][model_selected]`
