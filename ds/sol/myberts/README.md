# My BERTS

Create your own BERTS models for Topic Modelling, Sentiment and Emotional Analysis in a multilingual way using different sources.

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
data_predict = data/data_bert/processed/whatsapp/chat_processed.csv             # Text to predict
data_train = data/data_bert/processed/sab/data_label.csv                        # Training data set

[OUTPUTS]
data_predicted = data/data_bert/output/out_data.csv                             # Text predicted

[MODELS]
bert_model = data/data_bert/pre_trained_models                                  # Pretrained models
bert_tokenizer = data/data_bert/pre_trained_models/vocab_tokenizer.txt          # Tokenizer vocabulary
batch_size = 10                                                                 # Number of batches
epochs = 10                                                                     # Number of epochs

[OUT_MODELS]
dir = data/data_bert/fine_tune_models                                           # Fine tune models created
model_selected = data/data_bert/fine_tune_models/BERT_ft_epoch10.model          # Model to predict
```

## Scripts

The scripts are enumarated as following:

* `_bert_sab_train_senti.py`
* `_bert_sab_predict_senti.py`

### About _bert_sab_train_senti.py

#### Description

The script does:
1. Create multiple fine tune BERT models for sentiment (set by the number of epochs) from `[INPUTS][data_train]` using SAB data.
2. Save the models in `[OUT_MODELS][dir]` folder.

The labels are related to SAB data. Please look at `dsaa.core.cleaners.sab.py` and `DICT_LABEL`.

You can find the original data in the following link:

http://sabcorpus.linkeddata.es/

### About _bert_sab_predict_senti.py

#### Description

The script does:
1. Predict the sentiment for new texts using the model selected in `[OUT_MODELS][model_selected]`
2. Save the prediction in `[OUTPUTS][data_predicted]`
