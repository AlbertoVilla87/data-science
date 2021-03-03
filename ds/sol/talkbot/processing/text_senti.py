# pylint: skip-file
# flake8: noqa
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

# BERT training
def train_bert(sentences, labels, bert_model, tokenizer_vocab):
    """
    Train BERT model
    :param iterable sentences:
    :param BERT bert_model:
    :param string tokenizer_vocab:
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_vocab,
                                              do_lower_case=True)
    encoded_sents = tokenizer.batch_encode_plus(sentences,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding=True,
                                                max_length=256,
                                                truncation=True,
                                                return_tensors="pt")
    sent_ids = encoded_sents["inputs_ids"]
    attention_masks = encoded_sents["attention_mask"]
    labels = torch.tensor(labels)
    data_train = TensorDataset(sent_ids, attention_masks, labels)
