# pylint: skip-file
# flake8: noqa
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
# Bert Tokenizer for sentences
def tokenize_bert_sents(sentences):
    """
    Tokenize the sentences using BERT methodology
    :param iterable sentences:
    """
    encoded_sents = TOKENIZER.batch_encode_plus(sentences,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                pad_to_max_length=True,
                                                max_length=256,
                                                return_tensors="pt"
    )