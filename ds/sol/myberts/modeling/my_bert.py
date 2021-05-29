# pylint: skip-file
# flake8: noqa
from configparser import ConfigParser
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

from ds.sol.myberts import logger, CONF_INI

CFG = ConfigParser()
CFG.read(CONF_INI)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting

TEST_SIZE = 0.2

# BERT training

def fit(data, x_label, y_label, bert_model, tokenizer_vocab, label_dict):
    """
    Train BERT model
    :param pandas data: info with text and labels
    :param iterable x_labels: input columns
    :param iterable y_labels: target columns
    :param BERT bert_model: binary model
    :param string tokenizer_vocab: tokenizer file
    :param dict label_dict: class dicctionary
    """
    logger.info("Split Training and Test..")
    x_data = data[x_label]
    y_data = data[y_label]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=TEST_SIZE,
                                                        random_state=107, stratify=y_data)
    logger.info("Distribution Train Labels: ")
    logger.info(str(np.unique(y_train, return_counts=True)))
    logger.info("Distribution Test Labels: ")
    logger.info(str(np.unique(y_test, return_counts=True)))
    data_train = pd.concat([x_train, y_train], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)
    logger.info("Create dataloader for training...")
    data_load_train = create_data_loader(data_train[x_label].values,
                                         tokenizer_vocab,
                                         data_train[y_label].values)
    logger.info("Create dataloader for test...")
    data_load_test = create_data_loader(data_test[x_label].values,
                                        tokenizer_vocab,
                                        data_test[y_label].values,
                                        train_mode=False)
    logger.info("Setting up BERT Pretrained Model...")
    num_labels = len(np.unique(data_train[y_label].values))
    model = BertForSequenceClassification.from_pretrained(bert_model,
                                                          num_labels=num_labels,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    logger.info("Training...")
    epochs = int(CFG["MODELS"]["epochs"])
    fine_tune(epochs, model, data_load_train, data_load_test, label_dict)

# Data Loader

def create_data_loader(sentences, tokenizer_vocab, labels=[], train_mode=True):
    """
    Create a dataloader BERT
    :param iterable sentences: text instances
    :param string tokenizer_vocab: tokenizer file
    :param iterable labels: sentiment or emotions class
    """
    logger.info("Loading Tokenize and Encoding data..")
    tokenizer = BertTokenizer(tokenizer_vocab,
                              do_lower_case=True)
    encoded_sents = tokenizer.batch_encode_plus(sentences,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                padding=True,
                                                max_length=256,
                                                truncation=True,
                                                return_tensors="pt")
    sent_ids = encoded_sents["input_ids"]
    attention_masks = encoded_sents["attention_mask"]
    if len(labels) > 0:
        labels = torch.tensor(labels)
        data = TensorDataset(sent_ids, attention_masks, labels)
    else:
        data = TensorDataset(sent_ids, attention_masks)
    logger.info("Creating Data Loaders...")
    batch_size = int(CFG["MODELS"]["batch_size"])
    if train_mode:
        dataloader = DataLoader(data,
                                sampler=RandomSampler(data),
                                batch_size=batch_size)
    else:
        dataloader = DataLoader(data,
                                sampler=SequentialSampler(data),
                                batch_size=batch_size)
    return dataloader

# Optimize

def set_optimizer(model, epochs, data_loader):
    """
    Optimize using AdamW method and schedule
    """
    steps_train = len(data_loader)*epochs
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)
    return optimizer


def set_scheduler(optimizer, data_loader):
    """
    [summary]
    :param optimizer: [description]
    :return: [description]
    """
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(data_loader))
    return scheduler

# Performance metrics

# Accuracy per class

def performance_acc_class(preds, labels, labels_dict):
    """
    Evaluate the model accuracy per class
    :param iterable preds: predicting values
    :param iterable labels: target values
    :param dict label_dict: class dicctionary
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        logger.info(f'Class: {label_dict_inverse[label]}')
        logger.info(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}')

# Fine-tune BERT model

def fine_tune(epochs, model, data_loader_train, data_loader_val, label_dict):
    """
    Fine-tune of BERT model using run_glue
    approach from HuggingFace
    """
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)           # For GPU
    model.to(DEVICE)
    logger.info("Device: " + str(DEVICE))
    label_values = list(label_dict.values())
    target_names=list(map(str,label_values))
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(data_loader_train,
                            desc="Epoch {:1d}".format(epoch),
                            leave=False,
                            disable=False)
        for step, batch in enumerate(progress_bar):
            model.zero_grad()
            # Each component of the batch is in the correct device
            batch = tuple(b.to(DEVICE) for b in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer = set_optimizer(model, epochs, data_loader_train)
            scheduler = set_scheduler(optimizer, data_loader_train)
            optimizer.step()
            scheduler.step()
            logger.info(str(progress_bar) + str({"training_loss": "{:.3f}". \
                        format(loss.item()/len(batch))}))

        model_name = CFG["OUT_MODELS"]["dir"] + "/" + f"BERT_ft_epoch{epoch}.model"
        torch.save(model.state_dict(), model_name)
        tqdm.write("\nEpoch {epoch}")
        loss_train_avg = loss_train_total/len(data_loader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_values = evaluate(data_loader_val, model)
        predictions = np.argmax(predictions, axis=1).flatten()
        true_values = true_values.flatten()
        val_f1 = f1_score(true_values, predictions, average="weighted")
        accuracy_s = accuracy_score(true_values, predictions)
        precision_s = precision_score(true_values, predictions, average=None)
        recall_s = recall_score(true_values, predictions, average=None)
        conf_matrix = confusion_matrix(true_values, predictions, labels=label_values)
        report = classification_report(predictions, true_values, labels=label_values,
                                       target_names=target_names)
        tqdm.write(f"Validation loss: {val_loss}")
        tqdm.write(f"F1 Score (weighted): {val_f1}")
        logger.info("Classification Report:")
        logger.info(report)
        logger.info(f"F1 Score (weighted): {val_f1}")
        logger.info(f"Precision: {precision_s}")
        logger.info(f"Recall: {recall_s}")
        logger.info(f"Accuracy: {accuracy_s}")
        logger.info("Dict label: " + str(label_dict))
        logger.info("Confusion Matrix: %s", str(conf_matrix))

# Evaluate

def evaluate_predict(data_loader, model):
    """
    [summary]
    :param data_loader: [description]
    :type data_loader: [type]
    """
    model.eval()
    predictions = []
    progress_bar = tqdm(data_loader,
                    desc="Prediction",
                    leave=False,
                    disable=False)
    for batch in progress_bar:
        batch = tuple(b.to(DEVICE) for b in batch)
        inputs = {"input_ids" : batch[0],
                  "attention_mask": batch[1],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        logger.info(str(progress_bar))

    predictions = np.concatenate(predictions, axis=0)
    return predictions

def evaluate(data_loader, model):
    """
    [summary]
    :param data_loader: [description]
    :type data_loader: [type]
    """
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in data_loader:
        batch = tuple(b.to(DEVICE) for b in batch)
        inputs = {"input_ids" : batch[0],
                  "attention_mask": batch[1],
                  "labels": batch[2]
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(data_loader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

# Predict

def predict(bert_model, my_bert_model,
            tokenizer_vocab, sentences, label_dict):
    """
    Predict the label of a list of sentences and save in a CSV
    :param string bert_model: path of the pretrained BERT
    :param string my_bert_model: path of the BERT created
    :param string tokenizer_vocab: path of the tokernizer vocabulary
    :param iterable sentences: list of sentences
    :param dict label_dict: label dictionary
    :return: list of predictions
    """
    label_list = list(label_dict)
    model = BertForSequenceClassification.from_pretrained(bert_model,
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(DEVICE)
    model.load_state_dict(torch.load(my_bert_model,
                                     map_location=torch.device("cpu")))
    data_loader = create_data_loader(sentences,
                                     tokenizer_vocab,
                                     train_mode=False)
    predictions = evaluate_predict(data_loader, model)
    logits = np.argmax(predictions, axis=1).flatten()
    probs = np.max(predictions, axis=1).flatten()
    labels_pred = [label_list[logit] for logit in logits]
    return labels_pred, probs
