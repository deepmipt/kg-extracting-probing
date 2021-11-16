import csv    
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from collections import Counter
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle, class_weight
from imblearn.under_sampling import RandomUnderSampler 

import matplotlib.pyplot as plt


#  костыльное решение того что я плохо записываю np array в csv
def read_array(emb: str):
    new_array = []
    for row in emb.strip('[]').split('\n'):
        for dig in row.split():
            new_array.append(float(dig))
            
    return np.array(new_array).reshape(1, -1)


def get_vectors_name(attentions_types, use_bert, use_lmms):
    attentions_to_be_used = ['h-r', 'r-t', 'h-t', 'r-h', 't-r', 't-h'] 
    attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attentions_types[i] == 1])
    att_names = '_'.join(attentions_to_use)
    name = f'{att_names}'
    
    if use_bert:
        name += '_bert'
    
    if use_lmms:
        name += '_lmms'
        
    return name

def get_Xy_data(data_type, vectors_name):
    with open(f'./vectors/{data_type}_{vectors_name}.pkl', 'rb') as file:
        embeddings = pickle.load(file)
    
    embeddings_df = pd.DataFrame(embeddings, columns=['embedding', 'triplet', 'rel_label'])

    embs = np.array(list(embeddings_df.embedding.values)).squeeze()
    labels = embeddings_df.rel_label.values

    assert len(embs) == len(labels)
    
    return embs, labels

def train_lr_bin(X_train, labels_train, X_test, labels_test, vectors_name):
    y_train = np.array(['0' if label == '0' else '1' for label in labels_train]).reshape(-1, 1)
    y_test = np.array(['0' if label == '0' else '1' for label in labels_test]).reshape(-1, 1)

    lr_bin = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    with open(f'./logreg_models/lr_bin_{vectors_name}.pkl', 'wb') as file:
          pickle.dump(lr_bin, file)
            
    y_pred = lr_bin.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'./logreg_models/report_bin_{vectors_name}.csv')
    

def train_lr_multi(X_train, y_train, X_test, y_test, vectors_name):    
    #  убираем мусорный класс
    ros = RandomUnderSampler(sampling_strategy={'0': 0})
    
    X_tr_resampled, y_tr_resampled = ros.fit_resample(X_train, y_train)
    X_train = np.array(X_tr_resampled)
    y_train = np.array(y_tr_resampled)
    
    X_test_resampled, y_test_resampled = ros.fit_resample(X_test, y_test)
    X_test = np.array(X_test_resampled)
    y_test = np.array(y_test_resampled)

    lr_multi = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_train, y_train)

    with open(f'./logreg_models/lr_multi_{vectors_name}.pkl', 'wb') as file:
          pickle.dump(lr_multi, file)

    y_pred = lr_multi.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'./logreg_models/report_multi_{vectors_name}.csv')
