import pickle
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations, product
import os

import torch
import nltk
from nltk.tokenize import sent_tokenize
from nltk.metrics.distance import jaccard_distance


import warnings
warnings.simplefilter('ignore')

RELATIONS = pd.read_csv('../data/meta/relations_trex.csv')


def get_vectorname(attention_types, use_bert, use_lmms):
    attentions_to_be_used = ['h-r', 'r-t', 'h-t', 'r-h', 't-r', 't-h'] 
    attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attention_types[i] == 1])
    name = '_'.join(attentions_to_use)
    
    if use_bert:
        name += '_bert'
    
    if use_lmms:
        name += '_lmms'
        
    return name


def load_lr_models(vector_names):
    with open(f'./logreg_models_recomputed_sum/lr_multi_{vector_names}.pkl', 'rb') as file:
        lr_multi = pickle.load(file)

    with open(f'./logreg_models_recomputed_sum/lr_bin_{vector_names}.pkl', 'rb') as file:
        lr_bin = pickle.load(file)
    
    return lr_bin, lr_multi


def get_title(relation_id):
    return RELATIONS[RELATIONS.relation==relation_id].title.values[0]


def generate_one_embedding(full_triplet_embedding: tuple, data_type, attention_types, use_bert, use_lmms):
    full_embedding = full_triplet_embedding[0]
    triplet = full_triplet_embedding[1]
    if data_type == 'train' or data_type == 'test':
        rel_label = full_triplet_embedding[2]
    
    head_rel_emb = full_embedding[0:144]
    rel_tail_emb = full_embedding[144:288]
    head_tail_emb = full_embedding[288:432]
    rel_head_emb = full_embedding[432:576]
    tail_rel_emb = full_embedding[576:720]
    tail_head_emb = full_embedding[720:864]
    
    bert_emb = full_embedding[864:1632]
    lmms_emb = full_embedding[1632:3468]
    
    attentions_to_be_used = [head_rel_emb, rel_tail_emb, head_tail_emb, rel_head_emb, tail_rel_emb, tail_head_emb] 
    attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attention_types[i] == 1])
    
    #  concat chosen vectors into one
    if len(attentions_to_use):
        triplet_emb = np.concatenate(attentions_to_use, axis=0).squeeze()
    else:
        triplet_emb = np.array([])
            
    if use_bert:
        triplet_emb = np.concatenate((triplet_emb, bert_emb))
                    
    if use_lmms:
        triplet_emb = np.concatenate((triplet_emb, lmms_emb))
        
    if data_type == 'train' or data_type == 'test':
        new_embedding = (triplet_emb, triplet, rel_label)
    else:
        new_embedding = (triplet_emb, triplet)
        
    return new_embedding


def get_embeddings(text_embeddings, attention_types, use_bert, use_lmms):
    new_text_embeddings = []
    
    for full_emb in text_embeddings:
        new_emb = generate_one_embedding(full_emb, 'valid', attention_types, use_bert, use_lmms)
        new_text_embeddings.append(new_emb)
        
    return new_text_embeddings


def deduplication(pred_list):
    pred_max_conf = {}
    filtered_pred = {}

    for ind, pred in enumerate(pred_list):
        pred_triplet = (pred[1][0], pred[1][1])

        if pred_triplet not in filtered_pred.keys():
            pred_max_conf[pred_triplet] = pred[2]
            filtered_pred[pred_triplet] = pred

        elif pred_triplet in filtered_pred and pred[2] > pred_max_conf[pred_triplet]:
            pred_max_conf[pred_triplet] = pred[2]
            filtered_pred[pred_triplet] = pred
    
    sorted_pred = sorted(list(filtered_pred.values()), key=lambda x: x[2], reverse=True)
    prediction = [el[1] for el in sorted_pred]
    predicted_labels = [el[0] for el in sorted_pred]
    multi_confidences = [el[3] for el in sorted_pred]
    
    return prediction, predicted_labels, multi_confidences


def get_predictions(text_embeddings, attention_types, use_bert, use_lmms, lr_bin, lr_multi, threshold_bin=0.7, threshold_multi=0.2):
    pred_list = []
    new_text_embeddings = get_embeddings(text_embeddings, attention_types, use_bert, use_lmms)
    for emb in new_text_embeddings:

        binary_conf = lr_bin.predict_proba(emb[0].reshape(1, -1))[0][1]
            
        if binary_conf > threshold_bin:
            multi_conf = max(lr_multi.predict_proba(emb[0].reshape(1, -1))[0])
            if multi_conf > threshold_multi:
                predicted_label = list(lr_multi.predict(emb[0].reshape(1, -1)))[0]
                triplet = emb[1]
                pred_list.append((predicted_label, triplet, binary_conf, multi_conf))
    return pred_list


def compare_triplets(targets, predict, dist_thresh=0.4):
    compare_result = []
    for target in targets:
        sub_compare = []
        for target, predict_ in zip(target, predict):
            answer =  False
            dist = jaccard_distance(set(target.lower()), set(predict_.lower()))
            if predict_ in target or dist < dist_thresh or target in predict_:
                answer = True
            sub_compare.append(answer)
        sub_compare = all(sub_compare)
        compare_result.append(sub_compare)
    return any(compare_result)    


def compute_logreg_nm(rel, rel_embeddings, subset_target, attention_types, use_bert, use_lmms, lr_bin, lr_multi, threshold_bin, threshold_multi):
    fp, tp, fn, rel_pred_count = 0, 0, 0, 0
    tp_predicts_dict = {}
    fp_predicts_dict = {}
    for idx, text_embeddings in enumerate(rel_embeddings):
#         try:
        predictions = get_predictions(text_embeddings, attention_types, use_bert, use_lmms, lr_bin, lr_multi, threshold_bin, threshold_multi)
#         except IndexError:
#             print(idx)
#             continue
        filtered_predictions, predicted_labels, multi_confidences = deduplication(predictions)
        target_triplets = [target[:3] for target in subset_target[idx]]
        tp_predicts = []
        fp_predicts = []
        
        for i, predict in enumerate(filtered_predictions):
            
            score_bool = compare_triplets(target_triplets, predict)

            if score_bool:
                tp_predicts.append((predict[0], predict[1], predict[2], predicted_labels[i], multi_confidences[i]))
                tp += 1
                
                if predicted_labels[i] == rel:
                    rel_pred_count += 1
            else:
                fp_predicts.append((predict[0], predict[1], predict[2], predicted_labels[i], multi_confidences[i]))
                fp += 1
                
        if len(tp_predicts):
            tp_predicts_dict[idx] = tp_predicts
        
        if len(fp_predicts):
            fp_predicts_dict[idx] = fp_predicts
        
        for target in target_triplets:
            score_bool = compare_triplets(filtered_predictions, target)
            if not score_bool:
                fn += 1

    try:        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
#         rel_pred_accuracy = rel_pred_count / tp
        return precision, recall, f1, tp_predicts_dict, fp_predicts_dict, rel_pred_count, tp
    
    except ZeroDivisionError:
        return 0, 0, 0, tp_predicts_dict, fp_predicts_dict, 0, 0


def compute_csv_default(dataset, lr_bin, lr_multi, attention_types, use_bert, use_lmms, full_embeddings, filename='filename', threshold_bin=0.7, threshold_multi=0.2):        

    rels, prs, rcls, f1s = [], [], [], []
    sizes, labels, tp_preds, fp_preds, rel_acc = [], [], [], [], []
    rel_pred_all, tp_count = 0, 0

    for rel in tqdm(sorted(lr_multi.classes_, key=lambda x: int(x[1:]))):
        mono_tr_subset = dataset[dataset.rel == rel]

        if not mono_tr_subset.empty:
            label = get_title(rel)
            size = mono_tr_subset.shape[0]
            subset_indx = mono_tr_subset.index
            subset_target = mono_tr_subset.target.apply(eval).values
            rel_embeddings = np.array(full_embeddings)[subset_indx]
            assert len(subset_target) == len(rel_embeddings)
            
            precision, recall, f1, tp_pred_dict, fp_pred_dict, rel_pred_count, tp = compute_logreg_nm(rel, rel_embeddings, subset_target, attention_types, use_bert, use_lmms, lr_bin, lr_multi, threshold_bin, threshold_multi)

            rels.append(rel)
            prs.append(precision)
            rcls.append(recall)
            f1s.append(f1)
            sizes.append(size)
            labels.append(label)
            tp_preds.append(tp_pred_dict)
            fp_preds.append(fp_pred_dict)
            if tp != 0:
                rel_acc.append(rel_pred_count / tp)
            else:
                rel_acc.append(0)
            
            tp_count += tp
            rel_pred_all += rel_pred_count
            
    rels.append('Average')
    prs.append(np.mean(prs))
    rcls.append(np.mean(rcls))
    f1s.append(np.mean(f1s))
    sizes.append(np.mean(sizes))
    labels.append('-')
    tp_preds.append('-')
    fp_preds.append('-')
    if tp_count == 0:
        rel_acc.append(0)
    else:
        rel_acc.append(rel_pred_all / tp_count)

    scoring_result = pd.DataFrame({'rel': rels,
                                   'label': labels,
                                   'size': sizes, 
                                   'precision': prs, 
                                   'recall': rcls, 
                                   'f1': f1s,
                                   'tps': tp_preds,
                                   'fps': fp_preds,
                                   'rel_acc': rel_acc})
    
    scoring_result.to_csv(f'./val_results_recomputed_sum/{filename}.csv', index=False)
