import pickle
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations, product
import os

import torch
from transformers import BertTokenizer, BertModel

import nltk
import fasttext as ft
from nltk.tokenize import sent_tokenize
import en_core_web_sm
from nltk.metrics.distance import jaccard_distance

import warnings
warnings.simplefilter('ignore')


nltk.download('averaged_perceptron_tagger')


string.punctuation += '’'
string.punctuation += '–'
nlp = en_core_web_sm.load()

ft_model = ft.load_model('../../../language-models-are-knowledge-graphs-pytorch/similarity/LMMS/cc.en.300.bin')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = bert_model.to(device)
bert_model = bert_model.eval()

encoder = BertModel.from_pretrained("bert-base-cased")
encoder = encoder.to(device)
encoder = encoder.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
RELATIONS = pd.read_csv('../data/meta/relations_trex.csv')
val_data = pd.read_csv('../data/train-val-test/valid.csv')

def get_bert_vector(text, rel=None):
    encoded_input = bert_tokenizer(text, return_tensors='pt').to(device)  
    output = bert_model(**encoded_input)
    logits = output[0].squeeze()[1:-1]
    
    default_return = torch.mean(logits, axis=0).detach().cpu().numpy()
    if not rel:
        return default_return
    
    encoded_rel = bert_tokenizer(rel, return_tensors='pt')

    tokens = bert_tokenizer.convert_ids_to_tokens(encoded_input['input_ids'].squeeze())[1:-1]
    
    rel_tokens = bert_tokenizer.convert_ids_to_tokens(encoded_rel['input_ids'].squeeze()[1:-1])
    indices = []
    for rel_tok in rel_tokens:
        try:
            r_t = tokens.index(rel_tok)
        except ValueError:
            try:
                r_t = tokens.index('##' + rel_tok)
            except:
                continue 
        indices.append(r_t)
    if len(indices):
        return torch.mean(logits[indices,:], axis=0).detach().cpu().numpy()

    return default_return


def vectorize_pred_rel(rel_pred, vectorized_dict):
    pred_vector = np.concatenate((ft_model.get_sentence_vector(rel_pred), vectorized_dict[rel_pred], vectorized_dict[rel_pred]))
    return pred_vector

def get_vectorname(attentions_types, use_bert, use_lmms):
    attentions_to_be_used = ['h-r', 'r-t', 'h-t', 'r-h', 't-r', 't-h'] 
    attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attentions_types[i] == 1])
    name = '_'.join(attentions_to_use)
    
    if use_bert:
        name += '_bert'
    
    if use_lmms:
        name += '_lmms'
        
    return name

def load_lr_models(vector_names):
    with open(f'./logreg_models/lr_multi_{vector_names}.pkl', 'rb') as file:
        lr_multi = pickle.load(file)

    with open(f'./logreg_models/lr_bin_{vector_names}.pkl', 'rb') as file:
        lr_bin = pickle.load(file)
    
    return lr_bin, lr_multi


def get_title(relation_id):
    return RELATIONS[RELATIONS.relation==relation_id].title.values[0]


def process(sentence, tokenizer, nlp, return_pt=True):
    doc = nlp(sentence)
    tokens = list(doc)

    chunk2id = {}

    start_chunk = []
    end_chunk = []
    noun_chunks = []
    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)
        start_chunk.append(chunk.start)
        end_chunk.append(chunk.end)

    sentence_mapping = []
    token2id = {}
    mode = 0 # 1 in chunk, 0 not in chunk       
    chunk_id = 0
    for idx, token in enumerate(doc):
        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            if sentence_mapping[-1] not in token2id:
                token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token.text)
            if sentence_mapping[-1] not in token2id:
                token2id[sentence_mapping[-1]] = len(token2id)


    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)
    
    return outputs, tokenid2word_mapping, token2id, noun_chunks, sentence_mapping


def compress_attention(attention, tokenid2word_mapping, operator=np.mean):

    new_index = []
    
    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index=  []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    
    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))
    
    new_matrix = np.array(new_matrix)
    
    return new_matrix.T


def get_outputs(sentence, tokenizer, encoder, nlp, use_cuda=True):

    tokenizer_name = str(tokenizer.__str__)
    inputs, tokenid2word_mapping, token2id, tokens, sentence_mapping = process(sentence, nlp=nlp, tokenizer=tokenizer, return_pt=True)
    id2token = {value: key for key, value in token2id.items()}
    for key in inputs.keys():
        inputs[key] = inputs[key].cuda()
    outputs = encoder(**inputs, output_attentions=True)
    
    return outputs[2], tokenid2word_mapping, token2id, sentence_mapping


def get_embeddings(sentence, attentions_types, use_bert, use_lmms):
    rel_pos = ['NN', 'NNP', 'NNS', 'JJR', 'JJS', 'MD', 'POS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    head_tail_pos = ['NN', 'NNP', 'NNS', 'PRP']

    use_cuda = True    
    att, tokenid2word_mapping, token2id, sentence_mapping = get_outputs(sentence, tokenizer, encoder, nlp, use_cuda=use_cuda)
    
    new_matr = []
    
    for layer in att:
        for head in layer.squeeze():
            attn = head.cpu()
            attention_matrix = attn.detach().numpy()
            attention_matrix = attention_matrix[1:-1, 1:-1]
            
            merged_attention = compress_attention(attention_matrix, tokenid2word_mapping)
            
            new_matr.append(merged_attention)
    
    new_matr = np.stack(new_matr)
    
    words = [token for token in sentence_mapping if token not in string.punctuation]
    
    nn_words = [word for word in words if nltk.pos_tag([word])[0][1] in head_tail_pos]
    other_words = [word for word in words if nltk.pos_tag([word])[0][1] in rel_pos]
    
    triplets = [triplet for triplet in list(product(nn_words, nn_words, other_words)) 
                if triplet[0] != triplet[1] and triplet[0] != triplet[2] and triplet[1] != triplet[2]]
    
    rel_toks = set([triplet[2] for triplet in triplets])
    vectorized_dict = {rel: get_bert_vector(rel, sentence) for rel in rel_toks}
    
    sent_embeddings = []
    
    for triplet in triplets:
       
        head_ind = sentence_mapping.index(triplet[0])
        tail_ind = sentence_mapping.index(triplet[1])
        rel_ind = sentence_mapping.index(triplet[2])   

        head_rel_emb = new_matr[:, head_ind, rel_ind]
        rel_tail_emb = new_matr[:, rel_ind, tail_ind]
        head_tail_emb = new_matr[:, head_ind, tail_ind]
        rel_head_emb = new_matr[:, rel_ind, head_ind]
        tail_rel_emb = new_matr[:, tail_ind, rel_ind]
        tail_head_emb = new_matr[:, tail_ind, head_ind]
        
        attentions_to_be_used = [head_rel_emb, rel_tail_emb, head_tail_emb, rel_head_emb, tail_rel_emb, tail_head_emb] 
        attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attentions_types[i] == 1])

        triplet_emb = np.concatenate(attentions_to_use, axis=0).squeeze()
        sentence = ' '.join(sentence_mapping)
        
        if use_bert:
            extra_embedding = vectorized_dict[triplet[2]]
            triplet_emb = np.concatenate((triplet_emb, extra_embedding))
            
        if use_lmms:
            extra_embedding = vectorize_pred_rel(triplet[2], vectorized_dict)
            triplet_emb = np.concatenate((triplet_emb, extra_embedding))
        sent_embeddings.append((triplet_emb, triplet))
        
    return sent_embeddings


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
    
    return prediction, predicted_labels


def get_predictions(sentence, attentions_types, use_bert, use_lmms, lr_bin, lr_multi, threshold_bin=0.5):
    pred_list = []
    emb_sent = get_embeddings(sentence, attentions_types, use_bert, use_lmms)
    for emb in tqdm(emb_sent, total=len(emb_sent), leave=False):
        try:
            binary_conf = lr_bin.predict_proba(emb[0].reshape(1, -1))[0][1]
        except:
            print(emb, sentence, emb_sent)
            raise Exception
        if binary_conf > threshold_bin:
            predicted_label = list(lr_multi.predict(emb[0].reshape(1, -1)))[0]
            triplet = emb[1]
            pred_list.append((predicted_label, triplet, binary_conf))
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


def compute_logreg_nm(dataset, attentions_types, use_bert, use_lmms, lr_bin, lr_multi):
    fp, tp, fn = 0, 0, 0
    tp_predicts_dict = {}
    fp_predicts_dict = {}
    for row in tqdm(dataset.itertuples(), total=dataset.shape[0], leave=False):
        try:
            predictions = get_predictions(row.text, attentions_types, use_bert, use_lmms, lr_bin, lr_multi, threshold_bin=0.7)
        except IndexError:
            continue
        
        filtered_predictions, predicted_labels = deduplication(predictions)
        target_triplets = [target[:3] for target in eval(row.target)]
        tp_predicts = []
        fp_predicts = []

        for predict in filtered_predictions:
            
            score_bool = compare_triplets(target_triplets, predict)

            if score_bool:
                tp_predicts.append(predict)
                tp += 1
            else:
                fp_predicts.append(predict)
                fp += 1
                
        if len(tp_predicts):
            tp_predicts_dict[row.text] = tp_predicts
        
        if len(fp_predicts):
            fp_predicts_dict[row.text] = fp_predicts
        
        for target in target_triplets:
            score_bool = compare_triplets(filtered_predictions, target)
            if not score_bool:
                fn += 1

    try:        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1, tp_predicts_dict, fp_predicts_dict
    
    except ZeroDivisionError:
        return 0, 0, 0, tp_predicts_dict, fp_predicts_dict


def compute_csv_default(dataset, lr_bin, lr_multi, attentions_types, use_bert, use_lmms, filename='filename'):
    rels, prs, rcls, f1s = [], [], [], []
    sizes, labels, tp_preds, fp_preds = [], [], [], []

    for rel in tqdm(sorted(lr_multi.classes_, key=lambda x: int(x[1:]))):
        mono_tr_subset = dataset[dataset.rel == rel]

        if not mono_tr_subset.empty:
            label = get_title(rel)
            size = mono_tr_subset.shape[0]
            precision, recall, f1, tp_pred_dict, fp_pred_dict = compute_logreg_nm(mono_tr_subset, attentions_types, use_bert, use_lmms, lr_bin, lr_multi)

            rels.append(rel)
            prs.append(precision)
            rcls.append(recall)
            f1s.append(f1)
            sizes.append(size)
            labels.append(label)
            tp_preds.append(tp_pred_dict)
            fp_preds.append(fp_pred_dict)
            
    rels.append('Average')
    prs.append(np.mean(prs))
    rcls.append(np.mean(rcls))
    f1s.append(np.mean(f1s))
    sizes.append(np.mean(sizes))
    labels.append('-')
    tp_preds.append('-')
    fp_preds.append('-')

    scoring_result = pd.DataFrame({'rel': rels,
                                   'label': labels,
                                   'size': sizes, 
                                   'precision': prs, 
                                   'recall': rcls, 
                                   'f1': f1s,
                                   'tps': tp_preds,
                                   'fps': fp_preds})
    

    scoring_result.to_csv(f'./val_results/{filename}.csv', index=False)