import csv
import pickle
import random
import string
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import product
from IPython.display import display

import nltk
import spacy
# import fasttext as ft
import en_core_web_sm

import torch
from transformers import BertModel, BertTokenizer

import warnings
from utils import *

warnings.simplefilter('ignore')


string.punctuation += '’'
string.punctuation += '–'
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

ft_model = ft.load_model('../../../language-models-are-knowledge-graphs-pytorch/similarity/LMMS/cc.en.300.bin')
#ft_model = ft.load_model('path/to/model/cc.en.300.bin')
nlp = en_core_web_sm.load()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = bert_model.eval()
bert_model = bert_model.to(device)

encoder = BertModel.from_pretrained("bert-base-cased")
encoder = encoder.eval()
encoder = encoder.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


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


def get_embs_for_triplets(triplets, sentence_mapping, attention, attentions_types, with_label=False, use_bert=False, use_lmms=False):
    new_triplets = []
    
    for triplet in triplets:
        #  tokenize target the same way as sentence to avoid index error
        if ',' in triplet[0] or "'" in triplet[0]:
            head = ' '.join(word_tokenize(triplet[0]))
        else:
            head = triplet[0]
            
        if ',' in triplet[1] or "'" in triplet[1]:
            tail = ' '.join(word_tokenize(triplet[1]))
        else:
            tail = triplet[1]
        
        if ',' in triplet[2] or "'" in triplet[2]:
            rel = ' '.join(word_tokenize(triplet[2]))
        else:
            rel = triplet[2]
        
        if with_label:
            rel_label = triplet[3]
            new_triplets.append((head, tail, rel, rel_label))
        else:
            new_triplets.append((head, tail, rel))

    
    sent_embeddings = []
    sentence = ' '.join(sentence_mapping)
    rel_toks = set([triplet[2] for triplet in triplets])
    
    if use_bert or use_lmms:
        vectorized_dict = {rel: get_bert_vector(rel, sentence) for rel in rel_toks}
    
    
    for triplet in new_triplets:
        #  tokenize target the same way as sentence to avoid index error
        if with_label:
            head, tail, rel, rel_label = triplet
        else:
            head, tail, rel = triplet

        try:
            if head in sentence_mapping and tail in sentence_mapping and rel in sentence_mapping:
                #  get head, tail, rel indices in the matrix (len(sentence_mapping) == len(att_matrix)) 
                head_ind = sentence_mapping.index(head)
                tail_ind = sentence_mapping.index(tail)
                rel_ind = sentence_mapping.index(rel)   
                
                #  get vector of attention from every head
                head_rel_emb = attention[:, head_ind, rel_ind]
                rel_tail_emb = attention[:, rel_ind, tail_ind]
                head_tail_emb = attention[:, head_ind, tail_ind]
                rel_head_emb = attention[:, rel_ind, head_ind]
                tail_rel_emb = attention[:, tail_ind, rel_ind]
                tail_head_emb = attention[:, tail_ind, head_ind]
                
                #  choose only needed vectors
                attentions_to_be_used = [head_rel_emb, rel_tail_emb, head_tail_emb, rel_head_emb, tail_rel_emb, tail_head_emb] 
                attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attentions_types[i] == 1])

                #  concat chosen vectors into one
                triplet_emb = np.concatenate(attentions_to_use, axis=0).squeeze()
                if use_bert:
                    extra_embedding = vectorized_dict[rel]
                    triplet_emb = np.concatenate((triplet_emb, extra_embedding))
                    
                if use_lmms:
                    extra_embedding = vectorize_pred_rel(rel, vectorized_dict)
                    triplet_emb = np.concatenate((triplet_emb, extra_embedding))
                    
                #  add label if 'train' 
                if with_label:
                    sent_embeddings.append((triplet_emb, triplet, rel_label))
                else:
                    sent_embeddings.append((triplet_emb, triplet))
            else:
                pass
        except:
            pass

    return sent_embeddings


def return_embeddings(sentence, attentions_types, tokenizer, encoder, nlp, use_cuda, use_bert, use_lmms, target=None, mode='train'):
    
    tokenizer_name = str(tokenizer.__str__)
    rel_pos = ['NN', 'NNP', 'NNS', 'MD', 'POS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    head_tail_pos = ['NN', 'NNP', 'NNS', 'PRP']

    if mode == 'train':
        #  to process data with rel labels from dataset (pass target)
        inputs, tokenid2word_mapping, token2id, sentence_mapping = create_mapping_target(sentence, 
                                                                                         target, 
                                                                                         return_pt=True, 
                                                                                         tokenizer=tokenizer)
    
    else:
        #  to process data to predict
        inputs, tokenid2word_mapping, token2id, sentence_mapping, noun_chunks = create_mapping(sentence, 
                                                                                               return_pt=True, 
                                                                                               nlp=nlp,
                                                                                               tokenizer=tokenizer)

    with torch.no_grad():
        if use_cuda:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        try:
            outputs = encoder(**inputs, output_attentions=True)
        except RuntimeError:
            print(sentence_mapping)
            return []

    attn = outputs[2]   

    new_matr = []
    
    for layer in attn:
        for head in layer.squeeze():
            if use_cuda:
                attn = head.cpu()
            else:
                attn = head
            attention_matrix = attn.detach().numpy()
            attention_matrix = attention_matrix[1:-1, 1:-1]
            merged_attention = compress_attention(attention_matrix, tokenid2word_mapping)
            new_matr.append(merged_attention)

    new_matr = np.array(new_matr)
    
    #  get candidates for head, tail and rel
    words = [token for token in sentence_mapping if token not in string.punctuation]
    nn_words = [word for word in words if nltk.pos_tag([word])[0][1] in head_tail_pos]
    other_words = [word for word in words if nltk.pos_tag([word])[0][1] in rel_pos]
    
    sent_embeddings = []

    if mode == 'train':
        #  get candidate triplets (in this case - for 'garbage class')
        triplets = [triplet for triplet in list(product(nn_words, nn_words, other_words)) 
                        if triplet[0] != triplet[1] and triplet[0] != triplet[2] and triplet[1] != triplet[2] and triplet not in target]
        other_triplets = [(t[0], t[1], t[2], '0') for t in triplets]
        
        #  get embeddings for 'garbage' class
        try:
            sent_embeddings.extend(get_embs_for_triplets(random.choices(other_triplets, k=len(target)), sentence_mapping, new_matr, attentions_types, 
                                                        with_label=True, 
                                                        use_bert=use_bert,
                                                        use_lmms=use_lmms))
        except IndexError:
            pass
        
        #  get embeddings for target class
        sent_embeddings.extend(get_embs_for_triplets(target, sentence_mapping, new_matr, attentions_types, 
                                                     with_label=True, 
                                                     use_bert=use_bert, 
                                                     use_lmms=use_lmms))

        
    else:
        #  get candidate triplets from the sentence
        triplets = [triplet for triplet in list(product(nn_words, nn_words, other_words)) 
                      if triplet[0] != triplet[1] and triplet[0] != triplet[2] and triplet[1] != triplet[2]]
        
        #  get embeddings for candidate triplets (to be classified further)
        sent_embeddings.extend(get_embs_for_triplets(triplets, sentence_mapping, new_matr, attentions_types, with_label=False))
    
    return sent_embeddings


def get_filename(data_type, attentions_types, use_bert, use_lmms):
    attentions_to_be_used = ['h-r', 'r-t', 'h-t', 'r-h', 't-r', 't-h'] 
    attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attentions_types[i] == 1])
    att_names = '_'.join(attentions_to_use)
    name = f'{data_type}_{att_names}'
    
    if use_bert:
        name += '_bert'
    
    if use_lmms:
        name += '_lmms'
        
    return name


def get_embeddings_corpus(data_type, attentions_types, use_bert, use_lmms):  
    use_cuda = False
    data = pd.read_csv(f'../data/train-val-test/{data_type}.csv', header=0)
    all_embeddings = []
    
    if data_type == 'train' or data_type == 'test':
        mode = 'train'
    else:
        mode = 'valid'
        
    for ind, row in tqdm(data.iterrows(), total=data.shape[0], desc=f'Getting embeddings for {data_type}'):
        text = row['text']
        target = eval(row['target'])
        embeddings_text = return_embeddings(text, 
                                       attentions_types, 
                                       tokenizer, 
                                       encoder, 
                                       nlp, 
                                       use_cuda, 
                                       use_bert=use_bert, 
                                       use_lmms=use_lmms, 
                                       target=target, 
                                       mode=mode)
        
        all_embeddings.extend(embeddings_text)
        
        file_name = get_filename(data_type, attentions_types, use_bert, use_lmms)
        
#         df = pd.DataFrame(embeddings_text)
#         df.to_csv(f'./vectors/{csv_name}.csv', mode='a', header=False, index=False)
    
    with open(f'./vectors/{file_name}.pkl', 'wb') as file:
        pickle.dump(all_embeddings, file)
        
        
def get_train_test_data(attentions_types, use_bert, use_lmms):
    get_embeddings_corpus('train', attentions_types, use_bert, use_lmms)
    get_embeddings_corpus('test', attentions_types, use_bert, use_lmms)
