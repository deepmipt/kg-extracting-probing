from collections import OrderedDict
from nltk.tokenize import word_tokenize, MWETokenizer
import numpy as np
import torch
import re

from copy import copy
from collections import defaultdict


def create_mapping(sentence, return_pt=False, nlp = None, tokenizer=None):
    '''
    Create a mapping
    nlp: spacy model
    tokenizer: huggingface tokenizer
    '''
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
    
    return outputs, tokenid2word_mapping, token2id, sentence_mapping, noun_chunks


def create_mapping_target(sentence, target, return_pt=False, tokenizer=None):
    tokenizer_mwe = MWETokenizer(separator=" ")
    single_tokens = word_tokenize(sentence)

    for triplet in target:
        for phrase in triplet[:-1]:
            phrase_tokens = word_tokenize(phrase)
            if len(phrase_tokens) > 1:
                tokenizer_mwe.add_mwe(tuple(phrase_tokens))

    sentence_mapping = tokenizer_mwe.tokenize(single_tokens)

    token2id = {}
    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        if token not in token2id:
            token2id[token] = len(token2id)

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

    return outputs, tokenid2word_mapping, token2id, sentence_mapping


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
    new_index = []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    
    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))
    
    new_matrix = np.array(new_matrix)

    return new_matrix.T


def index2word(tokenid2word_mapping, token2id):
    tokens = []
    prev = -1
    for token_id in tokenid2word_mapping:
        if token_id == prev:
            continue

        tokens.append(token2id[token_id])
        prev = token_id

    return tokens
