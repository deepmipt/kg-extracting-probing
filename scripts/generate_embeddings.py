import numpy as np
import pickle


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


def get_filename(data_type, attention_types, use_bert, use_lmms):
    attentions_to_be_used = ['h-r', 'r-t', 'h-t', 'r-h', 't-r', 't-h'] 
    attentions_to_use = tuple([att for i, att in enumerate(attentions_to_be_used) if attention_types[i] == 1])
    att_names = '_'.join(attentions_to_use)
    name = f'{data_type}_{att_names}'
    
    if use_bert:
        name += '_bert'
    
    if use_lmms:
        name += '_lmms'
        
    return name


def generate_new_embeddings(data_type, attention_types, use_bert, use_lmms):
    with open(f'./vectors_recomputed_sum/{data_type}_h-r_r-t_h-t_r-h_t-r_t-h.pkl', 'rb') as file:
        full_embeddings = pickle.load(file)
        
    new_vectors_name = get_filename(data_type, attention_types, use_bert, use_lmms)
    new_embeddings = []
    
    for full_emb in full_embeddings:
        new_embeddings.append(generate_one_embedding(full_emb, data_type, attention_types, use_bert, use_lmms))
    
    with open(f'./vectors_recomputed_sum/{new_vectors_name}.pkl', 'wb') as file:
         pickle.dump(new_embeddings, file)
