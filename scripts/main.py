import os
import pandas as pd
from tqdm import tqdm 

from generate_embeddings import generate_new_embeddings
from services_logreg import (get_vectors_name, get_Xy_data, train_lr_bin, train_lr_multi)
from services_metrics_with_multi import (get_vectorname, load_lr_models, compute_csv_default)
import pickle

os.makedirs('./logreg_models_recomputed_sum/', exist_ok=True)
os.makedirs('./val_results_recomputed_sum/', exist_ok=True)

combinations = pd.read_excel('../data/meta/extract_kg_from_lms.xlsx')
combinations = combinations.dropna().astype(int).drop_duplicates().loc[2:,:].reset_index(drop=True)
combinations = combinations.iloc[3:,:6].drop_duplicates().reset_index(drop=True)

val_data = pd.read_csv('../data/train-val-test/valid.csv')

with open(f'./vectors_recomputed/valid_h-r_r-t_h-t_r-h_t-r_t-h.pkl', 'rb') as file:
    full_embeddings = pickle.load(file)

for comb in tqdm(combinations.itertuples(), total=len(combinations)):
    # set params
    attentions_types = list(comb)[1:]
    print('!!! combinations', attentions_types)

    # generate new embeddings
    generate_new_embeddings('train', attentions_types, False, False)    
    generate_new_embeddings('test', attentions_types, False, False)    
    
    # train models
    vectors_name = get_vectors_name(attentions_types, False, False)
    X_train, y_train = get_Xy_data('train', vectors_name)
    X_test, y_test   = get_Xy_data('test',  vectors_name)    
    train_lr_bin(X_train,   y_train, X_test, y_test, vectors_name)
    train_lr_multi(X_train, y_train, X_test, y_test, vectors_name)
    
    # compute metrics
    vectorname = get_vectorname(attentions_types, False, False)    
    lr_bin, lr_multi = load_lr_models(vectorname)
    compute_csv_default(val_data, lr_bin, lr_multi, attentions_types, False, False, full_embeddings, filename=f'res_{vectorname}')
