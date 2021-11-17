import os
import pandas as pd

from services_embeddings import get_train_test_data
from services_logreg import (get_vectors_name, get_Xy_data, train_lr_bin, train_lr_multi)
from services_metrics import (get_vectorname, load_lr_models, compute_csv_default)


os.makedirs('./logreg_models/', exist_ok=True)
os.makedirs('./vectors/', exist_ok=True)
os.makedirs('./val_results/', exist_ok=True)

combinations = pd.read_excel('../data/meta/extract_kg_from_lms.xlsx')
combinations = combinations.dropna().astype(int).drop_duplicates().loc[2:,:]

for comb in combinations.itertuples():
    # set params
    attentions_types = (comb.head_rel, comb.rel_tail, comb.head_tail, comb.rel_head, comb.tail_rel, comb.tail_head)
    use_bert, use_lmms = comb.use_bert, comb._8
    
    # compute vectors
    get_train_test_data(attentions_types, use_bert, use_lmms)
    
    # train models
    vectors_name = get_vectors_name(attentions_types, use_bert, use_lmms)
    X_train, y_train = get_Xy_data('train', vectors_name)
    X_test, y_test = get_Xy_data('test', vectors_name)
    train_lr_bin(X_train, y_train, X_test, y_test, vectors_name)
    train_lr_multi(X_train, y_train, X_test, y_test, vectors_name)

    # compute metrics
    vectorname = get_vectorname(attentions_types, use_bert, use_lmms)    
    lr_bin, lr_multi = load_lr_models(vectorname)
    compute_csv_default(val_data, lr_bin, lr_multi, attentions_types, use_bert, use_lmms, filename=f'res_{vectorname}')
