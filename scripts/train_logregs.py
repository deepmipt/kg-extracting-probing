import os
import pandas as pd
from services_logreg import (get_vectors_name, get_Xy_data, train_lr_bin, train_lr_multi)

combinations = [
#                 ([1, 1, 0, 0, 0, 0], False, False),
#                 ([1, 1, 0, 0, 0, 0], True, False),]
                ([1, 1, 0, 0, 0, 0], False, True),
                ([1, 1, 0, 0, 0, 0], True, True),
                ([1, 1, 1, 0, 0, 0], False, False),
                ([1, 1, 1, 0, 0, 0], True, False),
                ([1, 1, 1, 0, 0, 0], False, True),
                ([1, 1, 1, 0, 0, 0], True, True)
]

os.makedirs('./logreg_models/', exist_ok=True)

for comb in combinations:
    attentions_types, use_bert, use_lmms = comb
    vectors_name = get_vectors_name(attentions_types, use_bert, use_lmms)
    
    X_train, y_train = get_Xy_data('train', vectors_name)
    X_test, y_test = get_Xy_data('test', vectors_name)
    
    train_lr_bin(X_train, y_train, X_test, y_test, vectors_name)
    train_lr_multi(X_train, y_train, X_test, y_test, vectors_name)