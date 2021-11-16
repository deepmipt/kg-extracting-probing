import os
import pandas as pd
from services_embeddings import get_train_test_data

combinations = [
#                ([1, 1, 0, 0, 0, 0], False, False),
#                ([1, 1, 0, 0, 0, 0], True, False),]
                ([1, 1, 0, 0, 0, 0], False, True),
                ([1, 1, 0, 0, 0, 0], True, True),
                ([1, 1, 1, 0, 0, 0], False, False),
                ([1, 1, 1, 0, 0, 0], True, False),
                ([1, 1, 1, 0, 0, 0], False, True),
                ([1, 1, 1, 0, 0, 0], True, True)
]
    
os.makedirs('./vectors/', exist_ok=True)

for comb in combinations:
    attentions_types, use_bert, use_lmms = comb
    get_train_test_data(attentions_types, use_bert, use_lmms)
    
    
