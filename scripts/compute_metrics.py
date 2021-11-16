import os
import pandas as pd
from services_metrics import (get_vectorname, load_lr_models, compute_csv_default)


combinations = [
#                 ([1, 1, 0, 0, 0, 0], True, False),
#                 ([1, 1, 0, 0, 0, 0], True, False),]
                 ([1, 1, 0, 0, 0, 0], False, True),
                 ([1, 1, 0, 0, 0, 0], True, True),
                 ([1, 1, 1, 0, 0, 0], False, False),
                 ([1, 1, 1, 0, 0, 0], True, False),
                 ([1, 1, 1, 0, 0, 0], False, True),
                 ([1, 1, 1, 0, 0, 0], True, True)
]

os.makedirs('./val_results/', exist_ok=True)

for comb in combinations:
    attentions_types, use_bert, use_lmms = comb
    vectorname = get_vectorname(attentions_types, use_bert, use_lmms)
    
    lr_bin, lr_multi = load_lr_models(vectorname)

    compute_csv_default(val_data, lr_bin, lr_multi, attentions_types, use_bert, use_lmms, filename=f'res_{vectorname}')