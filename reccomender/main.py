# -*- coding: cp1251 -*-
import sys
import numpy as np

# ПРИНУДИТЕЛЬНЫЙ ПОЛНЫЙ ПАТЧ ДЛЯ СОВМЕСТИМОСТИ NUMPY 2.0 + RECBOLE
# Добавляем абсолютно все удаленные типы, включая unicode
deprecated_types = {
    'float_': np.float64,
    'float': np.float64,
    'complex_': np.complex128,
    'complex': np.complex128,
    'bool_': np.bool,
    'int_': np.int64,
    'int': np.int64,
    'unicode_': np.str_,
    'unicode': np.str_
}

for attr_name, target_type in deprecated_types.items():
    if not hasattr(np, attr_name):
        setattr(np, attr_name, target_type)
    sys.modules[f'numpy.{attr_name}'] = target_type

# Теперь безопасно импортируем RecBole
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer
from recbole.utils.case_study import full_sort_topk

def main():
    config = Config(model='SLIMElastic', dataset='movies', config_file_list=['config.json'])
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, dataset).to(config['device'])
    trainer = get_trainer(config['model'], config['model_type'])(config, model)

    print("\n--- Training SLIMElastic ---")
    trainer.fit(train_data, valid_data, show_progress=False)
    print("--- Training complete ---")

    target_user_external_id = "1" 
    
    model.eval()
    try:
        uid_idx = dataset.token2id(dataset.uid_field, target_user_external_id)
        
        topk_scores, topk_iid_list = full_sort_topk(
            uid_series=[uid_idx], 
            model=model, 
            test_data=test_data, 
            k=5, 
            device=config['device']
        )
        
        recommended_item_ids = topk_iid_list.cpu().numpy()
        external_item_ids = dataset.id2token(dataset.iid_field, recommended_item_ids)
        
        print(f"\n[RESULT] Top-5 recommendations for user '{target_user_external_id}':")
        for i, item_id in enumerate(external_item_ids, 1):
            print(f"{i}. Item ID: {item_id}")
            
    except KeyError:
        print(f"\n[ERROR] User ID '{target_user_external_id}' not found in the dataset.")

if __name__ == '__main__':
    main()

