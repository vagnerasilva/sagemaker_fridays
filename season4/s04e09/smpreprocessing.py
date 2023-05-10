import argparse,os,ast

import subprocess, sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
install('s3fs')
install('mxnet')
install('autogluon')

import pandas as pd
import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.features.feature_metadata import FeatureMetadata

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--eval-metric', type=str, default='roc_auc')
    parser.add_argument('--presets', type=str, default='best_quality')
    parser.add_argument('--excluded-model-types', type=str, default='[]')

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    excluded_model_types=ast.literal_eval(args.excluded_model_types)
    
    dataset_path = os.path.join('/opt/ml/processing/input', args.filename)    
    train_data = TabularDataset(dataset_path)
    train_data = train_data.sample(n=args.num_samples, random_state=0)
    
    feature_metadata = FeatureMetadata({
        'product_category_1':'category',
        'product_category_2':'category',
        'user_group_id':'category',
        'age_level':'category',
        'user_depth':'category',
        'city_development_index':'category',
        'session_id':'category',
        'user_id':'category',
        'campaign_id':'category',
        'webpage_id':'category',
        'gender':'object',
        'var_1':'int',
        'product':'object',
        'DateTime':'datetime64'
    })
    
    predictor = TabularPredictor(
        label='is_click', 
        eval_metric=args.eval_metric,
        path='/opt/ml/processing/output'
    )
    
    predictor.fit(
        train_data, 
        presets=args.presets,
        auto_stack=True,
        num_bag_folds=10,
        num_bag_sets=20,
        feature_metadata=feature_metadata,
        excluded_model_types=excluded_model_types
    )
    
    predictor.leaderboard() 
    