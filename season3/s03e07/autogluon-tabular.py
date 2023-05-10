import argparse, os, subprocess, sys

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

def pip_uninstall(package):
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", package])
    
pip_uninstall('typing')            # Fixes Python weirdness in the container. Meh.
pip_install('mxnet<2.0.0')
pip_install('autogluon==0.2.0')

import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    # preprocessing arguments
    parser.add_argument('--filename', type=str)
    parser.add_argument('--label', type=str)
    parser.add_argument('--eval-metric', type=str)
    parser.add_argument('--time-limit', type=int, default=3600)
    parser.add_argument('--presets', type=str, default='best_quality')

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.filename
    label = args.label
    eval_metric = args.eval_metric
    time_limit = args.time_limit
    presets = args.presets

    # Load dataset
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    train_data = TabularDataset(input_data_path)

    # Configure predictor
    predictor = TabularPredictor(
        label=label, 
        eval_metric=eval_metric,
        path='/opt/ml/processing/output/'
    )
    
    # Launch AutoGluon job
    predictor.fit(
        train_data, 
        time_limit=time_limit, 
        presets=presets
    )
    
    leaderboard = predictor.leaderboard()
