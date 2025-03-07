import os
import sys
import argparse
import importlib

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.running import run_dataset


def run_experiment(experiment_module: str, experiment_name: str, net_path: str, 
                   result_path: str, debug=0, threads=0, cascade_level=3):
    """Run experiment.
    args:
        experiment_module: Name of experiment module in the experiments/ folder.
        experiment_name: Name of the experiment function.
        debug: Debug level.
        threads: Number of threads.
    """
    expr_module = importlib.import_module('pytracking.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers, dataset = expr_func()
    trackers[0].results_dir = '{}'.format(result_path)
    trackers[0].cascade_level = cascade_level

   
    run_dataset(dataset, trackers, net_path, debug, threads)

def main():
    parser = argparse.ArgumentParser(description='Run tracker.')
    
    parser.add_argument('result_path', type=str)
    parser.add_argument('net_path', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('cascade_level', type=int)

    parser.add_argument('--cuda_number', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('--experiment_module', type=str, default='myexperiments', help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--debug', type=int, default=0)

    args = parser.parse_args()
    run_experiment(args.experiment_module, args.experiment_name, args.net_path,  
                   args.result_path,       args.debug,           args.threads ,   args.cascade_level)


if __name__ == '__main__':
    main()
