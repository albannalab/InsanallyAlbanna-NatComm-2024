import os
import json
import h5py
import pandas as pd
from numpy import copy

RESULTS_DIR = './results'
SAVED_DIR = './example networks'

def make_directories_if_needed(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    pass


def check_if_file_exists(file_name, run_name, directory):
    if run_name not in os.listdir(os.path.join(directory)):
        print(f'{run_name}: Directory not found.')
        raise FileNotFoundError
    if file_name not in os.listdir(os.path.join(directory, run_name)):
        print(f'{run_name}: "{file_name}" not found.')
        raise FileNotFoundError
    pass


def save_params(run_name, params, directory=RESULTS_DIR, file_name='params.json'):
    directory_name = os.path.join(directory, run_name)
    make_directories_if_needed(directory_name)

    # Saving parameters as json
    with open(os.path.join(directory_name, file_name), 'w') as params_file:
        json.dump(params, params_file)
    pass


def save_convergence(run_name, convergence, directory=RESULTS_DIR, file_name='convergence.json'):
    directory_name = os.path.join(directory, run_name)
    make_directories_if_needed(directory_name)

    # Saving parameters as json
    with open(os.path.join(directory_name, file_name), 'w') as convergence_file:
        json.dump(convergence, convergence_file)
    pass

def save_meta(run_name, meta, directory=RESULTS_DIR, file_name='metadata.json'):
    directory_name = os.path.join(directory, run_name)
    make_directories_if_needed(directory_name)

    # Saving parameters as json
    with open(os.path.join(directory_name, file_name), 'w') as meta_file:
        json.dump(meta, meta_file)
    pass


def save_state(run_name, network_vars, directory=RESULTS_DIR, file_name='network.h5'):
    directory_name = os.path.join(directory, run_name)
    make_directories_if_needed(directory_name)

    # Saving network state in hdf5 file
    with h5py.File(os.path.join(directory_name, file_name), 'w') as network_file:
        for name, n_var in network_vars.items():
            if isinstance(n_var, float):
                network_file.create_dataset(name, (), data=n_var) 
            else:
                network_file.create_dataset(name, n_var.shape, data=n_var) 
    pass


def save_network_all(run_name, params, network_vars, meta, directory=RESULTS_DIR):
    save_params(run_name, params, directory=directory)
    save_state(run_name, network_vars, directory=directory)
    save_meta(run_name, meta, directory=directory)
    pass


def save_outputs(run_name, output_vars, directory=RESULTS_DIR, file_name='outputs.csv'):
    directory_name = os.path.join(directory, run_name)
    make_directories_if_needed(directory_name)

    # Saving outputs as pandas DataFrame
    output_df = pd.DataFrame({name: o_var for name, o_var in output_vars.items()})
    output_df.to_csv(os.path.join(directory_name, file_name), index_label='trial')
    pass


def save_activity(run_name, activity, directory=RESULTS_DIR, file_name='activity.h5'):
    directory_name = os.path.join(directory, run_name)
    make_directories_if_needed(directory_name)

    # Saving activity as hdf5 file
    with h5py.File(os.path.join(directory_name, file_name), 'w') as network_file:
        network_file.create_dataset("Activity", activity.shape, data=activity) 
    pass
    

def load_network(run_name, directory=SAVED_DIR, file_name='network.h5'):
    check_if_file_exists(file_name, run_name, directory)
    with h5py.File(os.path.join(directory, run_name, file_name), 'r') as network_file:
        net_vars = {n_var: copy(network_file[n_var]) for n_var in network_file.keys()}
    return net_vars


def load_params(run_name, directory=SAVED_DIR, file_name='params.json' ):
    check_if_file_exists(file_name, run_name, directory)
    with open(os.path.join(directory, run_name, file_name), 'r') as params_file:
        params = json.load(params_file)
    return params


def load_meta(run_name, directory=SAVED_DIR, file_name='metadata.json'):
    check_if_file_exists(file_name, run_name, directory)
    with open(os.path.join(directory, run_name, file_name), 'r') as meta_file:
        meta = json.load(meta_file)
    return meta

def load_experiment_meta(exp_name, directory=".", file_name='exp_metadata.json'):
    check_if_file_exists(file_name, exp_name, directory)
    with open(os.path.join(directory, exp_name, file_name), 'r') as meta_file:
        meta = json.load(meta_file)
    return meta


def load_experiment_results(exp_name, results_name, directory=""):
    check_if_file_exists(results_name, exp_name, directory)
    with open(os.path.join(directory, exp_name, results_name), 'r') as exp_file:
        results_df = pd.read_csv(exp_file, index_col=0)
    return results_df


def load_network_all(run_name, directory=SAVED_DIR):
    '''Creates an index of runs and parameters.

    Arguments
    ---------
    1. run_name (string): name of run to load
    * directory (string): name of directory (default SAVED_DIR)

    Results
    -------
    1. metadata (dictionary)
    2. parameters (dictionary)
    3. network_variables (dictionary)
    '''
    params = load_params(run_name, directory=directory)
    meta = load_meta(run_name, directory=directory)
    network = load_network(run_name, directory=directory)
    return meta, params, network


def load_outputs(run_name, directory=SAVED_DIR, file_name='inputs_outputs.csv'):
    check_if_file_exists(file_name, run_name, directory)
    return(pd.read_csv(os.path.join(directory, run_name, file_name)))


def load_activity(run_name, directory=SAVED_DIR, file_name='activity.h5'): 
    check_if_file_exists(file_name, run_name, directory)
    with h5py.File(os.path.join(directory, run_name, file_name), 'r') as network_file:
        return(network_file["Activity"][:])

def load_responsiveness(run_name, directory=SAVED_DIR, file_name='analysis_sets.json'):
    check_if_file_exists(file_name, run_name, directory)
    with open(os.path.join(directory, run_name, file_name), 'r') as sets_file:
        sets = json.load(sets_file)
    return sets

def load_rates(run_name, directory=SAVED_DIR, file_name='population rates.csv'):
    check_if_file_exists(file_name, run_name, directory)
    return(pd.read_csv(os.path.join(directory, run_name, file_name)))

def load_weights(run_name, directory=SAVED_DIR, file_name='weights.csv'):
    check_if_file_exists(file_name, run_name, directory)
    return(pd.read_csv(os.path.join(directory, run_name, file_name)))

def load_current(run_name, directory=SAVED_DIR, file_name='current.csv'):
    check_if_file_exists(file_name, run_name, directory)
    return(pd.read_csv(os.path.join(directory, run_name, file_name)))

def load_convergence(run_name, directory=SAVED_DIR, file_name='convergence.json' ):
    check_if_file_exists(file_name, run_name, directory)
    with open(os.path.join(directory, run_name, file_name), 'r') as convergence_file:
        convergence = json.load(convergence_file)
    return convergence

def load_sources(run_name, directory=SAVED_DIR, file_name='analysis_sources.json' ):
    check_if_file_exists(file_name, run_name, directory)
    with open(os.path.join(directory, run_name, file_name), 'r') as sources_file:
        sources = json.load(sources_file)
    return sources
