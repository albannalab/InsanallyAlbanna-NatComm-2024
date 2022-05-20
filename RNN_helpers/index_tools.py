import os
import json
import pandas as pd
from IPython.display import display, HTML
from .io import RESULTS_DIR, load_params, load_meta

def make_index(directory=RESULTS_DIR):
    '''Creates an index of runs and parameters.

    Arguments
    ---------
    * directory (string): directory to look for runs in `directory` (default = RESULTS_DIR)

    Results
    -------
    1. index_df (pandas.DataFrame): index of runs and parameters
    '''
    data = []
    for r_dir in os.listdir(directory):
        if r_dir[0:3] == 'run':
            try:
                params = load_params(r_dir, directory=directory)
                exp_params = load_params(r_dir, directory=directory, file_name='exp_params.json')
                meta = load_meta(r_dir, directory=directory)
                params.update(meta)
                params.update(exp_params)
                data.append(params)
            except:
                continue
    index_df = pd.DataFrame(data)
    index_df = index_df.sort_values(by='run').reset_index(drop=True)
    index_df.to_csv(os.path.join('.', directory, 'index.df'))
    return(index_df)


def show_df(df):
    '''Prints dataframe in HTML format for jupyter notebooks

    Arguments
    ---------
    1. (pandas dataframe) 

    Results
    -------
    None
    '''
    with pd.option_context('display.max_rows', 1000, 'display.max_colwidth', 100):
        display(HTML(df.to_html()))
    pass