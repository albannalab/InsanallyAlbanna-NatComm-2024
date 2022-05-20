import numpy as np
import os
from .loadNetworkForAnalysis import loadNetwork
from ..analysis import parse_activity
from ..io import SAVED_DIR


def spike_rates(activity, window_size=20, start_time=0, end_time=300):
    n_units, n_trials = activity.shape
    n_bins = int((end_time - start_time) / window_size)
    firing_rates = np.zeros((n_units, n_trials, n_bins))
    start_times = np.arange(start_time, end_time, window_size)
    for i in range(n_units):
        for j in range(n_trials):
            firing_rates[i, j] = [np.sum((activity[i, j]>=st) * (activity[i, j]<st+window_size)) for st in start_times]
    return(firing_rates, start_times+window_size)

def flatten_rates(firing_rates, trial_labels=None):
    n_units, n_trials, n_bins = firing_rates.shape
    flat_firing_rates = np.zeros((n_units, n_bins * n_trials))
    for i in range(n_units):
        flat_firing_rates[i] = [fr for j in range(n_trials) for fr in firing_rates[i, j]]
    if trial_labels is None:
        return(flat_firing_rates)
    else:
        new_labels = []
        for labels in trial_labels:
            new_labels.append([lab for lab in labels for i in range(n_bins)])
        return(flat_firing_rates, new_labels)

def angle(v_1, v_2, include_sign=False, units='degree'):
    angle = np.arccos(np.dot(v_1, v_2) / (np.sqrt(np.dot(v_1, v_1) * np.dot(v_2, v_2))))
    if not include_sign:
        angle = np.min([angle, np.abs(np.pi - angle)])
    if units == 'radians':
        return angle 
    else:
        return angle * 180 / np.pi 

def find_inactive(C, thresh=.1):
    max_val = np.max(np.abs(C))
    inactive = [i for i, row in enumerate(C) if np.all(np.abs(row) <= thresh*max_val)]
    return inactive

def drop_inactive(C):
    idx = list(set(range(C.shape[0])) - set(find_inactive(C)))
    C_mod = C[idx][:, idx]
    return C_mod

def calc_var_covar(C):
    var = np.mean(np.diag(C))
    cross = np.mean(C[~np.eye(*C.shape, dtype=bool)])
    return var, cross

def calc_dim_var_measures(C):
    assert C.shape[0] == C.shape[1]
    N = C.shape[0]

    var = C[np.eye(N,N, dtype=bool)]
    var4 = np.mean(var**2)

    covar = C[~np.eye(N,N, dtype=bool)]
    covar2 = np.mean(covar**2)

    C_mod_1 = np.outer(var, var)
    var2 = np.mean(C_mod_1[~np.eye(N,N, dtype=bool)])

    return(var4, var2, covar2)

def calc_dim_GT(C):
    return(np.trace(C)**2 / np.trace(C@C))

def calc_dim(N, var2, cv2):
    return((1 + (N-1)*var2)/(1 + (N-1)*cv2))

def calc_dims(N, N_inac, var2, cv2):
    return(calc_dim(N, var2, 0), calc_dim(N, var2, cv2), calc_dim(N-N_inac, var2, cv2))

def calculate_dimensionality_stats(run_name, directory=SAVED_DIR, auto_save=False):
    activity = parse_activity(run_name, directory=directory)
    net = loadNetwork(directory, run_name)

    # all_activity = np.vstack([activity[0:net.p.N_in], activity[-net.p.N_out-net.p.NI:, :]])
    out_activity = activity[-net.p.NI-net.p.N_out:-net.p.NI, :]

    out_FR_comp, start_times = spike_rates(out_activity)
    out_FR, (inputs_all, stimulus_all, outputs_all) = flatten_rates(out_FR_comp, trial_labels=(net.task.inputs, net.task.all["inputs"], net.task.outputs))     

    try:
        resp = np.array(net.responsiveness)
        ind = np.argsort(resp)
        FR_sort = out_FR[ind, :]
        C = np.cov(FR_sort)
    except:
        C = np.cov(out_FR)
        
    N = C.shape[0]
    inactive = find_inactive(C)
    N_inac = len(inactive)

    C = drop_inactive(C)
    v4, v2, cv2 = calc_dim_var_measures(C)

    d1, d2, d3 = calc_dims(N, N_inac, v2/v4, cv2/v4)
    d_gt = calc_dim_GT(C)

    results = {"N":N, 
                "N_inac": N_inac,
                "var2": v2,
                "covar2": cv2,
                "var4": v4,
                "var_ratio": v2/v4,
                "covar_ratio": cv2/v4,
                "D_var": d1, 
                "D_covar": d2, 
                "D_inac": d3,
                "D_gt": d_gt
                }
    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_dimensionality_stats.json'), 'w') as f:
            json.dump(results, f)
    return(results)