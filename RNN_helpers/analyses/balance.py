import numpy as np
import os
import json
import math
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu, shapiro, linregress
from ..io import SAVED_DIR, RESULTS_DIR, load_network, load_params, load_responsiveness, load_activity
from ..analysis import find_response_range, parse_activity, find_stimulus_range, find_response_time, binarize_inputs, binarize_outputs
from .statistics import paired_counts, perm_test_means
from tqdm import tqdm

def calculate_balance(run_name, directory=SAVED_DIR, version="v2", verbose=False, index_convention="julia", auto_save=True):
    print("Calculate Input Characterization")

    #Load Weights
    network_weights = load_network(run_name,directory=directory)
    w = network_weights["W"]
    #Load params
    p = load_params(run_name,directory=directory)

    #Load Activity
    activity = load_activity(run_name,directory=directory)
    unit = activity[1,:]
    time = activity[2,:]
    #Activity [0] = ???
    #Activity [1] = Unit
    #Activity [2] = Time

    #Baseline,stim,resp]
    Net_current = np.zeros([p["N"],3])
    Exc_current = np.zeros([p["N"],3])
    Inh_current = np.zeros([p["N"],3])
    Current_by_unit = np.zeros([p["N"],p["N"],3])

    #times = []*p["N"]
    for i in tqdm(range(len(unit)), desc="Counting"):
        unitID = int(unit[i])
        for j in range(p["N"]):
            if (unitID > p["NE"]):
                #Inhibitory
                baseline = True
                if time[i] > p["stim_offset"] and time[i] <= p["stim_offset"]+p["stim_dur"]:
                    Net_current[j,1] -= w[j,unitID-1]
                    Inh_current[j,1] -= w[j,unitID-1]
                    Current_by_unit[j,unitID-1,1] -= w[j,unitID-1]
                    baseline = False
                if time[i] > p["resp_offset"] and time[i] <= p["resp_offset"]+p["resp_dur"]:
                    Net_current[j,2] -= w[j,unitID-1]
                    Inh_current[j,2] -= w[j,unitID-1]
                    Current_by_unit[j,unitID-1,2] -= w[j,unitID-1]
                    baseline = False
                if baseline:
                    Net_current[j,0] -= w[j,unitID-1]
                    Inh_current[j,0] -= w[j,unitID-1]
                    Current_by_unit[j,unitID-1,0] -= w[j,unitID-1]
            else:
                baseline = True
                if time[i] > p["stim_offset"] and time[i] <= p["stim_offset"]+p["stim_dur"]:
                    Net_current[j,1] += w[j,unitID-1]
                    Exc_current[j,1] += w[j,unitID-1]
                    Current_by_unit[j,unitID-1,1] += w[j,unitID-1]
                    baseline = False
                if time[i] > p["resp_offset"] and time[i] <= p["resp_offset"]+p["resp_dur"]:
                    Net_current[j,2] += w[j,unitID-1]
                    Exc_current[j,2] += w[j,unitID-1]
                    Current_by_unit[j,unitID-1,2] += w[j,unitID-1]
                    baseline = False
                if baseline:
                    Net_current[j,0] += w[j,unitID-1]
                    Exc_current[j,0] += w[j,unitID-1]
                    Current_by_unit[j,unitID-1,0] += w[j,unitID-1]
            #if w[j,unitID-1] > 0:
                #times[j].append(times[i])
    #for i in tqdm(range(p["N"]), desc="Analyzing"):
    #    times[i] = np.diff(times[i])
    #    times[i] = np.std(times[i])


    balance = {}
    balance["I_Net"] = Net_current
    balance["I_Exc"] = Exc_current
    balance["I_Inh"] = Inh_current
    balance["I_unit"] = Current_by_unit
    #balance["ISIstd"] = times

    metadata = {'index_convention': index_convention}
    balance.update(metadata)

    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_balance.json'), 'w') as f:
            json.dump(balance, f)
    return (balance)
