import numpy as np
import os
import json
import math
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu, shapiro, linregress
from ..io import SAVED_DIR, RESULTS_DIR, load_network, load_params, load_responsiveness
from ..analysis import find_response_range, parse_activity, find_stimulus_range, find_response_time, binarize_inputs, binarize_outputs
from .statistics import paired_counts, perm_test_means
from tqdm import tqdm

def calculate_sources(run_name, directory=SAVED_DIR, version="v2", verbose=False, index_convention="julia", auto_save=True):
    print("Calculate source")

    #Load Weights
    network_weights = load_network(run_name,directory=directory)
    w = network_weights["W"]
    #Load params
    p = load_params(run_name,directory=directory)
    #Load sets
    sets = load_responsiveness(run_name,directory=directory)

    sources = {}
    
    #Analyze sources
    for i in tqdm(range(p["N"]), desc="Analyzing"):
        if index_convention == 'python':
            key = i
        elif index_convention == 'julia':
            key = i + 1
        if key not in sources:
            sources[key] = {}

        #Excitatory input
        sources[key]["exc"] = Float64(np.mean(w[i,1:p["NE"]]))
        #Inhibitory input
        sources[key]["inh"] = Float64(np.mean(w[i,p["NE"]:p["N"]]))
        #Input layer input
        sources[key]["input"] = Float64(np.mean(w[i,1:p["N_in"]]))
        #Output layer input
        sources[key]["output"] = Float64(np.mean(w[i,p["N_in"]:p["NE"]]))
        #Number of inputs
        sources[key]["numinputs"] = len(findnz(w[i,:]))

        NNR = 0
        NNRnum = 0
        R = 0
        Rnum = 0
        NNRinput = 0
        NNRinputnum = 0
        Rinput = 0
        Rinputnum = 0
        NNRinhib = 0
        NNRinhibnum = 0
        Rinhib = 0
        Rinhibnum = 0
        NNRoutput = 0
        NNRoutputnum = 0
        Routput = 0
        Routputnum = 0
        for j in range(p["N"]):
            if j in sets["non"]:
                NNR += w[i,j]
                NNRnum += 1
                if j <= p["N_in"]:
                    NNRinput += w[i,j]
                    NNRinputnum += 1
                elif j > p["NE"]:
                    NNRinhib += w[i,j]
                    NNRinhibnum += 1
                else:
                    NNRoutput += w[i,j]
                    NNRoutputnum += 1
            else:
                R += w[i,j]
                Rnum += 1
                if j <= p["N_in"]:
                    Rinput += w[i,j]
                    Rinputnum += 1
                elif j > p["NE"]:
                    Rinhib += w[i,j]
                    Rinhibnum += 1
                else:
                    Routput += w[i,j]
                    Routputnum += 1
        sources[key]["NNR"] = (NNR/NNRnum)
        sources[key]["R"] = (R/Rnum)
        sources[key]["NNRinput"] = (NNRinput/NNRinputnum)
        sources[key]["Rinput"] = (Rinput/Rinputnum)
        sources[key]["NNRoutput"] = (NNRoutput/NNRoutputnum)
        sources[key]["Routput"] = (Routput/Routputnum)
        sources[key]["NNRinhib"] = (NNRinhib/NNRinhibnum)
        sources[key]["Rinhib"] = (Rinhib/Rinhibnum)

    metadata = {'index_convention': index_convention}
    sources.update(metadata)

    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_sources.json'), 'w') as f:
            json.dump(sources, f)
    return (sources)
