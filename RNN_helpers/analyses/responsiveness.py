import numpy as np
import os
import json
import math
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu, shapiro, linregress
from ..io import SAVED_DIR, RESULTS_DIR
from ..analysis import find_response_range, parse_activity, find_stimulus_range, find_response_time, binarize_inputs, binarize_outputs
from .statistics import paired_counts, perm_test_means
from tqdm import tqdm

#This returns a variety of points of the distrubution of means. Send the points you want in with sig
def perm_test_dist(xs, sig, sub_sample_frac=1.0, replacement=True, repetitions=20000, **kwargs):
    n = len(xs)
    sub_n = int(n*sub_sample_frac)
    k = np.empty(repetitions)
    for i in range(repetitions):
        k[i] = np.mean(np.random.choice(xs,sub_n,replace=replacement),dtype=np.float32)
    k = np.sort(k,kind='heapsort')
    sig = np.rint(sig * repetitions).astype(int)
    res = (k[sig].flatten())
    res = res.tolist()
    return res

def calculate_responsiveness(run_name, dist_points = [0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975], repetitions = 0, bin_size=100, FRthresh=3.33, directory=SAVED_DIR, version="v2", verbose=False, index_convention="julia", auto_save=True):
    print("Continuous Responsiveness")
    sig = np.array(dist_points)
    #s0 is stimulus input. T for stim, F for no stim
    s0 = binarize_inputs(run_name, directory=directory)
    #a0 is behavioural response. NP for Nose Poke, W for Withhold
    try:
        a0 = binarize_outputs(run_name, directory=directory)
    except:
        a0 = None
    #np1 seems to be how long each trial is
    np1 = find_response_time(run_name, directory=directory)
    #r0 seems to be the spike times for each trial for each unit (indexed by unit then trial)
    r0 = parse_activity(run_name, directory=directory, version=version, stim_offset=False, units='ms')
    #r_start and r_end define the response window (the green section of the graph)
    r_start, r_finish = find_response_range(run_name, directory=directory)
    #s_start and s_end define the stimulus window (the gray section of the graph)
    s_start, s_finish = find_stimulus_range(run_name, directory=directory)
    num_trials = len(s0)
    num_cells = len(r0)

    #cells will store the individual responsiveness of each cell in the network for stim and response
    #s_ stimulus, T = true, F = false
    #c_ choice, NP = nose poke, W = withhold
    cells = {}
    #{'s_mean': [],
    #'sT_mean': [],
    #    'sF_mean': [],
    #        'c_mean': [],
    #        'cNP_mean': [],
    #        'cW_mean': [],}

    #for i in range(num_cells):
    for i in tqdm(range(num_cells), desc="Analyzing"):
        if index_convention == 'python':
            key = i
        elif index_convention == 'julia':
            key = i + 1

        if key not in cells:
            cells[key] = {}

        if verbose: print("checking non-evoked with threshold = {0}...".format(rate_thresh))
        #All trials for this unit
        all_r = r0[i]
        #all nontarget trials for this unit
        nontarget_r = r0[i][s0 =='F']
        #all target trials for this unit
        target_r = r0[i][s0 == 'T']

        for name, group in [('s_', all_r)]:

            #overall firing rate
            counts = np.array([paired_counts(response, time_intervals=[(0, r_start + bin_size), (0, r_start + bin_size)]) for response in group])
            firing_rates = counts[:, 0] / float(3*bin_size) * 1000.0
            cells[key]['fr_mean'] = float(np.mean(firing_rates,dtype=np.float32))

            #Baseline Firing rate
            counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (s_start, s_start + bin_size)]) for response in group])
            firing_rates = counts[:, 0] / float(bin_size) * 1000.0
            cells[key]['pre_fr_mean'] = float(np.mean(firing_rates,dtype=np.float32))
            if repetitions>0:
                cells[key]['pre_fr_dist'] = perm_test_dist(firing_rates, sig, repetitions=repetitions)
            cells[key]['pre_fr_std'] = float(np.std(firing_rates,dtype=np.float32, ddof=1))

            #Calculate overall stimulus-induced change here
            new_counts = np.hstack((counts[:,1] - counts[:,0])) / float(bin_size) * 1000.0
            mean_change = float(np.mean(new_counts,dtype=np.float32))
            std_change = float(np.std(new_counts,dtype=np.float32, ddof=1))
            if verbose: print("   " + name + " mean: {0}, std:{1}".format(mean_change,std_change))
            cells[key][name + 'mean'] = mean_change
            if repetitions>0:
                cells[key][name + 'dist'] = perm_test_dist(new_counts, sig, repetitions=repetitions)
            cells[key][name + 'std'] = std_change

        try:
            for name, group in [('sT_', target_r), ('sF_', nontarget_r)]:
                #counts stores the number of spikes in the baseline and stimulus periods
                counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (s_start, s_start + bin_size)]) for response in group])
                #new_counts stores the difference between stimulus period and baseline spike counts
                new_counts = np.hstack((counts[:,1] - counts[:,0])) / float(bin_size) * 1000.0

                mean_change = float(np.mean(new_counts,dtype=np.float32))
                std_change = float(np.std(new_counts,dtype=np.float32, ddof=1))
                if verbose: print("   " + name + " mean: {0}, std:{1}".format(mean_change,std_change))
                cells[key][name + 'mean'] = mean_change
                if repetitions>0:
                    cells[key][name + 'dist'] = perm_test_dist(new_counts, sig, repetitions=repetitions)
                cells[key][name + 'std'] = std_change
            for name, group in [('cT_', target_r), ('cF_', nontarget_r)]:
                #counts stores the number of spikes in the baseline and stimulus periods
                counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (r_start, r_start + bin_size)]) for response in group])
                #new_counts stores the difference between stimulus period and baseline spike counts
                new_counts = np.hstack((counts[:,1] - counts[:,0])) / float(bin_size) * 1000.0

                mean_change = float(np.mean(new_counts,dtype=np.float32))
                std_change = float(np.std(new_counts,dtype=np.float32, ddof=1))
                if verbose: print("   " + name + " mean: {0}, std:{1}".format(mean_change,std_change))
                cells[key][name + 'mean'] = mean_change
                if repetitions>0:
                    cells[key][name + 'dist'] = perm_test_dist(new_counts, sig, repetitions=repetitions)
                cells[key][name + 'std'] = std_change
        except:
            pass
            #print("Error: skipping T/F")

        #This section is the same as above, but for response rather than stim
        try:
            np_r = r0[i][a0 =='NP']
            w_r =  r0[i][a0 == 'W']
            for name, group in [('c_', all_r), ('cNP_', np_r), ('cW_', w_r)]:
                counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (r_start, r_start+bin_size)]) for response in group])
                new_counts = np.hstack((counts[:,1] - counts[:,0])) / float(bin_size) * 1000.0

                mean_change = float(np.mean(new_counts,dtype=np.float32))
                std_change = float(np.std(new_counts,dtype=np.float32, ddof=1))
                if verbose: print("   " + name + " mean: {0}, std:{1}".format(mean_change,std_change))
                cells[key][name + 'mean'] = mean_change
                if repetitions>0:
                    cells[key][name + 'dist'] = perm_test_dist(new_counts, sig, repetitions=repetitions)
                cells[key][name + 'std'] = std_change
            for name, group in [('sNP_', np_r), ('sW_', w_r)]:
                counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (s_start, s_start+bin_size)]) for response in group])
                new_counts = np.hstack((counts[:,1] - counts[:,0])) / float(bin_size) * 1000.0

                mean_change = float(np.mean(new_counts,dtype=np.float32))
                std_change = float(np.std(new_counts,dtype=np.float32, ddof=1))
                if verbose: print("   " + name + " mean: {0}, std:{1}".format(mean_change,std_change))
                cells[key][name + 'mean'] = mean_change
                if repetitions>0:
                    cells[key][name + 'dist'] = perm_test_dist(new_counts, sig, repetitions=repetitions)
                cells[key][name + 'std'] = std_change
        except:
            print("Error: skipping NP/W")

        #Calculate responsiveness
        resp = math.sqrt(cells[key]['s_mean']**2+cells[key]['c_mean']**2)
        cells[key]['responsiveness'] = resp

    active_dict = {}
    for k, v in cells.items():
        if v['fr_mean'] > FRthresh:
            active_dict[k]=v['responsiveness']
    active_list = list(active_dict.items())
    NNR_list = sorted(active_list,key=lambda x:abs(x[1]),reverse=False)
    R_list = sorted(active_list,key=lambda x:abs(x[1]),reverse=True)
    Alt_list = []
    for i in range(int(num_cells/2)):
        Alt_list.append(NNR_list[i])
        Alt_list.append(R_list[i])
    cells.update({'NNR_sorted':NNR_list})
    cells.update({'R_sorted':R_list})
    cells.update({'Alt_sorted':Alt_list})

    metadata = {'bin_size':bin_size, 'index_convention': index_convention, 'distrubution_points': dist_points}
    cells.update(metadata)

    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_cells.json'), 'w') as f:
            json.dump(cells, f)
    return(cells)
