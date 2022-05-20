import numpy as np
import os
import json
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu, shapiro, linregress
from ..io import SAVED_DIR, RESULTS_DIR
from ..analysis import find_response_range, parse_activity, find_stimulus_range, find_response_time, binarize_inputs, binarize_outputs 
from tqdm import tqdm

def flatten(list_of_lists):
    flat_list = np.array([item for items in list_of_lists for item in items])
    return(flat_list)

def spike_counts(spike_times, time, bin_size=.050):
    time_counts = np.array([sum((spike_times >= start)*(spike_times < (start + bin_size))) 
                   for start in np.arange(0, time, bin_size)])
    return(Counter(time_counts))

def paired_counts(response, time_intervals=[(-100, 0), (0, 100)]):
    spike_counts = []
    for time_interval in time_intervals:
        spike_counts.append(sum((response>=time_interval[0])*(response<time_interval[1])))
    return(tuple(spike_counts))

def counters_to_contingency(list_of_counters):
    max_spikes = max([max(counter.keys()) for counter in list_of_counters])
    num_groups = len(list_of_counters)
    contingency = np.zeros((max_spikes + 1, num_groups))
    for i in range(max_spikes + 1):
        for j in range(num_groups):
            contingency[i, j] = list_of_counters[j][i]
    return(contingency)

def counter_to_list(counter):
    counter_list = []
    for value, count in counter.items():
        counter_list.extend(count*[value])
    return(counter_list)

def my_shuffle(an_array):
    np.random.shuffle(an_array)
    return(an_array)

def sample_means(samples, sub_sample_size=200, repetitions=5000):
    list_of_means = [np.mean(my_shuffle(samples)[:sub_sample_size]) for rep in range(repetitions)]
    return(list_of_means)

def chi2_loop(table, limit=5, **kwargs):
    table_copy = table.copy()
    while True:
        try:
            chi2, p, dof, expected = chi2_contingency(table_copy, **kwargs)
        except:
            table_copy[-2,:] = table_copy[-2,:] + table_copy[-1,:]
            table_copy = table_copy[0:-1,:]
            continue
        if np.all(expected > limit):
            return(chi2, p, dof, expected)
        else:
            table_copy[-2,:] = table_copy[-2,:] + table_copy[-1,:]
            table_copy = table_copy[0:-1,:]

def std_diff(xs, ys, shift=0):
    return(np.std(xs, ddof=1) - np.std(ys, ddof=1) + shift)

def mean_diff(xs, ys, shift=0):
    return(np.mean(xs) - np.mean(ys) + shift)

def perm_test(xs, ys, test_stat=mean_diff, repetitions=200, shift=0, **kwargs):
    n = len(xs)
    k = 0
    diff = test_stat(xs, ys, shift=shift, **kwargs)
    zs = np.concatenate([xs, np.array(ys)-shift])
    for j in range(repetitions):
        np.random.shuffle(zs)
        k += np.abs(diff) < np.abs(test_stat(zs[:n], zs[n:], shift=0, **kwargs))
    return k / float(repetitions)

#This version of perm_test_means is the old Jackknifing version. It is
#slightly faster but is limited to jackknifing resampling
def perm_test_means_Fast(xs, threshold, sub_sample_frac=.9, repetitions=20000, **kwargs):
    n = len(xs)
    sub_n = int(n*sub_sample_frac)
    k = 0
    for i in range(repetitions):
        np.random.shuffle(xs)
        if np.mean(xs[:sub_n]) > threshold:
            k += 1
    return(k / float(repetitions))

#This version of perm_test_means is the new more general version. It can perform
#either jackknifing or bootstrapping based on the sub_sample_frac and replacement
#variables. By default it performs bootstrapping
def perm_test_means(xs, threshold, sub_sample_frac=1.0, replacement=True, repetitions=20000, **kwargs):
    n = len(xs)
    sub_n = int(n*sub_sample_frac)
    k = 0
    for i in range(repetitions):
        if np.mean(np.random.choice(xs,sub_n,replace=replacement)) > threshold:
            k += 1
    return(k / float(repetitions))

def test_ramp(r_np, thresh_fac=.5, sub_sample_frac=.9, repetitions=1000, window=1, bin_size=100, **kwargs):
    n = len(r_np)
    sub_n = int(n*sub_sample_frac)
    k = 0

    r_np_collapse = np.sort(np.array([ spike_time for resp in r_np for spike_time in resp ]))
    times = np.arange(0, window, bin_size)
    firing_rates = [ np.sum((r_np_collapse >= start_time)*(r_np_collapse < (start_time + bin_size))) / (bin_size*num_trials) for start_time in times ]
    avg_firing_rate = numpy.mean(firing_rates)
    avg_firing_rates.append(avg_firing_rate)
    # firing_rates = np.array(firing_rates) / avg_firing_rate
    slope, intercept, r_value, p_value, std_err = linregress(times, firing_rates)
    
    avg_firing_rate = numpy.mean(avg_firing_rates)
    # ramp_threshold = 1. / avg_np
    # ramp_threshold = .75 / WINDOW
    ramp_threshold = np.max((thresh_fac * avg_firing_rate / window, 1. / window))
    
    for i in range(repetitions):
        np.random.shuffle(r_np)
        r_np_sub = r_np[0:sub_n]  
        r_np_collapse = np.sort(np.array([ spike_time for resp in r_np_sub for spike_time in resp ]))
        times = np.arange(0, window, bin_size)
        firing_rates = [ np.sum((r_np_collapse >= start_time)*(r_np_collapse < (start_time + bin_size))) / (bin_size*num_trials) for start_time in times ]
        avg_firing_rate = numpy.mean(firing_rates)
        # avg_firing_rates.append(avg_firing_rate)
        # firing_rates = np.array(firing_rates) / avg_firing_rate
        slope_temp, _, _, _, _ = linregress(times, firing_rates)
        if np.abs(slope_temp) >= np.abs(ramp_threshold):
            k += 1
    p_boot = float(k) / repetitions
    return p_boot, ramp_threshold, slope, r_value, p_value

def my_wilcoxon(xs, ys, shift=0, **kwargs):
    return(wilcoxon(xs - ys, shift, **kwargs)[1])

def my_mannwhitneyu(xs, ys, shift=0, **kwargs):
    return(mannwhitneyu(xs - ys, shift, **kwargs)[1])

def test_responsiveness(run_name, rate_thresh=0.2, rate_alpha=0.05, bin_size=100, repetitions=1000, directory=SAVED_DIR, version="v2", verbose=False, index_convention="julia", auto_save=True):
    print("Discrete Responsiveness")
    #s0 is stimulus input. T for stim, F for no stim
    s0 = binarize_inputs(run_name, directory=directory)
    #a0 is behavioural response. NP for Nose Poke, W for Withhold
    a0 = binarize_outputs(run_name, directory=directory)
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

    #stats is intended to be a count
    stats = {'non':0, 'non_ramp':0, 'non_resp': 0, 'total': 0}
    statistics = {}
    #sets is intended to be a list of the units in each category
    #s_ = all stimulus
    #sT_ = true stimulus
    #sF_ = false stimulus
    sets = {'non': [],
            'non_resp': [],
            'non_ramp':[],#NOTE: ramping cells defined here as difference between baseline and response period
            's_evoked': [],
            's_suppressed': [],
            'sT_evoked': [],
            'sT_suppressed': [],
            'sF_evoked': [],
            'sF_suppressed': [],
            'c_evoked': [],
            'c_suppressed': [],
            'cNP_evoked': [],
            'cNP_suppressed': [],
            'cW_evoked': [],
            'cW_suppressed': [],
            'all': []}

    #for i in range(num_cells):
    for i in tqdm(range(num_cells), desc="Analyzing"):
        if index_convention == 'python':
            key = i
        elif index_convention == 'julia':
            key = i + 1
        
        if key not in statistics:
            statistics[key] = {}
            sets['all'].append(key)
        if verbose: print(" - unit {0}".format(i))

        non_ramp = False
        non_resp = False
        stats['total'] += 1

        if verbose: print("   checking non-evoked with threshold = {0}...".format(rate_thresh))
        #All trials for this unit
        all_r = r0[i]
        #all nontarget trials for this unit
        nontarget_r = r0[i][s0 =='F']
        #all target trials for this unit
        target_r = r0[i][s0 == 'T']

        for name, group in [('s_', all_r), ('sT_', target_r), ('sF_', nontarget_r)]:
            #counts stores the number of spikes in the baseline and stimulus periods
            counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (s_start, s_start + bin_size)]) for response in group])
            #new_counts stores the difference between stimulus period and baseline spike counts
            new_counts = np.hstack((counts[:,1] - counts[:,0]))
            #The p value associated with a random mean change in spike count being > rate_thresh
            p_value_evoked = perm_test_means(new_counts, rate_thresh, repetitions=repetitions)
            #The p value associated with a random mean change in spike count being < -rate_thresh
            p_value_suppressed = perm_test_means(-new_counts, rate_thresh, repetitions=repetitions)
            statistics[key][name + 'evoked_p'] = p_value_evoked
            statistics[key][name + 'suppressed_p'] = p_value_suppressed
            if verbose: print("   " + name + ":{0}, {1}".format(p_value_evoked, p_value_suppressed))

            #The test for non-responsiveness is as follows:
            #If the unit is significantly unlikely to have a random mean firing rate be rate_thresh greater
            #for stimulus than baseline and is also significantly unlikely to have a random mean firing rate be
            #rate_thresh less for stimulus than baseline, then it is called non-responsive.

            if (p_value_evoked < rate_alpha) and (p_value_suppressed < rate_alpha): # and (p_value_evoked_2 < rate_alpha) and (p_value_suppressed_2 < rate_alpha):
                if name == 's_':
                    if verbose: print("   " + name + "non-resp!")
                    non_resp = True
                    stats['non_resp'] += 1
                    sets['non_resp'].append(key)

            if not non_resp and (p_value_evoked >= (1. - rate_alpha)): # or (p_value_evoked_2 >= (1. - rate_alpha)))
                sets[name + 'evoked'].append(key)

            if not non_resp and (p_value_suppressed >= (1. - rate_alpha)): # or (p_value_suppressed_2 >= (1. - rate_alpha)))
                sets[name + 'suppressed'].append(key)

        #This section is the same as above, but for response rather than stim
        np_r = r0[i][a0 =='NP']
        w_r =  r0[i][a0 == 'W']
        for name, group in [('c_', all_r), ('cNP_', np_r), ('cW_', w_r)]:
            counts = np.array([paired_counts(response, time_intervals=[(0, bin_size), (r_start, r_start+bin_size)]) for response in group])
            
            # Saving firing rates
            firing_rates = counts[:, 1] / bin_size * 1000 
            statistics[key][name + 'mean_fr'] = np.mean(firing_rates)
            statistics[key][name + 'std_fr'] = np.std(firing_rates, ddof=1)
            
            new_counts = np.hstack((counts[:,1] - counts[:,0]))
            p_value_evoked = perm_test_means(new_counts, rate_thresh, repetitions=2000)
            p_value_suppressed = perm_test_means(-new_counts, rate_thresh, repetitions=2000)
            statistics[key][name + 'evoked_p'] = p_value_evoked
            statistics[key][name + 'suppressed_p'] = p_value_suppressed
            if verbose: print("   " + name + ":{0}, {1}".format(p_value_evoked, p_value_suppressed))

            if (p_value_evoked < rate_alpha) and (p_value_suppressed < rate_alpha):
                if name == 'c_':
                    if verbose: print("   " + name + "non-ramp!")
                    non_ramp = True
                    stats['non_ramp'] += 1
                    sets['non_ramp'].append(key)

            if not non_resp and ((p_value_evoked >= (1. - rate_alpha))):
                sets[name + 'evoked'].append(key)

            if not non_resp and ((p_value_suppressed >= (1. - rate_alpha))):
                sets[name + 'suppressed'].append(key)

        #Any units that were responsive to stim and response are called "non"
        if non_ramp and non_resp:
            sets['non'].append(key)
            stats['non'] += 1
            if verbose: print("   NON!")

    metadata = {'rate_thresh':rate_thresh, 'rate_alpha':rate_alpha, 'bin_size':bin_size, 'repetitions':repetitions, 'index_convention': index_convention}
    statistics.update(metadata)
    sets.update(metadata)
    stats.update(metadata)
    
    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_evoked_statistics.json'), 'w') as f:
            json.dump(statistics, f)

        with open(os.path.join('.', directory, run_name, 'analysis_sets.json'), 'w') as f:
            json.dump(sets, f)

        with open(os.path.join('.', directory, run_name, 'analysis_pop_stats.json'), 'w') as f:
            json.dump(stats, f)

    return(stats)
