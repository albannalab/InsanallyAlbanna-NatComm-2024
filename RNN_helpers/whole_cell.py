from tracemalloc import start
from urllib import response
import numpy as np
import pandas as pd
import os
from collections import Counter
from random import choice

WC_DIR = "../results_experiments/whole_cell/"
PRE_INPUT_FILE = "MNI pre tone period BFA edit.xlsx"
POST_INPUT_FILE = "MNI tone period BFA edit.xlsx"
SPIKE_FILE = "MNI spiking responses BFA edit.xlsx"
BEHAVIOR_FILE = "MNI responses BFA edit.xlsx"

def identify_recordings(cell_df, w_inputs=True, w_spikes=False, behavior_only=False, include_list=False, exclusion=True, show_dir=True):
    temp_df = cell_df[:]
    temp_df = temp_df.set_index(["date", "mouse", "cell"])
    
    included_recordings_list = []
    included_recordings = {}

    for id, data in temp_df.groupby(level=(0,1,2)):
        if exclusion:
            exclude = data["exclude"]
        else:
            exclude = np.array([False] * len(data), dtype='bool')
        
        if behavior_only: 
            data = data[data["behavior"] == True]
            exclude = exclude[data["behavior"] == True]
        exc = data["exc"] & ~exclude
        inh = data["inh"] & ~exclude
        spikes = data["attach"] & ~exclude

        if w_inputs & w_spikes:
            include = np.any(exc) & np.any(inh) & np.any(spikes)
        elif w_inputs:
            include = np.any(exc) & np.any(inh)
        elif w_spikes: 
            include = np.any(spikes)
        if not include:
            continue

        if (id not in included_recordings.keys()):
            included_recordings[id] = {}
            if show_dir:
                included_recordings[id]["dir"] = data["directory"].iloc[0]
        if w_inputs:
            included_recordings[id]['exc'] = [(str(row["filename"]), row["stim_time"], row["n_trials"]) for i, row in data[exc].iterrows()]
            included_recordings[id]['inh'] = [(str(row["filename"]), row["stim_time"], row["n_trials"]) for i, row in data[inh].iterrows()]
            included_recordings_list.extend([str(row["filename"]) for i, row in data[exc].iterrows()])
            included_recordings_list.extend([str(row["filename"]) for i, row in data[inh].iterrows()])
        if w_spikes:
            included_recordings[id]['spikes'] = [(str(row["filename"]), row["stim_time"], row["n_trials"]) for i, row in data[spikes].iterrows()]
        included_recordings_list.extend([str(row["filename"]) for i, row in data[spikes].iterrows()])
    if include_list:
        return included_recordings, included_recordings_list
    else:
        return included_recordings

def calculate_modulation_inputs(recording_ids, directory=WC_DIR, method='paired', N=2000, pre_filename=PRE_INPUT_FILE, post_filename=POST_INPUT_FILE):
    assert method in ['paired', 'random']
    pre_spikes = []
    post_spikes = []
    for i, (id, t_stim, n_trials) in enumerate(recording_ids): 
        pre_df = pd.read_excel(os.path.join(directory, pre_filename), sheet_name=id, engine='openpyxl', usecols=range(17))
        post_df = pd.read_excel(os.path.join(directory, post_filename), sheet_name=id, engine='openpyxl', usecols=range(17))
        pre_spikes.extend(list(pre_df["trace"] + i*100))
        post_spikes.extend(list(post_df["trace"] + i*100))
    pre_counts = Counter(pre_spikes)
    post_counts = Counter(post_spikes)
    all_trials = list(set(pre_counts.keys()).union(set(post_counts.keys())))
    if method == 'paired':
        delta = [post_counts[t] - pre_counts[t] for t in all_trials]
    elif method == 'random':
        delta = [post_counts[choice(all_trials)] - pre_counts[choice(all_trials)] for i in range(N)]
    modulation = np.mean(delta)
    return modulation


def find_all_spikes(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE):
    spike_times = []
    trials = []
    n_trials_total = 0
    for id, stim_time, n_trials in recording_ids:
        spike_df = pd.read_excel(os.path.join(directory, spike_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        spike_idx = (spike_df['trace'] <= n_trials)
        spike_times.extend(list(spike_df[spike_idx]['peak time']))
        trials.extend(list(spike_df[spike_idx]['trace'] - 1 + n_trials_total))
        n_trials_total += n_trials
    return(np.array(spike_times), np.array(trials), int(n_trials_total))


def calculate_active_fraction(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE, spike_thresh=3):
    n_trials_total = np.int(np.sum([n_trials for _, _, n_trials in recording_ids]))
    spikes_over_thresh = np.zeros(n_trials_total)
    start_trial=0
    for id, stim_time, n_trials in recording_ids:
        n_trials = int(n_trials)
        spike_df = pd.read_excel(os.path.join(directory, spike_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        spike_df = spike_df.set_index('trace').sort_values('trace')
        for j in range(n_trials):
            try:
                current_trial_spike_times = spike_df.loc[j]["peak time"]
            except:
                continue
            current_n_spikes = current_trial_spike_times.size
            spikes_over_thresh[start_trial + j] = (current_n_spikes >= spike_thresh)
        start_trial += n_trials
    return(np.mean(spikes_over_thresh))

def calculate_average_n_spikes(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE, stim_duration=100, pre_duration=100):
    n_trials_total = np.int(np.sum([n_trials for _, _, n_trials in recording_ids]))
    n_spikes = np.zeros(n_trials_total)
    start_trial=0
    for id, stim_time, n_trials in recording_ids:
        n_trials = int(n_trials)
        spike_df = pd.read_excel(os.path.join(directory, spike_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        spike_df = spike_df.set_index('trace').sort_values('trace')
        pre_time = stim_time - pre_duration
        for j in range(n_trials):
            try:
                current_trial_spike_times = spike_df.loc[j]["peak time"]
            except:
                continue
            current_n_spikes = current_trial_spike_times.size 
            n_spikes[start_trial + j] = current_n_spikes
        start_trial += n_trials
    return(np.mean(n_spikes))

def calculate_baseline_firing_rate(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE, pre_duration=100, verbose=False):
    n_trials_total = np.int(np.sum([n_trials for _, _, n_trials in recording_ids]))
    baseline_fr = np.zeros(n_trials_total)
    start_trial=0
    for id, stim_time, n_trials in recording_ids:
        n_trials = int(n_trials)
        spike_df = pd.read_excel(os.path.join(directory, spike_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        spike_df = spike_df.set_index('trace').sort_values('trace')
        pre_time = stim_time - pre_duration
        for j in range(n_trials):
            try:
                current_trial_spike_times = spike_df.loc[j]["peak time"]
            except:
                continue
            n_pre_spikes = np.sum((current_trial_spike_times >= pre_time) & (current_trial_spike_times < (pre_time + pre_duration))) 
            baseline_fr[start_trial + j] = n_pre_spikes*1000/pre_duration
        start_trial += n_trials
    if verbose: print("FR:", baseline_fr)
    return(np.mean(baseline_fr))

def calculate_stimulus_modulation(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE, stim_duration=100, pre_duration=100, verbose=False):
    n_trials_total = np.int(np.sum([n_trials for _, _, n_trials in recording_ids]))
    stimulus_modulation = np.zeros(n_trials_total)
    start_trial=0
    for id, stim_time, n_trials in recording_ids:
        n_trials = int(n_trials)
        spike_df = pd.read_excel(os.path.join(directory, spike_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        spike_df = spike_df.set_index('trace').sort_values('trace')
        pre_time = stim_time - pre_duration
        for j in range(n_trials):
            try:
                current_trial_spike_times = spike_df.loc[j]["peak time"]
            except:
                continue
            n_pre_spikes = np.sum((current_trial_spike_times >= pre_time) & (current_trial_spike_times < (pre_time + pre_duration))) 
            n_stim_spikes = np.sum((current_trial_spike_times >= stim_time) & (current_trial_spike_times < (stim_time + stim_duration)))
            stimulus_modulation[start_trial + j] = n_stim_spikes*1000/stim_duration - n_pre_spikes*1000/pre_duration
        start_trial += n_trials
    if verbose: print("Stimulus:", stimulus_modulation)
    return(np.mean(stimulus_modulation))


def calculate_response_times(recording_ids, directory=WC_DIR, behavior_filename=BEHAVIOR_FILE):
    n_trials_total = np.int(np.sum([n_trials for _, _, n_trials in recording_ids]))
    response_times = np.zeros(n_trials_total)

    # Compile responses
    start_trial=0
    for id, stim_time, n_trials in recording_ids:
        n_trials = int(n_trials)
        behavior_df = pd.read_excel(os.path.join(directory, behavior_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        behavior_df = behavior_df.set_index('trace').sort_values('trace')
        for j in range(n_trials):
            try:
                current_response_time = np.min(behavior_df.loc[j]["start"])
                response_times[j + start_trial] = current_response_time
            except:
                response_times[j + start_trial] = np.nan
        start_trial += n_trials
    return response_times

def calculate_choice_modulation(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE, behavior_filename=BEHAVIOR_FILE, pre_duration=100, stim_duration=100, average_response=500, choice_duration=300, window_type='prior', verbose=False):
    n_trials_total = np.int(np.sum([n_trials for _, _, n_trials in recording_ids]))
    choice_modulation = np.zeros(n_trials_total)

    response_times = calculate_response_times(recording_ids, directory=directory, behavior_filename=behavior_filename)

    # Calculate average response
    if np.all(np.isnan(response_times)):
        response_times = np.nan_to_num(response_times, nan=average_response)
    else:
        average_response = np.nanmean(response_times)
        response_times = np.nan_to_num(response_times, nan=average_response)

    # Calculate choice modulation
    start_trial = 0
    for id, stim_time, n_trials in recording_ids:
        n_trials = int(n_trials)
        spike_df = pd.read_excel(os.path.join(directory, spike_filename), sheet_name=id, usecols=range(17), engine='openpyxl')
        spike_df = spike_df.set_index('trace').sort_values('trace')
        pre_time = stim_time - pre_duration
        for j in range(n_trials):
            try:
                current_trial_spike_times = spike_df.loc[j]["peak time"]
            except:
                continue
            
            n_pre_spikes = np.sum((current_trial_spike_times >= pre_time) & (current_trial_spike_times < (pre_time + pre_duration)))
            current_response_time = response_times[j + start_trial]
            
            if window_type == 'prior':
                if current_response_time <= (stim_time + stim_duration + choice_duration):
                    current_response_time = (stim_time + stim_duration + choice_duration)
                n_choice_spikes = np.sum((current_trial_spike_times >= (current_response_time - choice_duration)) & (current_trial_spike_times < current_response_time))
            elif window_type == 'symmetric':
                if current_response_time <= (stim_time + stim_duration + choice_duration / 2.):
                    current_response_time = (stim_time + stim_duration + choice_duration / 2.)
                n_choice_spikes = np.sum((current_trial_spike_times >= (current_response_time - choice_duration / 2.)) & (current_trial_spike_times < current_response_time + choice_duration / 2.))

            if verbose: print(current_response_time, choice_duration, current_trial_spike_times)
            
            choice_modulation[start_trial + j] = n_choice_spikes*1000/choice_duration - n_pre_spikes*1000/pre_duration
        start_trial += n_trials 
    if verbose: print("Choice:", choice_modulation)
    return(np.mean(choice_modulation))


def calculate_modulation(recording_ids, directory=WC_DIR, spike_filename=SPIKE_FILE, behavior_filename=BEHAVIOR_FILE, pre_duration=100, stim_duration=100, average_response=1000, choice_duration=500, choice_window_type='prior', verbose=False):
    stimulus_modulation = calculate_stimulus_modulation(recording_ids, stim_duration=stim_duration, pre_duration=pre_duration, directory=directory, spike_filename=spike_filename, verbose=verbose)
    choice_modulation = calculate_choice_modulation(recording_ids, directory=directory, spike_filename=spike_filename, behavior_filename=behavior_filename, stim_duration=stim_duration, average_response=average_response, choice_duration=choice_duration, window_type=choice_window_type, verbose=verbose)
    total_modulation = np.sqrt(stimulus_modulation**2 + choice_modulation**2)
    return(stimulus_modulation, choice_modulation, total_modulation)


def load_recording(recording_id, directory=WC_DIR, period='pre', t_pre=-50, t_stim=200, set_peak=None, drop_missing=True):
    assert period in ['pre', 'post']
    if period == 'pre':
        filename = "MNI pre tone period BFA edit.xlsx"
    elif period == 'post':
        filename = "MNI tone period BFA edit.xlsx"
    df = pd.read_excel(os.path.join(directory, filename), sheet_name=recording_id, engine='openpyxl',usecols=range(17))
    df = df.fillna(df.mean())
    if set_peak is not None:
        df["peak time"] = set_peak
    else:
        if period == 'post':
            df["peak time"] -= t_stim
            df["start"] -= t_stim
            df["end"] -= t_stim
        elif period == 'pre':
            df["peak time"] = t_pre
    # TODO: deal with multiple peaks
    if not drop_missing:
        trace_nums = df["trace"]
        n_traces = int(np.max(trace_nums))
        for j in range(1, n_traces+1):
            if np.all(j != trace_nums):
                df = df.append({'trace': j}, ignore_index=True)
    df = df.astype({'trace': int})
    df = df.sort_values('trace')
    return df 

def load_pulse_data(recording_id, directory=WC_DIR):
    filename = "sweep stats BFA.xlsx"
    try:
        df = pd.read_excel(os.path.join(directory, filename), sheet_name=recording_id, engine='openpyxl', usecols=range(10))
        df = df.astype({'trace': int})
        df = df.sort_values('trace')
    except:
        df = None
    return df

def process_pulse_data(df, filter=False, alternate_method=False):
    if alternate_method:
        if "std" in df.columns:
            idx = (6*np.abs(df["std"]) > np.abs(df["I_pulse_inf"])) | (df["I_pulse_inf"] > 0)
            df.loc[idx, "I_pulse_inf"] = -6*df.loc[idx, "std"]
        else:
            df["I_pulse_inf"] = np.nan
            print(df["I_pulse_inf"].head(5))
    if not filter:
        df["I_pulse_inf"] = - np.abs(df["I_pulse_inf"])
    df['R_s'] = df["V_step"] / df["I_pulse_0"] * 1000 # in megaOhms
    df['R_in'] = df["V_step"] / df["I_pulse_inf"] * 1000 # in megaOhms

    df['R_m'] = df['R_in'] - df['R_s'] # in megaOhms
    df['C'] = - (df["I_decay_time"] - df["I_pulse_0_time"]) / ((np.log((df["I_decay"] - df["I_pulse_inf"]) / (df["I_pulse_0"] - df["I_pulse_inf"] ))) * df['R_m']) 
    if filter:
        indx =(df["R_s"] > 0.0) & (df["R_m"] > 0.0) & (df["R_in"] > 0)
        df = df.loc[indx]
    return df 
    
def load_all_data(recording_id, directory=WC_DIR, drop_missing=True, set_post_peak=None, set_pre_peak=None, t_stim=200, C_default=2.0, R_default=10.0, filter=True, alternate_method=False):
    pre_df = load_recording(recording_id, directory=directory, period='pre', drop_missing=drop_missing, set_peak=set_pre_peak)
    pre_df["period"] = "pre"
    
    post_df = load_recording(recording_id, directory=directory, period='post', drop_missing=drop_missing,set_peak=set_post_peak, t_stim=t_stim)
    post_df["period"] = "post"

    pulse_df = load_pulse_data(recording_id, directory=directory)

    if pulse_df is not None:
        pulse_df = process_pulse_data(pulse_df, filter=filter, alternate_method=alternate_method)
        pre_df = pd.merge(pre_df, pulse_df, on="trace", how='left')
        post_df = pd.merge(post_df, pulse_df, on="trace", how='left')
    else: # filling with 
        pre_df['C'] = C_default
        post_df['R'] = R_default
    
    df = pre_df.append(post_df)
    df = df.reset_index()#.set_index(["period", "trace"])
    return df

def load_exc_and_inh(exc_ids, inh_ids, V_e = 0, V_i = -70, directory=WC_DIR, drop_missing=True, set_post_peak=None, set_pre_peak=None, t_stim=200, C_default=2.0, R_default=10.0, filter=False, alternate_method=False, limit_trials=True):
    n_start = 0
    for i, (id, t_stim, n_trials) in enumerate(exc_ids):
        print(f"Processing exc. {id}")
        temp_df = load_all_data(id, 
            directory=directory, 
            drop_missing=drop_missing, 
            set_post_peak=set_post_peak, 
            set_pre_peak=set_pre_peak, 
            t_stim=t_stim, 
            C_default=C_default, 
            R_default=R_default, 
            filter=filter,
            alternate_method=alternate_method)
        temp_df["id"] = id
        temp_df["EI"] = 'exc'
        temp_df["V_h"] = V_i
        temp_df["V_r"] = V_e
        if limit_trials:
            n_trials = temp_df["trace"].max()
        temp_df["trace"] += n_start
        if i == 0:
            df = temp_df
        else:
            df = df.append(temp_df)
        n_start += n_trials
    df = df.astype({'trace': int})
    
    n_start = 0
    for id, t_stim, n_trials in inh_ids:
        print(f"Processing inh. {id}")
        temp_df = load_all_data(id, 
        directory=directory, 
        drop_missing=drop_missing, 
        set_post_peak=set_post_peak, 
        set_pre_peak=set_pre_peak, 
        t_stim=t_stim, 
        filter=filter, 
        alternate_method=alternate_method)
        temp_df["id"] = id
        temp_df["EI"] = 'inh'
        temp_df["V_h"] = V_e
        temp_df["V_r"] = V_i
        if limit_trials:
            n_trials = temp_df["trace"].max()
        temp_df["trace"] += n_start
        df = df.append(temp_df)  
        n_start += n_trials
    df = df.astype({'trace': int})
    df = df.reset_index().set_index(["period", "EI", "trace"])
    return df

def I_curve(max_value, max_time, rise_time_to_half, decay_time_to_half):
    if np.isnan(max_value):
        def I(t):
            return 0.0
    else:
        def I(t):
            offset = rise_time_to_half * np.log2(decay_time_to_half / rise_time_to_half + 1.0)
            if (t > (max_time - offset)):
                return (max_value * (1.0 - 2**(-(t- max_time + offset)/rise_time_to_half)) * 2**(-(t - max_time - offset) / decay_time_to_half))
            else:
                return 0.0
    return np.vectorize(I)

def calculate_conductance(df, dt=1, T=150, scale=1.0, fixed_g_scale=None, R_adjust=True):
    DF_max = df["peak amp"]


    time_max = df["peak time"]
    rise_time_to_half = df["time to rise half-amplitude"]
    decay_time_to_half = df["time to decay half-amplitude"]

    V_h = df["V_h"]
    V_r = df["V_r"]
    R_s = df["R_s"]
    R_m = df["R_m"]
    R_in = df["R_in"]

    if scale != 1.0:
        fixed_I_scale = fixed_g_scale * (V_h - V_r) * (R_m) / (R_m + R_s)
        DF_max = (DF_max - fixed_I_scale)*scale + fixed_I_scale
    try:
        DI = I_curve(DF_max, time_max, rise_time_to_half, decay_time_to_half)
        Ts = np.arange(0, T, dt)
        DIs = DI(Ts)
        if R_adjust:
            I_syn = DIs * (R_m + R_s) /  R_m
        else:
            I_syn = DIs
        g = I_syn / (V_h - V_r) # in nS
    except:
        print(DF_max, time_max, rise_time_to_half, decay_time_to_half)
        raise

    return scale * g

def sample_path(T):
    dt = 1.0 / (T - 1)                                                          
    dt_sqrt = np.sqrt(dt)
    noise = np.empty(T, dtype=np.float32)
    noise[0] = 0
    for n in range(T - 2):                                           
         t = n * dt
         xi = np.random.randn() * dt_sqrt
         noise[n + 1] = noise[n] * (1 - dt / (1 - t)) + xi
    noise[-1] = 0
    return np.abs(noise)

def construct_prior(x, scale):
    def relative_prior(y):
        return np.exp(-(y-x)**2 / (2*scale**2))
    return np.vectorize(relative_prior)

def simulate_neuron(PSC_df, N=200, T=150, period='post', RMP=-60, threshold=-30, t_refrac=5, noise=0.1, noise_func=np.random.randn, scale_noise=True, E_exc=0, E_inh=-70, record_g=False, C=None, R=None, tau=None, selection_method="random", prior_scale=2, scale=1.0, fixed_exc_scale=None, fixed_inh_scale=None):
    assert selection_method in ["random", "matched"]
    if C is None:
        PSC_df["C"] = PSC_df["C"].replace([np.inf, -np.inf], np.nan)
        C = PSC_df["C"].dropna().mean()
        C *= 1000
    else:
        C = float(C)

    if R is None:
        PSC_df["R_m"] = PSC_df["R_m"].replace([np.inf, -np.inf], np.nan)
        R = PSC_df["R_m"].dropna().mean()
        R_s = PSC_df["R_s"].dropna().mean()
        R_in = PSC_df["R_in"].dropna().mean()
        R /= 1000
        R_s /= 1000
        R_in /= 1000
    else:
        R = float(R)
        R_s = 0.0
        R_in = 0.0

    if tau is not None:
        C = tau / R

    print(f"Simulating {period} w/ C = {C:.4}, R = {R:.4}, (R_s = {R_s:.4}, R_in = {R_in:.4}), tau = {R*C:.4}")

    # Setting up variables
    V_m_t = np.zeros((N, T)) # history of membrane potential (mV)
    V_m_t[:,0] = RMP
    refrac_counter = 0
    postsyn_spikes = np.zeros((N, T)) # binary spike train history of postsyn cell
    postsyn_spikes_list = N*[[], ] #  spike train history of postsyn cell using times
    if record_g:
        g_t = np.zeros((N, T))

    i_df = PSC_df.loc[period, 'inh']
    e_df = PSC_df.loc[period, 'exc']

    # run simulation
    for n in range(N):
        if selection_method == "random":
            g_ex_df = e_df.iloc[np.random.randint(len(e_df))]
            g_inh_df = i_df.iloc[np.random.randint(len(i_df))]
        elif selection_method == "matched":
            g_ex_df = e_df.iloc[np.random.randint(len(e_df))]
            matching_amp = np.abs(g_ex_df["peak amp"])
            prior = construct_prior(matching_amp, prior_scale)
            relative_prob = prior(i_df["peak amp"])
            relative_prob /= np.sum(relative_prob)
            cum_prob = np.cumsum(relative_prob)
            inh_indx = np.argmax(cum_prob > np.random.rand())
            g_inh_df = i_df.iloc[inh_indx]

        g_ex_t = calculate_conductance(g_ex_df, T=T, scale=scale, fixed_g_scale=fixed_exc_scale)
        g_inh_t = calculate_conductance(g_inh_df, T=T, scale=scale, fixed_g_scale=fixed_inh_scale)

        if scale_noise:
            g_ex_t = noise * g_ex_t * noise_func(len(g_ex_t))
            g_inh_t = noise * g_inh_t * noise_func(len(g_inh_t))
        else:
            g_ex_t += noise * noise_func(len(g_ex_t))
            g_inh_t += noise * noise_func(len(g_inh_t))
        g_t[n,:] = g_ex_t - g_inh_t

        for t in range(T):
            refrac_counter -= 1
            if (refrac_counter <= 0):
                V_m_t[n,t] += ((RMP - V_m_t[n,t]) / (R*C)) + (g_ex_t[t] * (E_exc - V_m_t[n,t]) / C) + (g_inh_t[t] * (E_inh - V_m_t[n,t]) / C) 

            # reinit spike generator if previously fired
            if ((t+1) < T):
                if (V_m_t[n,t] > threshold):
                    postsyn_spikes[n,t+1] = 1
                    postsyn_spikes_list[n].append(t)
                    V_m_t[n, t+1] = RMP
                    refrac_counter = t_refrac
                else:
                    V_m_t[n,t+1] = V_m_t[n,t]
    if record_g:
        return postsyn_spikes, postsyn_spikes_list, V_m_t, g_t
    else:
        return postsyn_spikes, postsyn_spikes_list, V_m_t

def calculate_peak_conductance(data_df):
    # Finding resistances 
    # data_df['R_s'] = data_df["V_step"] / data_df["I_pulse_0"] # in megaOhms
    # data_df['R_in'] = data_df["V_step"] / data_df["I_pulse_inf"] - data_df['R_s'] # in megaOhms

    # Calculating synaptic current
    data_df["I_syn"] = data_df["peak amp"] * (data_df["R_in"] + data_df["R_s"]) /  data_df["R_in"] 
    
    # Calculating conductances 
    data_df["g"] = data_df["I_syn"] / (data_df["V_h"] - data_df["V_r"]) # in nS
    return data_df


