import numpy as np, statsmodels.api as sm
from .io import load_activity, load_outputs, load_params, load_meta, load_rates, load_weights, load_current, save_convergence, RESULTS_DIR, SAVED_DIR
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

def binarize_inputs(run_name, directory=SAVED_DIR, start_trial=1, input_name="inputs", target_name="target_freq"):
    '''Returns the stimuli presented on each trial of a run as True/False where True = Target, False = Non-target.
    '''
    outputs = load_outputs(run_name, directory=directory)
    params = load_params(run_name, directory=directory)
    target = (outputs[input_name] == params[target_name])
    target = target[start_trial-1:]
    return(np.array(['T' if i else 'F' for i in target]))


def all_inputs(run_name, directory=SAVED_DIR, input_name="inputs"):
    '''Returns the stimuli presented on each trial of a run as the value of the input frequency. 
    '''
    outputs = load_outputs(run_name, directory=directory)
    return(outputs[input_name])


def binarize_outputs(run_name, directory=SAVED_DIR, start_trial=1, input_name="inputs", output_name="outputs"):
    '''Returns the network output on each trial of a run to True/False where True = Go, False = No-go
    '''
    outputs = load_outputs(run_name, directory=directory)
    target = (outputs[input_name] == 2.0)
    target = target[start_trial-1:]
    out = np.array(outputs[output_name]).reshape(-1, 1)
    out = out[start_trial-1:]
    reg = LogisticRegressionCV(cv=10)
    reg.fit(out, target)
    out_class = reg.predict(out)
    return(np.array(['NP' if i else 'W' for i in out_class]))


def find_response_time(run_name, directory=SAVED_DIR, start_trial=1, version="v2"):
    '''Returns the response times on each trial of a run.
    '''
    if version == "v1":
        params = load_params(run_name, directory=directory)
        nRunTot = params['nRunTot'] - start_trial + 1
        dt = params["dt"]
        rOffset = (params['rOffset'] + params['rDur']) * dt
        return(np.array(nRunTot * [rOffset]))
    elif version == "v2":
        meta = load_meta(run_name, directory=directory)
        params = load_params(run_name, directory=directory)        
        n_trials = meta['n_trials'] - start_trial + 1
        resp_offset = (params['resp_offset'] + params['resp_dur'])
        return(np.array(n_trials * [resp_offset]))


def find_response_range(run_name, directory=SAVED_DIR, version="v2"):
    if version=="v1":
        params = load_params(run_name, directory=directory)
        dt = params["dt"]
        return(params['rOffset']*dt, (params['rOffset'] + params['rDur']) * dt)
    elif version=="v2":
        params = load_params(run_name, directory=directory)
        return(params['resp_offset'], params['resp_offset'] + params['resp_dur'])


def find_stimulus_range(run_name, directory=SAVED_DIR, version="v2"):
    if version=="v1":
        params = load_params(run_name, directory=directory)
        dt = params["dt"]
        return(params['rOffset']*dt, (params['rOffset'] + params['rDur']) * dt)
    elif version=="v2":
        params = load_params(run_name, directory=directory)
        return(params['stim_offset'], params['stim_offset'] + params['stim_dur'])


def parse_activity(run_name, directory=SAVED_DIR, version="v2", start_trial=1, units='ms', stim_offset=False):
    activity = load_activity(run_name, directory=directory)
    if version == "v1":
        nRun, nCells, _ = activity.shape
        return(np.array([[activity[run, cell][activity[run, cell] != -1] for run in range(nRun)] for cell in range(nCells)]))
    elif version =="v2":
        trial = np.array(activity[0,:], dtype="int")
        spike_id = np.array(activity[1,:], dtype="int")
        params = load_params(run_name, directory=directory)
        meta = load_meta(run_name, directory=directory)
        N = params["N"]
        n_trials = meta["n_trials"]
        spike_times = np.array(activity[2,:], dtype="float")
        total_activity = []
        #for i in range(1, N+1):
        for i in range(1, N+1):
            cell_activity = []
            idx = (spike_id==i)
            cell_idx = trial[idx]
            cell_spike_times = spike_times[idx]
            for j in range(start_trial, n_trials+1):
                if stim_offset:
                    times = np.array(cell_spike_times[cell_idx==j]) - params['stim_offset']
                else:
                    times = np.array(cell_spike_times[cell_idx==j]) 
                if units == 'ms':
                    cell_activity.append(times)
                elif units == 's':
                    cell_activity.append(times/1000)
            total_activity.append(cell_activity)
        return np.array(total_activity, dtype=object)

