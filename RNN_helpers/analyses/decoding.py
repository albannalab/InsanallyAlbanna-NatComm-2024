import sys
sys.path.append('../baysian_neural_decoding/')
import baysian_neural_decoding as bnd
sys.path.remove('../baysian_neural_decoding/')

#sys.path.append('../../AL_data_analysis/')
#import AL_data_analysis as alda
#sys.path.remove('../../AL_data_analysis/')

sys.path.append('..')
from RNN_helpers import parse_activity, load_params, load_meta, binarize_inputs, binarize_outputs, find_response_time
sys.path.remove('..')

import warnings; warnings.filterwarnings('ignore')
import os, pickle, json

import datetime, time
import numpy
import pandas as pd
import pdb
from random import shuffle
from itertools import chain, combinations

from ..io import SAVED_DIR, RESULTS_DIR


#START ALDA IMPORTS
def calc_function_over_reps(dict, key, function):
    results = {}
    for i in dict.keys():
        results[i] = numpy.array([function(item) for item in dict[i][key]])
    return(results)

def calc_statistic_over_reps(dict, statistic):
    results = {}
    for key, vals in dict.items():
        results[key] = statistic(vals)
    return(results)

def calc_hit_and_false_alarm_rates(joint_counts):
    joint_counts = numpy.array(joint_counts, dtype='float')
    hit_rate = joint_counts[0,0] / (joint_counts[0,0] + joint_counts[0,1])
    false_alarm_rate = joint_counts[1,0] / (joint_counts[1,0] + joint_counts[1,1])
    if hit_rate == 1:
        hit_rate = 1 - 1/(2.*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 1:
        false_alarm_rate = 1 - 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))
    if hit_rate == 0:
        hit_rate = 1/(2*(joint_counts[0,0] + joint_counts[0,1]))
    if false_alarm_rate == 0:
        false_alarm_rate = 1/(2.*(joint_counts[1,0] + joint_counts[1,1]))
    return hit_rate, false_alarm_rate

def sensitivity_rescale(joint_counts):
    hit_rate, false_alarm_rate = calc_hit_and_false_alarm_rates(joint_counts)
    return((hit_rate - false_alarm_rate + 1.) / 2.)
#END ALDA IMPORTS




def current_time(abbreviated = False):
    if not abbreviated:
        time_string = '%Y-%m-%d %H:%M:%S'
    else:
        time_string = '%Y%m%d-%H%M%S'
    return(datetime.datetime.fromtimestamp(time.time()).strftime(time_string))

def append_log(file_name, message, echo=False):
    full_message = '[' + current_time() + '] ' + message + '\n'
    with open(file_name, "a") as f:
        f.write(full_message)
    if echo:
        print(full_message)
    pass

def convert_data_format(s, a, np, data):
    c = (s == 'T')*(a == 'NP')+(s == 'F')*(a == 'W')

    return({'all_stimulus': [ar['stimulus']['counts_summary'] for ar in data],
            'all_action': [ar['choice']['counts_summary'] for ar in data],
            'correct_stimulus': [ar['stimulus']['correct_counts_summary'] for ar in data],
            'correct_action': [ar['choice']['correct_counts_summary'] for ar in data],
            'incorrect_stimulus': [ar['stimulus']['incorrect_counts_summary'] for ar in data],
            'incorrect_action': [ar['choice']['incorrect_counts_summary'] for ar in data],
            'all_stimulus_probs': [ar['stimulus']['probs_summary'] for ar in data],
            'all_action_probs': [ar['choice']['probs_summary'] for ar in data],
            'correct_stimulus_probs': [ar['stimulus']['correct_probs_summary'] for ar in data],
            'correct_action_probs': [ar['choice']['correct_probs_summary'] for ar in data],
            'incorrect_stimulus_probs': [ar['stimulus']['incorrect_probs_summary'] for ar in data],
            'incorrect_action_probs': [ar['choice']['incorrect_probs_summary'] for ar in data],
            'stimulus_choices': [ar['stimulus']['counts'] for ar in data],
            'action_choices': [ar['choice']['counts'] for ar in data],
            'stimulus_probs': [ar['stimulus']['probs'] for ar in data],
            'action_probs': [ar['choice']['probs'] for ar in data],
            'stimulus_times': [ar['stimulus']['times'] for ar in data],
            'action_times': [ar['choice']['times'] for ar in data],
            'stimulus': s,
            'action': a,
            'nosepokes': np,
            'correct' : c})





#@do_profile(follow=[bnd.main_script, bnd.probability_trace, bnd.find_all_probs_script, bnd.find_all_params_script, bnd.ISI_from_poisson])
def animal_script(run, directory, neuron, r1, log_file, RESP_FUNCTION, PROB_FUNCTION, PRE_FUNCTION, NUM_FOLDS, NUM_REPETITIONS, FLAGS, case='whole_trial', lock=False, spike_cutoff=3, start_trial=1, **kwargs):
        # Collecting the basic information
        multiple = FLAGS['multiple']
        append_log(log_file, "  Loading data...")
        params = load_params(run, directory=directory)
        meta = load_meta(run, directory=directory)
        trial_duration = params["T"] - params["stim_offset"] / 1000
        pre_trial_duration = params["stim_offset"] / 1000

        # TODO: modify so that this will load once
        # if not multiple:
        #     neuron = [neuron]

        s0 = binarize_inputs(run, directory=directory, start_trial=start_trial)
        a0 = binarize_outputs(run, directory=directory, start_trial=start_trial)
        np0 = find_response_time(run, directory=directory, start_trial=start_trial)
        np0 = numpy.array(np0) / 1000
        #r0 = parse_activity(run, directory=directory, start_trial=start_trial)

        if not multiple:
            r0 = r1[neuron]

        # Filtering out low spike counts
        cutoff_filter = bnd.spike_cutoff_script(r0, multiple=multiple, spike_cutoff=spike_cutoff)
        #st0 = st0[cutoff_filter]
        s0 = s0[cutoff_filter]
        a0 = a0[cutoff_filter]
        np0 = np0[cutoff_filter]
        if multiple:
            r0 = [r[cutoff_filter] for r in r0]
        else:
            r0 = r0[cutoff_filter]

        # Replacing withhold trials with average response time
        avg_np = numpy.nanmean(np0)
        trial_times = np0[:]
        trial_times[numpy.isnan(trial_times)] = avg_np
        num_trials = len(s0)

        # Needed if using set bins
        def round_to_bin(value, bin_size):
            return((int(value // bin_size) + 1) * bin_size)

        if FLAGS['lock']:
            bin_size = FLAGS['bin_size']
            trial_times  = numpy.array([round_to_bin(i, bin_size) for i in trial_times])

        # Creating spikes, inference times, and offsets on a case by case basis
        if case is 'whole_trial':
            # for standard analysis
            s_offset = numpy.matrix(num_trials*[0]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[0], num_trials*[numpy.max(trial_times)])])
            s_trial_times = numpy.matrix([*zip(num_trials*[0], trial_times[:])])
            c_offset = numpy.matrix(trial_times[:]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[-numpy.max(trial_times)], num_trials*[0])])
            c_trial_times = numpy.matrix([*zip(-trial_times[:], num_trials*[0])])
        elif case is 'stim_choice_period':
            # for standard analysis
            s_offset = numpy.matrix(num_trials*[0]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[0], num_trials*[numpy.max(trial_times)])])
            s_trial_times = numpy.matrix([*zip(num_trials*[0], num_trials*[.1])])
            c_offset = numpy.matrix(trial_times[:]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[-numpy.max(trial_times)], num_trials*[0])])
            c_trial_times = numpy.matrix([*zip(num_trials*[-.5], num_trials*[0])])
        elif case is 'whole_trial_pre':
            # for standard analysis
            window = FLAGS['window']
            s_offset = numpy.matrix(num_trials*[0]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[- window], num_trials*[numpy.max(trial_times) + window])])
            s_trial_times = numpy.matrix([*zip(num_trials*[0], trial_times[:])])
            c_offset = numpy.matrix(trial_times[:]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[-numpy.max(trial_times) - window], num_trials*[window])])
            c_trial_times = numpy.matrix([*zip(-trial_times[:], num_trials*[0])])
        elif case is 'first_second':
            # for standard analysis
            trial_length = FLAGS['trial_length']
            s_offset = numpy.matrix(num_trials*[0]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[0], num_trials*[trial_length])])
            s_trial_times = numpy.matrix([*zip(num_trials*[0], num_trials*[trial_length])])
            c_offset = numpy.matrix(num_trials*[0]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[0], num_trials*[trial_length])])
            c_trial_times = numpy.matrix([*zip(num_trials*[0], num_trials*[trial_length])])
        elif case is 'last_second':
            # for standard analysis
            trial_length = FLAGS['trial_length']
            s_offset = numpy.matrix(trial_times[:]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[- trial_length], num_trials*[0])])
            s_trial_times = numpy.matrix([*zip(num_trials*[- trial_length], num_trials*[0])])
            c_offset = numpy.matrix(trial_times[:]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[- trial_length], num_trials*[0])])
            c_trial_times = numpy.matrix([*zip(num_trials*[- trial_length], num_trials*[0])])
        elif case is 'first_second_pre':
            # for standard analysis
            window = FLAGS['window']
            trial_length = FLAGS['trial_length']
            s_offset = numpy.matrix(num_trials*[0]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[- window], num_trials*[trial_length + window])])
            s_trial_times = numpy.matrix([*zip(num_trials*[0], num_trials*[trial_length])])
            c_offset = numpy.matrix(num_trials*[0]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[- window], num_trials*[trial_length + window])])
            c_trial_times = numpy.matrix([*zip(num_trials*[0], num_trials*[trial_length])])
        elif case is 'last_second_pre':
            # for standard analysis
            window = FLAGS['window']
            trial_length = FLAGS['trial_length']
            s_offset = numpy.matrix(trial_times[:]).T
            s_inf_times = numpy.matrix([*zip(num_trials*[- trial_length - window], num_trials*[window])])
            s_trial_times = numpy.matrix([*zip(num_trials*[-trial_length], num_trials*[0])])
            c_offset = numpy.matrix(trial_times[:]).T
            c_inf_times = numpy.matrix([*zip(num_trials*[-trial_length - window], num_trials*[window])])
            c_trial_times = numpy.matrix([*zip(num_trials*[-trial_length], num_trials*[0])])
#         elif case is 'whole_trial_avg':
#             # for control using only prestimulus
#             start_times = num_trials*[0]
#             end_times = num_trials*[avg_np]
#             s_offset = 0
#             a_offset = 0
#         elif case is 'complete_trial':
#             # for control using only prestimulus
#             start_times = num_trials*[0]
#             end_times = num_trials*[trial_duration]
#             s_offset = 0
#             a_offset = 0
#         elif case is 'prestimulus':
#             # for control using only prestimulus
#             start_times = num_trials*[-numpy.max(trial_times)]
#             end_times = num_trials*[0]
#             s_offset = 0
#             a_offset = 0
#         elif case is 'stimulus':
#             # for control using stim only
#             start_times = num_trials*[0]
#             end_times = num_trials*[.1]
#             s_offset = 0
#             a_offset = 0
#         elif case is 'no_stimulus':
#             # for control using w/o stim only
#             start_times = num_trials*[.1]
#             end_times = num_trials*[numpy.max(trial_times)]
#             s_offset = 0
#             a_offset = trial_times[:] - .1
#         elif case is 'first_second':
#             # Using first second
#             start_times = num_trials*[0]
#             end_times = num_trials*[1]
#             s_offset = 0
#             a_offset = 0
#         elif case is 'last_second':
#             # Using last second
#             start_times = num_trials*[0]
#             end_times = num_trials*[1]
#             s_offset = 0
#             a_offset = 0
        else:
            raise ValueError()


        # Converting the responses
        append_log(log_file, "  Converting responses...")

        max_time = numpy.max([s_inf_times + s_offset, s_trial_times + s_offset,  c_inf_times + c_offset,  c_trial_times + c_offset])
        min_time = numpy.min([s_inf_times + s_offset, s_trial_times + s_offset,  c_inf_times + c_offset,  c_trial_times + c_offset])
        if multiple:
            spikes = [[ numpy.array(resp[(resp > min_time)*(resp < max_time)]) for resp in r ] for r in r0]
        else:
            spikes = [ numpy.array(resp[(resp > min_time)*(resp < max_time)]) for resp in r0 ]

        times = {
            'total': (min_time, max_time),
            'stimulus': [s_offset, s_inf_times, s_trial_times],
            'choice': [c_offset, c_inf_times, c_trial_times]
                }

        variable_dict = {
            'stimulus': [{'T':0, 'F':1}, s0],
            'choice': [{'NP':0, 'W':1}, a0]
                }

        condition_variables = {
            'correct': (s0 == 'T')*(a0 == 'NP')+(s0 == 'F')*(a0 == 'W'),
            'incorrect': (s0 == 'T')*(a0 == 'W')+(s0 == 'F')*(a0 == 'NP')
                }

        append_log(log_file, "  Calculating parameters...")

        PARAMS = bnd.pre_script(variable_dict, spikes, times, RESP_FUNCTION, PRE_FUNCTION, condition_variables=condition_variables, num_folds=NUM_FOLDS, **FLAGS)
        #dv.push({'PARAMS': PARAMS})

        #dv.push({'variable_dict' : variable_dict,
        #         'condition_variables' : condition_variables,
        #         'spikes' : spikes,
        #         'times' : times,
        #         'trial_duration' : trial_duration})

        # Running loop on engines
        append_log(log_file, "  Calculating predictions...")
        #%px <below>
        collection = [ bnd.main_script(variable_dict, spikes, times, RESP_FUNCTION, PROB_FUNCTION, condition_variables=condition_variables, num_folds=NUM_FOLDS, params=PARAMS, **FLAGS) for i in range(NUM_REPETITIONS) ]

        # Gathering data from engines
        append_log(log_file, "  Gathering predictions...")
        #collection = dv.gather('collection', block = True)
        return(s0, a0, np0, collection)






#
##rc = Client('/Users/balbanna/.starcluster/ipcluster/SecurityGroup:@sc-MI-us-east-1.json', sshkey='/Users/badr/.ssh/EC2key.rsa')
#rc = Client()
#dv = rc[:]
#dv.block = True
#dv.activate()
#print("Number of active engines: {0}".format(len(dv)))
#



#
#%%px --block
#
#import sys
## AWS path
##sys.path.append('/home/ubuntu/AL_RNN_notebooks/baysian_neural_decoding/')
## ACE cluster path
##sys.path.append('/app/home/balbanna/research/AL_RNN_notebooks/baysian_neural_decoding/')
## Mac path
##sys.path.append('/Users/badr/research/git/AL_RNN_notebooks/baysian_neural_decoding/')
## insan
#sys.path.append('/Users/insanallylab/Documents/Julia Workspace/AL_RNN_notebooks/baysian_neural_decoding/')
#import numpy
#import baysian_neural_decoding as bnd
#from random import shuffle
#%load_ext autoreload
#%autoreload 2
##%run animal_info
#



#
## try:
#from line_profiler import LineProfiler
#
#
#def do_profile(follow=[]):
#    def inner(func):
#        def profiled_func(*args, **kwargs):
#            try:
#                profiler = LineProfiler()
#                profiler.add_function(func)
#                for f in follow:
#                    profiler.add_function(f)
#                profiler.enable_by_count()
#                return func(*args, **kwargs)
#            finally:
#                profiler.print_stats()
#        return profiled_func
#    return inner
#

# except ImportError:
#     def do_profile(follow=[]):
#         "Helpful if you accidentally leave in production!"
#         def inner(func):
#             def nothing(*args, **kwargs):
#                 return func(*args, **kwargs)
#             return nothing
#         return inner



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))






#
####TESTING ON ONE ANIMAL
#def testone(DIR,RUN):
#
#    DIRECTORY = '../multiruns'
#    RUN = 'run 2021-01-07 051523030'
#    #DIRECTORY = '../results_experiments/results_experiment_LIFv2_gross_stdp_200106'
#    #RUN = 'run 2021-01-05 031906879'
#    FILE_NAME = 'decoding_ISI'
#    NEURON = 1
#    START_TRIAL = 1
#
#    NUM_FOLDS = 10
#    NUM_REPETITIONS = 2
#    SPIKE_CUTOFF = 3
#    CASE = 'whole_trial_pre'
#    RESP_FUNCTION = bnd.calc_ISIs
#    PROB_FUNCTION = bnd.timed_prob
#    PRE_FUNCTION = bnd.set_bw_ISI_in_time
#    FLAGS = {'multiple': False,
#        'use_false': False,
#        'within_class': False,
#        'shuffle': False,
#        'at_best': False,
#        # General options
#        'window':  0.2,
#        'step': .1,
#        # ISI options
#        'log': True,
#        'use_PSTH': False,
#        # PSTH options
#        'prob_from_spikes': False,
#        'latency': False,
#        'lock': True,
#        'bin_size': .10,
#        'num_bins': 1,
#        'model': 'kde'}
#
#    log_file = '../test.log'
#    open(log_file, "w").close()
#
#    append_log(log_file, "Sending settings to engines.")
#    #dv.push({'RESP_FUNCTION': RESP_FUNCTION,
#    #     'PROB_FUNCTION': PROB_FUNCTION,
#    #     'PRE_FUNCTION': PRE_FUNCTION,
#    #     'NUM_FOLDS': NUM_FOLDS,
#    #     'NUM_REPETITIONS': NUM_REPETITIONS,
#    #     'FLAGS': FLAGS})
#
#    r0 = parse_activity(RUN, directory=DIRECTORY, start_trial=START_TRIAL, units='s', stim_offset=True)
#    s, a, np, data = animal_script(RUN, DIRECTORY, NEURON, r0, log_file, RESP_FUNCTION, PROB_FUNCTION, PRE_FUNCTION, NUM_FOLDS, NUM_REPETITIONS, FLAGS, spike_cutoff=SPIKE_CUTOFF, case=CASE, start_trial=START_TRIAL)
#    new_data = convert_data_format(s, a, np, data)
#
#    return new_data
#












#Note that if you intend to run control=True you should do it after running control=False, else
#the JSONization will likely fail
def calculate_informativity(run_name,directory=SAVED_DIR,reps=16,control=False):
    print("starting informativity calculation: "+run_name)
    FILE_NAME = 'decoding_ISI'
    use_false = False
    if control:
        FILE_NAME = 'decoding_ISI_control'
        use_false = True

    #62
    NUM_REPETITIONS = reps
    params = load_params(run_name, directory=directory)
    CHOICE_NEURONS = [*range(1,params["N"])]

    NUM_FOLDS = 10
    SPIKE_CUTOFF = 3
    CASE = 'whole_trial_pre'
    START_TRIAL = 1
    print("Is it here?")
    RESP_FUNCTION = bnd.calc_ISIs
    PROB_FUNCTION = bnd.timed_prob
    PRE_FUNCTION = bnd.set_bw_ISI_in_time
    print("Or here?")

    FLAGS = {'multiple': False,
        'use_false': use_false,
        'within_class': False,
        'shuffle': False,
        'at_best': False,
        # General options
        'window':  0.2,
        'step': .1,
        # ISI options
        'log': True,
        'use_PSTH': False,
        # PSTH options
        'prob_from_spikes': False,
        'latency': False,
        'lock': True,
        'bin_size': .10,
        'num_bins': 1,
        'model': 'kde'}
    #num_engines = len(dv)
    #num_engine_reps = int(NUM_REPETITIONS / num_engines)
    multiple = FLAGS['multiple']

    log_file = directory + '/' + run_name + '/' + FILE_NAME + '.log'
    print("Decoding test A")
    if not os.path.exists(directory):
        os.mkdir(directory)
    print("Decoding test A.5")
    if not os.path.exists(log_file):
        open(log_file, "w").close()
    print("Decoding test B")

    append_log(log_file, '*** Starting new run ***', echo=True)
    append_log(log_file, "Sending settings to engines.")
    #dv.push({'RESP_FUNCTION': RESP_FUNCTION,
    #     'PROB_FUNCTION': PROB_FUNCTION,
    #     'PRE_FUNCTION': PRE_FUNCTION,
    #     'NUM_FOLDS': NUM_FOLDS,
    #     'NUM_REPETITIONS': num_engine_reps,
    #     'FLAGS': FLAGS})

    append_log(log_file, "Starting run {0}".format(run_name))

    if not multiple:
        current_file = directory+'/'+run_name+'/'+ FILE_NAME + '.pickle'
        try:
            with open(current_file, 'rb') as f:
                result_array = pickle.load(f)
            append_log(log_file, "  already have {0}...".format(run_name))
        except:
            result_array = {}

    # Picking what combinations
    if multiple:
        neurons = powerset(CHOICE_NEURONS)
    else:
        neurons = CHOICE_NEURONS

    r0 = parse_activity(run_name, directory=directory, start_trial=START_TRIAL, units='s', stim_offset=True)
    for neuron in neurons:
        if multiple and len(neuron) == 0:
            continue
        if multiple:
            current_file = directory + '/' + animal + '-' + str(neuron) + '.pickle'
            try:
                with open(current_file, 'rb') as f:
                    result_array = pickle.load(f)
                append_log(log_file, "  already have {0}, {1}...".format(run_name, neuron))
            except:
                result_array = {}
        if neuron in result_array:
            append_log(log_file, "  ...skipping neuron {0}.".format(neuron))
            continue
        else:
            append_log(log_file, "{0}, neuron {1}".format(run_name, neuron))
        try:
            result_array[neuron] = convert_data_format(*animal_script(run_name, directory, neuron, r0, log_file, RESP_FUNCTION, PROB_FUNCTION, PRE_FUNCTION, NUM_FOLDS, NUM_REPETITIONS, FLAGS, spike_cutoff=SPIKE_CUTOFF, case=CASE, start_trial=START_TRIAL))
        except KeyboardInterrupt:
            append_log(log_file, "KEYBOARD INTERRUPT!")
            raise
        except Exception as e:
            append_log(log_file, "  Problem with {0}, neuron {1}".format(run_name, neuron))
            append_log(log_file,str(e))
            #raise
        finally:
            # Saving the output
            append_log(log_file, "  Saving output...")
            with open(current_file, 'wb') as f:
                pickle.dump(result_array, f)
    #open(DIRECTORY + "/run complete", "wb").close()
    append_log(log_file, "Analysis complete.", echo=True)
    print("finishing informativity calculation: "+run_name)
    return packagePickle(run_name,directory=directory,control=False)

def packagePickle(run_name,directory=SAVED_DIR,control=False):
    print("repackaging informativity calculation: "+run_name)
    EXP_FILE = "decoding_ISI.pickle"
    if control:
        NULL_FILE = "decoding_ISI_control.pickle"

    TAGS = "_test"

    with open(os.path.join(directory, run_name, EXP_FILE), 'rb') as f:
        counts = pickle.load(f)
    if control:
        with open(os.path.join(directory, run_name, NULL_FILE), 'rb') as f:
            null_counts = pickle.load(f)

    stim_all = calc_function_over_reps(counts, 'all_stimulus_probs', sensitivity_rescale)
    stim = calc_statistic_over_reps(stim_all, numpy.mean)
    #stim_sem = alda.calc_statistic_over_reps(stim_all, alda.sem)

    choice_all = calc_function_over_reps(counts, 'all_action_probs', sensitivity_rescale)
    choice = calc_statistic_over_reps(choice_all, numpy.mean)
    #choice_sem = alda.calc_statistic_over_reps(choice_all, alda.sem)

    #if control:
    #    stim_all_null = alda.calc_function_over_reps(null_counts, 'all_stimulus_probs', alda.sensitivity_rescale)
    #    stim_null = alda.calc_statistic_over_reps(stim_all_null, numpy.mean)
    #    choice_all_null = alda.calc_function_over_reps(null_counts, 'all_action_probs', alda.sensitivity_rescale)
    #    choice_null = alda.calc_statistic_over_reps(choice_all_null, numpy.mean)
    #    p_stim = alda.test_sig_over_instances(stim_all, stim_all_null, alda.direct)
    #    p_choice = alda.test_sig_over_instances(choice_all, choice_all_null, alda.direct)

    #Package dataframe

    stim_pd = pd.DataFrame(data=stim.values(), index=stim.keys(), columns=['stim'])
    #stim_sem_pd = pd.DataFrame(data=stim_sem.values(), index=stim_sem.keys(), columns=['stim_sem'])
    choice_pd = pd.DataFrame(data=choice.values(), index=choice.keys(), columns=['choice'])
    #choice_sem_pd = pd.DataFrame(data=choice_sem.values(), index=choice_sem.keys(), columns=['choice_sem'])
    #df = stim_pd.join(stim_sem_pd, how='outer').join(choice_pd, how='outer').join(choice_sem_pd, how='outer')
    df = stim_pd.join(choice_pd, how='outer')

    #if control:
    #    stim_null_pd = pd.DataFrame(data=stim_null.values(), index=stim_null.keys(), columns=['stim_null'])
    #    choice_null_pd = pd.DataFrame(data=choice_null.values(), index=choice_null.keys(), columns=['choice_null'])
    #    p_stim_pd = pd.DataFrame(data=p_stim.values(), index=p_stim.keys(), columns=['p_stim'])
    #    p_choice_pd = pd.DataFrame(data=p_choice.values(), index=p_choice.keys(), columns=['p_choice'])
    #    df = df.join(stim_null_pd, how='outer').join(choice_null_pd, how='outer').join(p_stim_pd, how='outer').join(p_choice_pd, how='outer')

    df_c = df.dropna()

    informativity = json.loads(df_c['stim'].to_json())
    informativity['ascending'] = list(df_c.sort_values('stim',ascending=True)['stim'].index)
    informativity['descending'] = list(df_c.sort_values('stim',ascending=False)['stim'].index)
    print("saving informativity calculation: "+run_name)
    with open(os.path.join(directory,run_name,"analysis_informativity.json"), 'w') as fp:
        json.dump(informativity, fp)
    return informativity
