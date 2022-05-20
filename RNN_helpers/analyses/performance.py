import numpy as np
import os
import json
from scipy.stats import norm
from ..io import SAVED_DIR
from ..analysis import binarize_inputs, binarize_outputs, confusion_matrix, all_inputs

def F1_score(cm, prior=np.array([[1,1], [1,1]])):
    precision = (cm[1,1] + prior[1,1]) / (cm[1,1] + cm[0,1] + prior[1,1] + prior[0,1])
    recall = (cm[1,1] + prior[1,1]) / (cm[1,1] + cm[1,0] + prior[1,1] + prior[1,0])
    return(2*precision*recall / (precision + recall))


def d_prime(cm, prior=np.array([[1,1], [1,1]])):
    hit_rate = (cm[1,1] + prior[1,1]) / (cm[1,1] + cm[1,0] + prior[1,1] + prior[1,0])
    false_alarm = (cm[0,1] + prior[0,1]) / (cm[0,0] + cm[0,1] + prior[0,0] + prior[0,1])
    return(norm.ppf(hit_rate) - norm.ppf(false_alarm)) 

def performance_curve(run_name, pct=True, directory=SAVED_DIR, empty=-100.0):
    '''Returns a dictionary of the number or percentage of responses to each input.
    '''
    outputs = binarize_outputs(run_name, directory=directory)
    outputs = (outputs == 'NP')
    inputs = all_inputs(run_name, directory=directory)
    count_dict = {i: outputs[inputs == i].sum() for i in set(inputs) if i != empty}
    total_dict = {i: (inputs == i).sum() for i in set(inputs) if i != empty}
    if pct:
        count_dict = {i: outputs[inputs == i].sum() / (inputs == i).sum() for i in set(inputs) if i != empty}
    else:
        count_dict = {i: outputs[inputs == i].sum() for i in set(inputs)if i != empty}
    #return count_dict
    return(count_dict, total_dict)

def calculate_performance(run_name, directory=SAVED_DIR, auto_save=True):
    print("starting performance analysis")
    inputs = binarize_inputs(run_name, directory=directory)
    outputs = binarize_outputs(run_name, directory=directory)
    cm = confusion_matrix(inputs == 'T', outputs =='NP')
    pc, _ = performance_curve(run_name, pct=True, directory=directory)
    pct_correct = pc[2.0]
    dp = d_prime(cm)
    F1 = F1_score(cm)
    results = {"pct_correct": pct_correct, "d_prime":dp, "F1": F1, "hits": cm[1,1], "misses":cm[1,0], "fAlarms":cm[0,1], "cRejects":cm[0,0]}
    print(results)
    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_performance.json'), 'w') as f:
            json.dump(results, f)
    return(results)