import numpy as np
import os
import json
from ..io import SAVED_DIR, load_network_all
from types import SimpleNamespace
from copy import deepcopy

def analyze_weights(run_name, directory=SAVED_DIR, auto_save=True):
    print("starting weights analysis")

    try:
        # Loading the Network
        print("1")
        meta, params, network = load_network_all(run_name, directory=directory)
        print("2")
        m = SimpleNamespace()
        p = SimpleNamespace()
        n = SimpleNamespace()
        vars(m).update(meta)
        vars(p).update(params)
        vars(n).update(network)
    except Exception as e:
        print(e)

    print("A")
    # Loading the network statistics
    pop = SimpleNamespace()
    pop.all = set(range(p.N))
    pop.E = set(range(p.NE))
    pop.I = set(range(p.NE, p.N))
    pop.InE = set(range(p.N_in))
    pop.OutE = set(range(p.NE-p.N_out, p.NE))

    print("B")
    # Loading categories
    #with open(os.path.join('.', DIR, RUN, 'analysis_sets.json'), 'r') as f:
    #sets = json.load(f)
    #pop.NR = set(sets['non'])
    #pop.R = set(range(p.N)) - set(sets['non'])

    #Calculating gross properties
    mean_weight = np.mean(n.W.flatten())
    print("C")
    std_weight = np.std(n.W.flatten())
    print("D")

    #Generating classes
    #groups = (pop.InE, pop.OutE, pop.I)
    #group_labels = ['InE', 'OutE', 'I']
    #n_g = len(groups)
    #categories = (pop.NR, pop.R)
    #category_labels = ['NNR', 'R']
    #n_c = len(categories)
    #colors = (NR_color, R_color)
    #crosstab = [g&c for g, c in product(groups, categories)]
    #n_ct = len(crosstab)
    #loc_ct = np.cumsum([len(i) for i in crosstab])
    #cross_labels = [gl + '-' + cl for gl, cl in product(group_labels, category_labels)]

    results = {"mean_weight": mean_weight, "std_weight":std_weight}
    print(results)
    if auto_save:
        with open(os.path.join('.', directory, run_name, 'analysis_weights.json'), 'w') as f:
            json.dump(results, f)
    return(results)

def import_W_signed(net):
    W_signed = deepcopy(net.n.W) 
    I_idx = range(net.p.NE, net.p.N)
    W_signed[:, I_idx] =- W_signed[:, I_idx]
    return W_signed

def import_W_force(net):
    W_force = np.outer(net.n.eta, net.n.W_out)*net.p.Q
    return W_force

def import_W_force_embedded(net):
    W_force_emd = np.zeros((net.p.N, net.p.N))
    W_force = import_W_force(net)
    W_force_emd[net.p.N_in:net.p.N - net.p.NI, net.p.N_in:net.p.N - net.p.NI] = W_force
    return W_force_emd