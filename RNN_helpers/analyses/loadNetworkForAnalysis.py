from RNN_helpers.io import load_outputs, load_network_all
from RNN_helpers.analysis import binarize_inputs, binarize_outputs
import numpy as np
import math
import os
import json
from types import SimpleNamespace
import sys
import h5py

def loadNetwork(DIR, RUN, OutputOnly=True):
    print("loading "+RUN)
    # Loading the Network
    meta, params, network = load_network_all(RUN, directory=DIR)
    m = SimpleNamespace()
    p = SimpleNamespace()
    n = SimpleNamespace()
    vars(m).update(meta)
    vars(p).update(params)
    vars(n).update(network)

    #Get changes to weight matrix across training epochs
    n.W = np.transpose(n.W)
    try:
        with h5py.File(os.path.join(DIR, RUN, "network_start.h5"), 'r') as network_file:
            net_vars = {n_var: np.copy(network_file[n_var]) for n_var in network_file.keys()}
        n.startW = np.transpose(net_vars["W"])
    except:
        print("loadNetwork: No network_start.h5")
    try:
        with h5py.File(os.path.join(DIR, RUN, "network_STDP_start.h5"), 'r') as network_file:
            net_vars = {n_var: np.copy(network_file[n_var]) for n_var in network_file.keys()}
        n.stdpstartW = np.transpose(net_vars["W"])
    except:
        print("loadNetwork: No network_STDP_start.h5")
    try:
        with h5py.File(os.path.join(DIR, RUN, "network_STDP_end.h5"), 'r') as network_file:
            net_vars = {n_var: np.copy(network_file[n_var]) for n_var in network_file.keys()}
        n.stdpendW = np.transpose(net_vars["W"])
    except:
        print("loadNetwork: No network_STDP_end.h5")
    try:
        with h5py.File(os.path.join(DIR, RUN, "network_FORCE_start.h5"), 'r') as network_file:
            net_vars = {n_var: np.copy(network_file[n_var]) for n_var in network_file.keys()}
        n.forcestartW = np.transpose(net_vars["W"])
    except:
        print("loadNetwork: No network_FORCE_start.h5")
    try:
        with h5py.File(os.path.join(DIR, RUN, "network_FORCE_end.h5"), 'r') as network_file:
            net_vars = {n_var: np.copy(network_file[n_var]) for n_var in network_file.keys()}
        n.forceendW = np.transpose(net_vars["W"])
    except:
        print("loadNetwork: No network_FORCE_end.h5")

    genSize = p.N
    offset = 0
    if OutputOnly:
        genSize = p.N_out
        offset = p.N_in

    # Loading the network statistics
    pop = SimpleNamespace()
    pop.all = set(range(1,p.N))
    pop.E = set(range(1,p.NE))
    pop.I = set(range(1+p.NE, 1+p.N))
    pop.InE = set(range(1,p.N_in))
    pop.InT = set(range(87,115))#TODO make this scale better
    pop.OutE = set(range(1+p.NE-p.N_out, 1+p.NE))

    bools = SimpleNamespace()
    bools.OutE = [0]*p.N
    bools.InE = [0]*p.N
    bools.I = [0]*p.N
    for i in pop.all:
        if i in pop.OutE:
            bools.OutE[i-1] = 1
            bools.InE[i-1] = 0
            bools.I[i-1] = 0
        elif i in pop.InE:
            bools.OutE[i-1] = 0
            bools.InE[i-1] = 1
            if i in pop.InT:
                bools.InE[i-1] = 2
            bools.I[i-1] = 0
        elif i in pop.I:
            bools.OutE[i-1] = 0
            bools.InE[i-1] = 0
            bools.I[i-1] = 1
        else:
            bools.OutE[i-1] = 0
            bools.InE[i-1] = 0
            bools.I[i-1] = 0
    
    # Loading inputs & outputs
    task = SimpleNamespace()
    try: 
        task.all = load_outputs(RUN, DIR)
        task.inputs = binarize_inputs(RUN, DIR)
        task.outputs = binarize_outputs(RUN, DIR)
    except:
        print("loadNetwork: Could not load inputs/outputs")

    # Loading Response Profiles
    try:
        with open(os.path.join('.', DIR, RUN, 'analysis_sets.json'), 'r') as f:
            sets = json.load(f)
        pop.NNR = set(sets['non'])
        pop.R = set(range(1, p.N+1)) - set(sets['non'])
        hasSets = True
    except:
        print("loadNetwork: No analysis_sets.json")
        hasSets = False

    if hasSets:
        bools.NNR = [0]*genSize
        bools.R = [0]*genSize
        for i in range(1,genSize+1):
            if (i+offset) in pop.NNR:
                bools.NNR[i-1] = 1
                bools.R[i-1] = 0
            elif (i+offset) in pop.R:
                bools.NNR[i-1] = 0
                bools.R[i-1] = 1
            else:
                bools.NNR[i-1] = 0
                bools.R[i-1] = 0

    # Loading Responsiveness
    try:
        with open(os.path.join('.', DIR, RUN, 'analysis_cells.json'), 'r') as f:
            cells = json.load(f)
        hasCells = True
    except:
        print("loadNetwork: no analysis_cells.json")
        hasCells = False

    # Loading Informativity
    try:
        with open(os.path.join('.', DIR, RUN, 'analysis_informativity.json'), 'r') as f:
            info = json.load(f)
        hasInfo = True
    except:
        print("loadNetwork: no analysis_informativity.json")
        hasInfo = False

    #Loading Balance
    try:
        with open(os.path.join('.', DIR, RUN, 'analysis_balance.json'), 'r') as f:
            balance = json.load(f)
        ECurr = np.array(balance["I_Exc"])
        ICurr = np.array(balance["I_Inh"])
        imbalance = ECurr - ICurr
        if OutputOnly:
            ECurr = ECurr[p.N_in:p.NE]
            ICurr = ICurr[p.N_in:p.NE]
            imbalance = imbalance[p.N_in:p.NE]
        hasBalance = True
    except:
        print("loadNetwork: No analysis_balance.json")
        hasBalance = False
    #Target Input units
    #input = range(87,115)

    #generate data
    FR = [None]*genSize
    choice = [None]*genSize
    choiceT = [None]*genSize
    choiceF = [None]*genSize
    choiceTF = [None]*genSize
    choiceNP = [None]*genSize
    choiceW = [None]*genSize
    choiceNPW = [None]*genSize
    stimulus = [None]*genSize
    stimulusT = [None]*genSize
    stimulusF = [None]*genSize
    stimulusNP = [None]*genSize
    stimulusW = [None]*genSize
    stimulusTF = [None]*genSize
    stimulusNPW = [None]*genSize
    responsiveness = [None]*genSize
    cvar = [None]*genSize
    svar = [None]*genSize
    wout = [0]*genSize
    eta = [0]*genSize
    if hasInfo:
        informativity = [None]*genSize

    for i in range(1,genSize+1):
        if hasCells:
            FR[i-1] = cells[str(i+offset)]['fr_mean']
            stimulus[i-1] = cells[str(i+offset)]['s_mean']
            stimulusT[i-1] = cells[str(i+offset)]['sT_mean']
            stimulusF[i-1] = cells[str(i+offset)]['sF_mean']
            stimulusNP[i-1] = cells[str(i+offset)]['sNP_mean']
            stimulusW[i-1] = cells[str(i+offset)]['sW_mean']
            stimulusTF[i-1] = stimulusT[i-1]-stimulusF[i-1]
            stimulusNPW[i-1] = stimulusNP[i-1]-stimulusW[i-1]
            choice[i-1] = cells[str(i+offset)]['c_mean']
            choiceNP[i-1] = cells[str(i+offset)]['cNP_mean']
            choiceW[i-1] = cells[str(i+offset)]['cW_mean']
            choiceT[i-1] = cells[str(i+offset)]['cT_mean']
            choiceF[i-1] = cells[str(i+offset)]['cF_mean']
            choiceTF[i-1] = choiceT[i-1]-choiceF[i-1]
            choiceNPW[i-1] = choiceNP[i-1]-choiceW[i-1]
            responsiveness[i-1] = cells[str(i+offset)]['responsiveness']
            cvar[i-1] = cells[str(i+offset)]['c_std']
            svar[i-1] = cells[str(i+offset)]['s_std']
        if hasInfo:
            try:
                informativity[i-1] = info[str(i+offset)]
            except KeyError as e:
                informativity[i-1] = np.nan
        if OutputOnly:
            wout[i-1] = n.W_out[i-1]
            eta[i-1] = n.eta[i-1]
        elif (i+offset) in pop.OutE:
            wout[i-1] = n.W_out[i-p.N_in-1]
            eta[i-1] = n.eta[i-p.N_in-1]

    #Effective Stimulation
    T = np.equal(bools.InE,2)
    inputToNetworkT = np.sum(np.array(n.W[:,T]),axis=1)
    inputToInhibT = np.sum(np.array(n.W[p.NE:p.N,T]),axis=1)
    InhibitoryToAllT = np.array(n.W[:, p.NE:p.N])
    EffInhT = np.matmul(inputToInhibT,InhibitoryToAllT.T)
    EffStimT = inputToNetworkT - EffInhT

    F = np.equal(bools.InE,1)
    inputToNetworkF = np.sum(np.array(n.W[:,F]),axis=1)
    inputToInhibF = np.sum(np.array(n.W[p.NE:p.N,F]),axis=1)
    InhibitoryToAllF = np.array(n.W[:, p.NE:p.N])
    EffInhF = np.matmul(inputToInhibF,InhibitoryToAllF.T)
    EffStimF = inputToNetworkF - EffInhF

    EffStimTF = EffStimT - EffStimF
    inputToNetworkTF = inputToNetworkT - inputToNetworkF
    if OutputOnly:
        EffStimTF = EffStimTF[np.equal(bools.OutE,1)]
        EffStimT = EffStimT[np.equal(bools.OutE,1)]
        EffStimF = EffStimF[np.equal(bools.OutE,1)]
        inputToNetworkTF = inputToNetworkTF[np.equal(bools.OutE,1)]
        inputToNetworkT = inputToNetworkT[np.equal(bools.OutE,1)]
        inputToNetworkF = inputToNetworkF[np.equal(bools.OutE,1)]


    ##Effective Inhibition
    #OG = stimulus
    #OutputToInhibitory = np.array(n.W[p.NE:p.N , p.N_in:p.NE])
    #InhibitoryToOutput = np.array(n.W[p.N_in:p.NE , p.NE:p.N])
    #EffInh = np.matmul(np.vstack(OG).T,OutputToInhibitory.T)
    #EffInh = np.matmul(EffInh,InhibitoryToOutput.T)
    ##EffInh = np.matmul(np.vstack(OG).T,InhibitoryToOutput)
    ##EffInh = np.matmul(EffInh,OutputToInhibitory)
    #EffInh = OG - EffInh

    ##Generate stimTF ortho choice
    #v1 = stimulusTF
    #v2 = choice
    #cosinesim = np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    #parallel = v2/np.linalg.norm(v2) * np.linalg.norm(v1) * cosinesim
    ##parallel = np.array(v2)*float(np.dot(v1, v2)/np.linalg.norm(v2)**2)
    #ortho = v1 - parallel

    result = SimpleNamespace()
    result.m = m
    result.n = n
    result.p = p
    result.pop = pop
    result.bools = bools
    result.task = task
    if hasCells:
        result.FR = FR
        result.stimulus = stimulus
        result.stimulusT = stimulusT
        result.stimulusF = stimulusF
        result.stimulusTF = stimulusTF
        result.stimulusNP = stimulusNP
        result.stimulusW = stimulusW
        result.stimulusNPW = stimulusNPW
        result.choice = choice
        result.choiceT = choiceT
        result.choiceF = choiceF
        result.choiceTF = choiceTF
        result.choiceNP = choiceNP
        result.choiceW = choiceW
        result.choiceNPW = choiceNPW
        result.responsiveness = responsiveness
        result.cvar = cvar
        result.svar = svar
    result.wout = wout
    result.eta = eta
    if hasBalance:
        result.ECurr = ECurr
        result.ICurr = ICurr
        result.imbalance = imbalance
    result.EffStimTF = EffStimTF
    result.EffStimT = EffStimT
    result.EffStimF = EffStimF
    result.inputToNetworkT = inputToNetworkT
    result.inputToNetworkF = inputToNetworkF
    result.inputToNetworkTF = inputToNetworkTF
    if hasInfo:
        result.informativity = informativity
    return result
