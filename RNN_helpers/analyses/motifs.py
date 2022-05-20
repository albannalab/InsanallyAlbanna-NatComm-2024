# Motif code

import numpy as np
import pandas as pd
from .weights import import_W_signed
from numpy.linalg import matrix_power as mp
from copy import deepcopy


def motif_mats_basic(W):
    N = W.shape[0]
    u = np.ones(N) / np.sqrt(N)
    ui = np.ones(N) / N
    H = np.outer(u,u)
    Th = np.identity(N) - H
    return N, u, H, Th, ui

def u(N, idx=None, norm=True):
    u = np.ones(N)
    if idx is None:
        if norm == True:
            u = u / np.sqrt(N)
        else:
            u = u / N
    else:
        u[list(set(range(N)) - set(idx))] = 0.0
        if norm == True:
            u = u / np.sqrt(len(list(idx)))
        else:
            u = u / len(list(idx))
    return u

def sgn_root(M, n):
    sgn = np.sign(M)
    val = np.abs(M)
    return(sgn * val**(1./n))

def motif_mats_multiple(W, pops):
    N = W.shape[0]
    U = np.block([[u(N, pop)] for pop in pops]).T
    Ui = np.block([[u(N, pop, norm=False)] for pop in pops]).T
    H = U @ U.T
    Th = np.identity(N) - H
    return N, U, H, Th, Ui

def ch_mat(W, n, pops=None, rescale=False):
    if pops is None:
        N, u, H, Th, _ = motif_mats_basic(W)
    else:
        N, U, H, Th, _ = motif_mats_multiple(W, pops)
    if n == 0:
        return np.identity(N)
    else:
        mat = (mp(W @ Th, n-1) @ W) 
        if not rescale:
            return(mat / N**(n-1))
        else:
            #return(sgn_root(mat, n) / N)
            return(mat)

def di_mat(W, m, n, pops=None, rescale=False):
    if pops is None:
        N, u, H, Th, _ = motif_mats_basic(W)
    else:
        N, U, H, Th, _ = motif_mats_multiple(W, pops)
    if m == 0:
        return ch_mat(W, n, pops=pops, rescale=rescale).T
    elif n == 0: 
        return ch_mat(W, m, pops=pops, rescale=rescale)
    else:
        mat = (mp(W @ Th, m-1) @ W @ Th @ (mp(W @ Th, n-1) @ W).T)
        if not rescale:
            return(mat / N**(m+n-1))
        else:
            #return(sgn_root(mat, m+n) / N)
            return(mat)

def co_mat(W, m, n, pops=None, rescale=False):
    if pops is None:
        N, u, H, Th, _ = motif_mats_basic(W)
    else:
        N, U, H, Th, _ = motif_mats_multiple(W, pops)
    if m == 0:
        return ch_mat(W, n, pops=pops, rescale=rescale)
    elif n == 0: 
        return ch_mat(W, m, pops=pops, rescale=rescale).T
    else:
        mat = ((mp(W @ Th, m-1) @ W).T @ Th @ mp(W @ Th, n-1) @ W)
        if not rescale:
            return(mat / N**(m+n-1))
        else:
            #return(sgn_root(mat, m+n) / N)  
            return(mat)  

def ch_moment(W, n):
    N = W.shape[0]
    return np.sum(mp(W, n)) / N**(n+1)

def di_moment(W, m, n):
    N = W.shape[0]
    if m==0:
        return ch_moment(W, n)
    elif n==0:
        return ch_moment(W, m)
    else:
        return np.sum(mp(W.T, m) @ mp(W, n)) / N**(m+n+1)

def co_moment(W, m, n):
    N = W.shape[0]
    if m==0:
        return ch_moment(W, n)
    elif n==0:
        return ch_moment(W, m)
    else:
        return np.sum(mp(W, m) @ mp(W.T, n)) / N**(m+n+1)

def calc_cumulants(net, method='net', n_pops=4, max_order=3, g=1, rescale=True, signed=False):
    assert method in ['net', 'indiv']

    if signed:
        W = g*import_W_signed(net)
    else:
        W = g*deepcopy(net.n.W) 
    
    N = net.p.N
    N_in = net.p.N_in
    N_out = net.p.N_out
    NI = net.p.NI

    if n_pops <= 1:
        pops = None
        names = ['all']
    elif n_pops == 2:
        pop_E = np.array(list(net.pop.E)) - 1
        pop_I = np.array(list(net.pop.I)) - 1
        pops = [pop_E, pop_I]
        names = ['E', 'I']
    elif n_pops == 3:
        pop_in = np.array(list(net.pop.InE)) - 1
        pop_out = np.array(list(net.pop.OutE)) - 1
        pop_I = np.array(list(net.pop.I)) - 1
        pops = [pop_in, pop_out, pop_I]
        names = ['in', 'out', 'I']
    elif n_pops == 4:
        pop_inT = np.array(list(net.pop.InT)) - 1
        pop_inF = np.array(list(net.pop.InE - net.pop.InT)) - 1
        pop_out = np.array(list(net.pop.OutE)) - 1
        pop_I = np.array(list(net.pop.I)) - 1
        pops = [pop_inT, pop_inF, pop_out, pop_I]
        names = ['inT', 'inF', 'out', 'I']

    if pops is None:
        N, U, H, Th, Ui = motif_mats_basic(W)
        if method == 'net':
            ch_size = (max_order+1,)
            di_co_size = (max_order+1, max_order+1)
            tr_2d_size = (max_order+1, max_order+1)
            tr_4d_size = (max_order+1, max_order+1, max_order+1, max_order+1)
        elif method == 'indiv':
            ch_size = (max_order+1, N)
            di_co_size = (max_order+1, max_order+1, N)
            tr_1d_size = (max_order+1, N)
            tr_2d_size = (max_order+1, max_order+1, N)
            tr_4d_size = (max_order+1, max_order+1, max_order+1, max_order+1, N)
    else:
        N, U, H, Th, Ui = motif_mats_multiple(W, pops)
        if method == 'net':
            ch_size = (max_order+1, len(pops), len(pops))
            di_co_size = (max_order+1, max_order+1, len(pops), len(pops))
            tr_2d_size = (max_order+1, max_order+1)
            tr_4d_size = (max_order+1, max_order+1, max_order+1, max_order+1)
        elif method == 'indiv':
            ch_size = (max_order+1, N, len(pops))
            di_co_size = (max_order+1, max_order+1, N, len(pops))
            tr_1d_size = (max_order+1, N)
            tr_2d_size = (max_order+1, max_order+1, N)
            tr_4d_size = (max_order+1, max_order+1, max_order+1, max_order+1, N)
            in_pop_size = (N, n_pops)
    
    if method=='net':
        cumulants = {'ch': np.nan*np.empty(ch_size), 
                    'di': np.nan*np.empty(di_co_size), 
                    'co': np.nan*np.empty(di_co_size), 
                    'tr2': np.nan*np.empty(tr_2d_size),
                    'tr4': np.nan*np.empty(tr_4d_size),
                     } 
    elif method == 'indiv':
        cumulants = {'ch': np.nan*np.empty(ch_size), 
                     'chR': np.nan*np.empty(ch_size), 
                    'di': np.nan*np.empty(di_co_size), 
                    'co': np.nan*np.empty(di_co_size), 
                    'tr1_ch': np.nan*np.empty(tr_1d_size),
                    'tr2_di': np.nan*np.empty(tr_2d_size),
                    'tr2_co': np.nan*np.empty(tr_2d_size),
                    'tr4_di': np.nan*np.empty(tr_4d_size),
                    'tr4_co': np.nan*np.empty(tr_4d_size),
                    'in_pop': np.zeros(in_pop_size)
                     } 

    # marking populations
    if method == 'indiv':
        for i in range(N):
            for j, pop in enumerate(pops):
                cumulants['in_pop'][i,j] = int(i in pop)

    # chain, divergent, and convergent calculations
    for n in range(1, max_order+1):
        ch_m = ch_mat(W, n, pops, rescale)
        if method == 'net':
            cumulants['ch'][n] = Ui.T @ ch_m @ Ui
        if method == 'indiv':
            cumulants['ch'][n] = ch_m @ Ui
            cumulants['chR'][n] = ch_m.T @ Ui
        for m in range(1, max_order - n + 1):
            di_m = di_mat(W, n, m, pops, rescale)
            co_m = co_mat(W, n, m, pops, rescale)
            if method == 'net':
                cumulants['co'][n, m] = Ui.T @ co_m @ Ui
                cumulants['di'][n, m] = Ui.T @ di_m @ Ui
            if method == 'indiv':
                cumulants['co'][n, m] = co_m @ Ui
                cumulants['di'][n, m] = di_m @ Ui

        ## trace cumulant calculations
        if method == 'net':
            for n in range(0, max_order+1):
                for m in range(0, max_order-n+1):
                    if n == 0 and m == 0:
                        continue
                    tr_m1 = co_mat(W, n, m, pops, rescale)
                    if rescale:
                        cumulants['tr2'][n, m] = np.trace(tr_m1 @ Th)
                    else:
                        cumulants['tr2'][n, m] = np.trace(tr_m1 @ Th) / N 
                    for l in range(0, max_order-n-m+1):
                        for k in range(0, max_order-n-m-l+1):
                            tr_m2 = co_mat(W, l, k, pops, rescale)
                            if rescale:
                                cumulants['tr4'][n, m, l, k] = np.trace(tr_m1 @ Th @ tr_m2 @ Th)
                            else:
                                cumulants['tr4'][n, m, l, k] = np.trace(tr_m1 @ Th @ tr_m2 @ Th) / N 
        elif method == 'indiv':
            for n in range(1, max_order+1):
                ch_m = ch_mat(W, n, pops, rescale)
                if rescale:
                    cumulants['tr1_ch'][n] = np.diag(ch_m @ Th)
                else:
                    cumulants['tr1_ch'][n] = np.diag(ch_m @ Th) / N
                for m in range(1, max_order-n+1):
                    di_m1 = di_mat(W, n, m, pops, rescale)
                    co_m1 = co_mat(W, n, m, pops, rescale)
                    if rescale:
                        cumulants['tr2_di'][n, m] = np.diag(di_m @ Th)
                        cumulants['tr2_co'][n, m] = np.diag(co_m @ Th)
                    else:
                        cumulants['tr2_di'][n, m] = np.diag(di_m @ Th) / N
                        cumulants['tr2_co'][n, m] = np.diag(co_m @ Th) / N
                    for l in range(0, max_order-n-m+1):
                        for k in range(0, max_order-n-m-l+1):
                            if (l == 0) and (k == 0):
                                continue
                            di_m2 = di_mat(W, l, k, pops, rescale)
                            co_m2 = co_mat(W, l, k, pops, rescale)
                            if rescale:
                                cumulants['tr4_di'][n, m, l, k] = np.diag(di_m1 @ Th @ di_m2 @ Th)
                                cumulants['tr4_co'][n, m, l, k] = np.diag(co_m1 @ Th @ co_m2 @ Th)
                            else:
                                cumulants['tr4_di'][n, m, l, k] = np.diag(di_m1 @ Th @ di_m2 @ Th) / N 
                                cumulants['tr4_co'][n, m, l, k] = np.diag(co_m1 @ Th @ co_m2 @ Th) / N
    return cumulants, names

def cumulants_to_DataFrame(cumulants, pop_names):
    sig = cumulants['ch'].shape
    max_order = sig[0] - 1
    if len(sig) == 1:
        method = 'net'
        n_pops = 1
        N = 1
    elif len(sig) == 2:
        method = 'indiv'
        n_pops = 1
        N = sig[1]
    elif len(sig) == 3:
        n_pops = sig[2] 
        if sig[1] == n_pops:
            method = 'net'
            N = 1
        else:
            method = 'indiv'
            N = sig[1]
    assert len(pop_names) == n_pops

    pre_df = []

    for i in range(max(N, n_pops)):
        default_row = {}
        if method == 'indiv':
            default_row['to'] = i + 1
        elif method == 'net':
            default_row['to'] = pop_names[i]
        for j in range(n_pops):
            default_row['from'] = pop_names[j]
            current_row = deepcopy(default_row)
            for n in range(1, max_order + 1):
                if n_pops == 1:
                    if method == 'net':
                        current_row[f'ch_{n}'] = cumulants['ch'][n]
                        current_row[f'tr1_ch_{n}'] = cumulants['tr2'][n, 0]
                    elif method == 'indiv':
                        current_row[f'ch_{n}'] = cumulants['ch'][n, i]
                        current_row[f'chR_{n}'] = cumulants['chR'][n, i]
                        current_row[f'tr1_ch_{n}'] = cumulants['tr1_ch'][n, i]
                else:
                    if method == 'net':
                        current_row[f'ch_{n}'] = cumulants['ch'][n, i, j]
                        current_row[f'tr1_ch_{n}'] = cumulants['tr2'][n, 0]
                    elif method == 'indiv':
                        current_row[f'ch_{n}'] = cumulants['ch'][n, i, j]
                        current_row[f'chR_{n}'] = cumulants['chR'][n, i, j]
                        current_row[f'tr1_ch_{n}'] = cumulants['tr1_ch'][n, i]
                for m in range(1, max_order - n + 1):
                    if n_pops == 1:
                        if method == 'net':
                            current_row[f'co_{n}{m}'] = cumulants['co'][n, m]
                            current_row[f'di_{n}{m}'] = cumulants['di'][n, m]
                            current_row[f'tr2_{n}{m}'] = cumulants['tr2'][n, m]
                        elif method == 'indiv':
                            current_row[f'co_{n}{m}'] = cumulants['co'][n, m, i]
                            current_row[f'tr2_co_{n}{m}'] = cumulants['tr2_co'][n, m, i]
                            current_row[f'di_{n}{m}'] = cumulants['di'][n, m, i]
                            current_row[f'tr2_di_{n}{m}'] = cumulants['tr2_di'][n, m, i]
                    else:
                        if method == 'net':
                            current_row[f'co_{n}{m}'] = cumulants['co'][n, m, i, j]
                            current_row[f'di_{n}{m}'] = cumulants['di'][n, m, i, j]
                            current_row[f'tr2_{n}{m}'] = cumulants['tr2'][n, m]
                        elif method == 'indiv':
                            current_row[f'co_{n}{m}'] = cumulants['co'][n, m, i, j]
                            current_row[f'tr2_co_{n}{m}'] = cumulants['tr2_co'][n, m, i]
                            current_row[f'di_{n}{m}'] = cumulants['di'][n, m, i, j]
                            current_row[f'tr2_di_{n}{m}'] = cumulants['tr2_di'][n, m, i]
                    for k in range(0, max_order-n-m+1):
                        for l in range(0, max_order-n-m+1):
                            if method == 'net':
                                current_row[f'tr4_{n}{m}{k}{l}'] = cumulants['tr4'][n, m, k, l]
                                current_row[f'tr4_{n}{m}{k}{l}'] = cumulants['tr4'][n, m, k, l]
                            elif method == 'indiv':
                                current_row[f'tr4_co_{n}{m}{k}{l}'] = cumulants['tr4_co'][n, m, k, l, i]
                                current_row[f'tr4_di_{n}{m}{k}{l}'] = cumulants['tr4_di'][n, m, k, l, i]
            pre_df.append(current_row)
    return pd.DataFrame(pre_df)

def find_order(name):
    try: 
        o = np.sum([int(n) for n in name.split('_')[-1]])
    except:
        o = 0
    return o

def find_type(name):
    o =  "_".join(name.split('_')[:-1])
    return o

def calc_motif_contributions(df_out, model, prefix="", eta="sum", max_order=3):
    # calculating contributions
    df_new = df_out.copy()
    
    for i in range(1, max_order+1):
        df_new[f"{prefix}con_net_{i}"] = 0.0
        df_new[f"{prefix}con_net"] = 0.0
    
    for param in dict(model.params).items():
        name, weight = param
        if (':' in name):
            continue

        key = f"{prefix}con_{name}"
        if eta is True:
            contribution = df_new[name] * model.params["eta:" + name]
        elif eta is False:
            contribution = df_new[name] * model.params[name]
        elif eta == "sum":
            contribution = df_new[name] * model.params[name] + df_new[name] * model.params["eta:" + name] * df_new["eta"]
        df_new[key] = contribution
        
        df_new[f"{prefix}con_net_{find_order(name)}"] += contribution
        df_new[f"{prefix}con_net"] += contribution 
    
    return df_new


