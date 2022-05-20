import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from ..io import SAVED_DIR, load_rates, load_current, load_weights, save_convergence
from ..math import ones

def test_convergence(run_name, directory=SAVED_DIR, conv_trials=100, conv_alpha=0.05, conv_thresh=0.01, conv_method="slope_scale", auto_save=True):
    '''
    Determines whether a run converged for firing rate, weights, and current.
    conv_trials refers to the number of trials to use for the regression, ending at
    the final trial of the simulation.

    Methods return either True/False or a numerical value. They include:
        1. "slope_ci".......Tests whether 0 is included in the CI for the slope (set 
                            by `conv_alpha`) and has a magnitude less than `thresh` * avg_value
        2. "aic"............Fits a linear model and constant model using OLS and uses
                            the AIC criterion to choose which is a better description
        3. "slope_scale"....Simply returns the value of the slope to use as a continuous
                            scale
    
    Returns a dict with relevant parameters and test values. 

    If `auto_save` is enabled stores the dict as a JSON in the run directory. 
    '''
    assert conv_method in ["slope_ci", "slope_scale", "aic", "rmse"]

    variables = ["rate_E_in", "rate_E_out", "rate_I", "bias_current", "W_EE_avg", "W_IE_avg", "W_o_avg"]
    # Loading data
    ## Load population firing rates
    rates = load_rates(run_name, directory=directory)
    rate_E_in = np.array(rates.iloc[-conv_trials:,1].values,dtype="float")
    #ex_non = np.array(rates.iloc[-conv_trials:,2].values,dtype="float")
    rate_E_out = np.array(rates.iloc[-conv_trials:,3].values,dtype="float")
    rate_I = np.array(rates.iloc[-conv_trials:,4].values,dtype="float")
    
    ## Load current
    current = load_current(run_name,directory=directory)
    bias_current = np.array(current.iloc[-conv_trials:,1].values,dtype="float")
    
    ## Load weights
    weights = load_weights(run_name, directory=directory)
    W_EE_avg = np.array(weights.iloc[-conv_trials:,1].values,dtype="float")
    W_IE_avg = np.array(weights.iloc[-conv_trials:,2].values,dtype="float")
    W_o_avg = np.array(weights.iloc[-conv_trials:,3].values,dtype="float")
    
    ## Create X-axis for linear regressions
    trials = np.linspace(1, conv_trials, conv_trials)
    lin_X = sm.add_constant(trials)
    const_X = ones(len(trials))

    ## Create X-axis for linear regressions
    conv_trials_array = np.linspace(1, conv_trials, conv_trials)
    lin_X = sm.add_constant(conv_trials_array)
    const_X = ones(len(conv_trials_array))
    # Adding method and paramaters to output
    convergence = {'conv_method': conv_method, "conv_trials": conv_trials}
    if conv_method == 'slope_ci':
        convergence["conv_alpha"] = conv_alpha
        convergence["conv_thresh"] = conv_thresh
    elif conv_method == 'rmse':
        convergence["conv_thresh"] = conv_thresh
    ## Calculate Regression models and determine convergence
    for var in variables:
        if conv_method == 'slope_ci':
            vals = eval(var)
            results = sm.OLS(vals, lin_X).fit()
            ci = results.conf_int(conv_alpha)
            slope = results.params[1]
            slope_thresh = np.mean(vals) * conv_conv
            convergence["conv_" + var] = bool(ci[1,0]<0 and ci[1,1]>0 and np.abs(slope)<slope_thresh)
        elif conv_method == 'slope_scale':
            vals = eval(var)
            results = sm.OLS(vals, lin_X).fit()
            slope = results.params[1]
            convergence["conv_" + var] = slope
        elif conv_method == 'aic':
            #df = {var: eval(var), "t": conv_trials}
            #lin_aic = smf.ols(f"{var} ~ t + 1", data=df).fit().aic
            #const_aic = smf.ols(f"{var} ~ 1", data=df).fit().aic
            lin_aic = sm.OLS(eval(var), lin_X).fit().aic
            const_aic = sm.OLS(eval(var), const_X).fit().aic
            if const_aic <= lin_aic:
                convergence["conv_" + var] = True
            else:
                convergence["conv_" + var] = False
        elif conv_method == 'rmse':
            vals = eval(var)
            rmse = np.sqrt(np.mean((vals-mean(vals))**2))
            if rmse <= conv_conv:
                convergence["conv_" + var] = True
            else:
                convergence["conv_" + var] = False
    if (conv_method in ["slope_ci", "aic", "rmse"]):
        if all_converged(convergence,variables):
            convergence["converged"] = True
        else:
            convergence["converged"] = False
    if auto_save:
        save_convergence(run_name,convergence,directory=directory,file_name="analysis_convergence.json")
    return convergence

def all_converged(convergence,variables):
    for var in variables:
        if not convergence["conv_" + var]:
            return False
    return True
