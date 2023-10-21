#!julia
using Base
using Pkg

# activate the environment defined by `Manifest.toml` and `Project.toml`
# You should have already run `Pkg.instantiate()` and `Pkg.precompile()`
Pkg.activate(joinpath(@__DIR__))

# Load the code necessary for simulation used below
include("spiking_LIF_w_FORCE_STDP.jl")

# Save directory (note that all runs will be saved in a subdirectory with `EXP_NAME``)
SAVE_DIR = "."
EXP_NAME = "example_simulations"

# Number of training and testing trials
N_train = 5000
N_test = 400

# Initializing the network
p = LIFv2Params();
w = LIFv2Weights(gen_recurrent_weights(p), gen_output_weights(p)..., gen_PW₀(p));

# Training the network
m_train, w_train, activity, freqs, outputs = LIFv2spikingnet(
    N_train, p, w; 
    previous_run=nothing, 
    auto_save=true, 
    show_figs=false, 
    save_figs=true, 
    fig_render_period=100, plot_units=[10, 310, 610, 910], 
    stdp=true, 
    force_learn=true, 
    force_feedback=true, 
    stdp_delay=40, 
    force_delay=100,  
    I_adjust=true, 
    I_adjust_dur=10000, 
    notes="training network", 
    save_dir=joinpath(SAVE_DIR, EXP_NAME));

save_experiment_params(
    (n_rep=n, run_type="train"), 
    m_train.run, 
    EXP_NAME, 
    save_dir=SAVE_DIR);

# Testing the network
run_meta_test, _, activity_test, freqs_test, outputs_test = LIFv2spikingnet(
    N_test, p, w_train;
    previous_run=m_train.run, 
    auto_save=true, 
    show_figs=false, 
    save_figs=true, 
    fig_render_period=100, 
    plot_units=[10, 310, 610, 910], 
    stdp=false, 
    force_learn=false, 
    force_feedback=true, 
    force_delay=0, 
    I_adjust=false, 
    I_bias₀=m_train.I_bias, 
    notes="testing network", 
    save_dir=joinpath(SAVE_DIR, EXP_NAME));

save_experiment_params(
    (n_rep=n, run_type="test"), 
    run_meta_test.run, 
    EXP_NAME, 
    save_dir=SAVE_DIR);

