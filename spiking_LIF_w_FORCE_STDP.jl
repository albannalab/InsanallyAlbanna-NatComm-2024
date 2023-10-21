using LinearAlgebra
using SparseArrays
using ProgressBars      # For progress indicator
using Printf            # For print formatting
using Parameters        # For default keywords
using DataStructures    # For Queue
using Statistics
using Distributions
import Distributions: Normal
import StatsBase: mean, sample, Weights
import PyPlot           # For plotting
import HDF5
const plt = PyPlot;     # Setting "plt" as shortcut to PyPlot

using RNNhelpers;

# Definining default network parameters

@with_kw struct LIFv2Params <: RecurrentNetworkModelParams
    NE::Int = 800 # number of excitatory units
    NI::Int = 200 # number of inhibitory units
    N::Int = NE + NI # total number of units
    Vᵣ::Float64 = -65.0 # Resting voltage (mV)
    V_th::Float64 = -55.0 # Threshold voltage (mV)

    inh_rate_target::Float64 = 20.0 # inhibitory rate target, set by external current (Hz)
    η_rate::Float64 = 0.005 # rate learning rate (Hz)
    τ::Float64 = 20.0 # membrane time constant (ms)
    d::Float64 = 8.0 # syanptic delays (ms)
    τ_epsp::Float64 = 20.0 # EPSP time constant (ms)
    τ_ipsp::Float64 = 20.0 # IPSP time constant (ms)

    # Task parameters
    T::Int = 300 # Length of a trial (ms)
    stim_dur::Int = 100 # Length of stimulus (ms)
    stim_offset::Int = 100 # Offset of stimulus from t=0 ms (ms)
    resp_dur::Int = 100 # Length of response (ms)
    resp_offset::Int = 200 # Offset of response from t=0 ms (ms)
    ITI_min::Int = 0 # Minimum additional time between trials, no STDP or FORCE (ms)
    ITI_max::Int = T # Maximum additional time between trials, no STDP or FORCE (ms)

    # Task inputs mechanisms
    N_in::Int = 200 # Number of input neurons
    freqs::Array{Float64, 1} = map(log2, [0.5, 1, 2, 4, 8, 16, 32])
    target_freq::Float64 = log2(4.0)
    p_target::Float64 = 0.5
    ν_bias::Float64 = 2.0
    ν_scale::Float64 = 2.0
    place_code::Bool = true
    output_type::String = "target_only"
    out_bias::Float64 = 0.0

    # FORCE mechamisms
    N_out::Int = 600 # Number of output neurons
    τ_out::Float64 = 100 # Output time constant (ms)
    t_learn_dur::Float64 = 4.0 # Determines the average time between output learning triggers (ms)
    Q::Float64 = 2.0 # Feedback strength (au)
    k₀::Float64 = 0.1 / N_out # Initial output scale (au)
    Pw₀::Float64 = 1.0 # Initial 'P' scale (acts as regularizer for output weight learning)

    # Recurrent weight initializations
    #j = desired w * sqrt(40) for EE, desired w * sqrt(10) for IE
    #jₑ₀::Float64 = 0.30 * (V_th - Vᵣ) * τ / τ_epsp   # initial weight scale from excitatory units (au)
    #jᵢ₀::Float64 = 0.40 * (V_th - Vᵣ) * τ / τ_ipsp  # initial weight scale from inhibitory units (au)
    jₑ₀::Float64 = 0.14 * (V_th - Vᵣ) * τ / τ_epsp   # initial weight scale from excitatory units (au)
    jᵢ₀::Float64 = 0.14 * (V_th - Vᵣ) * τ / τ_ipsp  # initial weight scale from inhibitory units (au)
    p_con::Float64 = 0.05 # connection probability

    # STDP mechamisms

    ## EE
    EE_stdp_type::String = "*" # for additive or "*" for multiplicative
    τ₊::Float64 = 20.0 # time scale of presynaptic trace for EE plasticity (ms)
    τ₋::Float64 = 20.0 # time scale of postsyanptic trace for EE plasticity (ms)
    A::Float64 = 0.00100  # LTP rate
    B::Float64 = 0.00105  # LTD rate
    hetero_stdp_type::String = "rate" # Heterosynaptic plasticity type, "rate" or "target"
    β::Float64 = 0.0005623413251903491 # Heterosynaptic pasticity strength parameter
    δ::Float64 = 0.000632455532033676 / sqrt(p_con*NE)#0.0001 # Transmitter triggered plasticity strength

    ## IE
    IE_stdp_type::String = "*" # or multiplicative
    τᵢ::Float64 = 10.0 # time scale of postsyanptic trace for IE plasticity (ms)
    exc_rate_target::Float64 = 10.0 # rate target for excitatory neurons, set by IE STDP (Hz)
    αᵢ::Float64 = 2 * exc_rate_target * τᵢ / 1000.0 # LTP rate = 2 * rho * tau_STDP
    ηᵢ::Float64 = 0.001 # IE pasticity rate
    βᵢ::Float64 = 0.0005623413251903491 # Heterosynaptic pasticity strength parameter
    δᵢ::Float64 = 0.000316227766016838 / sqrt(p_con*NI)#0.0001 # Transmitter triggered plasticity strength

    w_min::Float64 = 0.0 # minimum synaptic weight
    w_max::Float64 = 5.0 # maximum synaptic weight
end

@with_kw mutable struct LIFv2Metadata <: RecurrentNetworkModelMetadata
    @add_metadata_fields
    n_trials::Integer
    stdp::Bool
    stdp_delay::Integer
    stdp_end::Integer
    force_learn::Bool
    force_feedback::Bool
    force_delay::Integer
    force_end::Integer
    I_adjust::Bool
    I_adjust_dur::Integer
    I_bias::Real
    perturbations::Dict
end

mutable struct LIFv2Weights <: RecurrentNetworkModelWeights
    W::Matrix
    W_out::Array
    eta::Array
    PW::Array
end

LIFv2 = RecurrentNetworkModelTypes(LIFv2Params, LIFv2Weights, LIFv2Metadata)

"""
    eT₊(I::Real, ΔV::Real)::Real

Compute the exponential time ``\exp(t/τ)`` as a function of the background current.

# Examples
```julia-repl
julia> eT₊(20.0, 10.0)
2.0
```
"""
function eT₊(I::Real, ΔV::Real)
    I / (I - ΔV)
end

function Voltage(et::Real, I::Real, V_ref::Real, ΔV::Real)
    I - V_ref - (I - ΔV) * et
end

"""
    spike_tf!(epsp, ipsp, j, W, NE)

Updates the exponential times ``et = \exp(t/τ)`` when neuron j fires using weight
matrix W.
"""
function spike_tf!(epsp::Array, ipsp::Array, j::Integer, W::SparseMatrixCSC, NE::Integer)
    if j <= NE
        for (i, c) in zip(findnz(W[:, j])...)
            epsp[i] += c
        end
    else
        for (i, c) in zip(findnz(W[:, j])...)
            ipsp[i] += c
        end
    end
end

function current_tf!(et::Array, I₁::Array, I₂::Array, ΔV::Real; perturbations::Dict=Dict(), silenced_cells=nothing)
    if haskey(perturbations,"inactivation") && perturbations["inactivation"]["method"]!="onlyoutput"
        cells = collect(range(1,length=length(et)))
        setdiff!(cells, silenced_cells)
        et[cells] .*= (I₁[cells] .- ΔV) ./ (I₂[cells] .- ΔV)
        et[cells] .+= (I₂[cells] - I₁[cells]) ./ (I₂[cells] .- ΔV)
    else
        et .*= (I₁ .- ΔV) ./ (I₂ .- ΔV)
        et .+= (I₂ - I₁) ./ (I₂ .- ΔV)
    end
end

function to_dt(δet::Real, τ::Real)
    τ * log(δet)
end

function advance_time!(var::Array, δet::Real, τ::Real, τ₂::Real)
    var ./= δet .^ (τ / τ₂)
end

# Input and target output functions
function place_code_stimulus(N_in::Integer, i::Real, n_freq::Integer, ν_scale::Real; perturbations::Dict=Dict())
    N_set_0 = Int(round((i - 1.0) * N_in / n_freq)) + 1
    N_set_1 = Int(round(i * N_in / n_freq)) + 1
    N_set = N_set_1 - N_set_0 + 1

    if haskey(perturbations,"inputnoise") && (perturbations["inputnoise"]["noise_type"]=="input_target")
        input_set = collect(N_set_0:N_set_1)
        N_replace = Int(round(N_set * noise_scale))
        input_set[rand(1:N_set, N_replace)] = rand(1:N_in, N_replace)
        return input_set, ν_scale
    else
        return N_set_0:N_set_1, ν_scale
    end
end

function dummy_inputs(N_in, N_dummy, ν_scale::Real)
    N_set_0 = N_in + 1
    N_set_1 = N_in + 1 + N_dummy
    return N_set_0:N_set_1, ν_scale
end


function input_current(t::Real, N_set, amp::Real, stim_offset::Real, stim_dur::Real, N::Integer; perturbations::Dict=Dict())
    I_new = zeros(N)
    if (t >= stim_offset) & (t < stim_offset + stim_dur)
        I_new[N_set] .+= amp

        if haskey(perturbations,"inputnoise") && (perturbations["inputnoise"]["noise_type"]=="amplitude")
            for i=N_set
                I_new[i] += amp * perturbations["inputnoise"]["noise_scale"] * rand(Normal())
            end
        end
    end
    I_new
end

function dummy_current(t::Real, N_set_dummy, amp_dummy::Real, stim_offset::Real, stim_dur::Real, N::Integer)
    I_new = zeros(N)
    if (t >= stim_offset) & (t < stim_offset + stim_dur)
        I_new[N_set_dummy] .+= amp_dummy
    end
    I_new
end

function gen_output_function(target_freq::Real, resp_offset::Real, resp_dur::Real, out_bias::Real=0.0, output_type::String="target_only")
    if output_type == "target_only"
        function f_out_target_only(freq, t, target=target_freq)
            if (freq == target) & (t >= resp_offset) & (t < (resp_offset + resp_dur))
                return sin((t - resp_offset) * pi / resp_dur) + out_bias
            else
                return out_bias
            end
        end
        return f_out_target_only
    elseif output_type == "symmetric"
        function f_out_symmetric(freq, t, target=target_freq)
            if  (t >= resp_offset) & (t < (resp_offset + resp_dur))
                if (freq == target)
                    return sin((t - resp_offset) * pi / resp_dur) + out_bias
                elseif (freq != target)
                    return -sin((t - resp_offset) * pi / resp_dur) + out_bias
                end
            else
                return out_bias
            end
        end
        return f_out_symmetric
    end
end

# STDP model

## EE homosynaptic terms

function STDP_EE_pre_add(w::Float64, A::Float64, z::Float64)
    A * z
end

function STDP_EE_pre_mul(w::Float64, A::Float64, z::Float64)
    A * z * w  
end

function STDP_EE_post_add(w::Float64, z::Float64, B::Float64)
    -(B * z)
end

function STDP_EE_post_mul(w::Float64, z::Float64, B::Float64)
    -(B * z) * w
end

## IE homosynaptic terms

function STDP_IE_pre_add(w::Float64, η::Float64, z::Float64)
    η * z
end

function STDP_IE_pre_mul(w::Float64, η::Float64, z::Float64)
    η * z * w  
end

function STDP_IE_post_add(w::Float64, z::Float64, η::Float64, α::Float64)
    η * (z - α)
end

function STDP_IE_post_mul(w::Float64, z::Float64, η::Float64, α::Float64)
    η * (z - α) * w
end

## Heterosynaptic terms - transmitter

function STDP_hetero_rate(w::Float64, z::Float64, β::Float64, total::Float64, set::Float64)
    -(β * z^3) * w
end

function STDP_hetero_target(w::Float64, z::Float64, β::Float64, total::Float64, set::Float64)
    β * (1. - total/set) * w
end

function STDP!(W::SparseMatrixCSC, j::Integer, NE::Integer, EE_pre::Function, A::Real, EE_post::Function, B::Real, EE_hetero::Function, E_set::Real, β::Real, δ::Real, IE_pre::Function, ηᵢ::Real, IE_post::Function, αᵢ::Real, IE_hetero::Function, I_set::Real, βᵢ::Real, δᵢ::Real, z₊::Array, z₋::Array, zᵢ₊::Array, zᵢ₋::Array, w_min::Real, w_max::Real)
    if j <= NE # Excitatory unit firing
        (pre, pre_vals) = findnz(W[j, :])
        (post, post_vals) = findnz(W[:, j])
        
        I_index = findfirst(>(NE), pre)
        if I_index === nothing
            I_index = 1 
        end
        E_in_tot = sum(pre_vals[1:I_index-1])
        I_in_tot = sum(pre_vals[I_index:end])

        for i in pre
            if i <= NE #EE
                W[j, i] += EE_pre(W[j, i], A, z₊[i]) 
                W[j, i] += EE_hetero(W[j, i], z₋[j], β, E_in_tot, E_set)
            else #IE
                W[j, i] += IE_pre(W[j, i], zᵢ₊[i-NE], ηᵢ)
                W[j, i] += IE_hetero(W[j, i], zᵢ₋[j], βᵢ, I_in_tot, I_set)             
            end
            W[j, i] = clip_range(W[j, i], w_min, w_max)
        end
        for i in post
            if i <= NE #EE
                W[i, j] += EE_post(W[i, j], z₋[i], B) + δ
                W[i, j] = clip_range(W[i, j], w_min, w_max)
            else
                # i refers to inhibitory cell can skip rest. 
                # Relies on findnz returning ordered indices for 1D array.
                break
            end
        end
    else # Inhibitory unit firing
        (post, post_vals) = findnz(W[:, j])
        for i in post
            if i <= NE #IE
                W[i, j] += IE_post(W[i, j], zᵢ₋[i], ηᵢ, αᵢ) + δᵢ
                W[i, j] = clip_range(W[i, j], w_min, w_max)
            else
                # i refers to inhibitory cell can skip rest. 
                # Relies on findnz returning ordered indices for 1D array.
                break
            end
        end
    end
end

function STDP_track!(W::SparseMatrixCSC, j::Integer, NE::Integer, EE_pre::Function, A::Real, EE_post::Function, B::Real, EE_hetero::Function, E_set::Real, β::Real, δ::Real, IE_pre::Function, ηᵢ::Real, IE_post::Function, αᵢ::Real, IE_hetero::Function, I_set::Real, βᵢ::Real, δᵢ::Real, z₊::Array, z₋::Array, zᵢ₊::Array, zᵢ₋::Array, w_min::Real, w_max::Real)
    Δplasticity = zeros(8)
    if j <= NE # Excitatory unit firing
        (pre, pre_vals) = findnz(W[j, :])
        (post, post_vals) = findnz(W[:, j])
        
        I_index = findfirst(>(NE), pre)
        E_in_tot = sum(pre_vals[1:I_index-1])
        I_in_tot = sum(pre_vals[I_index:end])

        for i in pre
            if i <= NE #EE
                EE_A = EE_pre(W[j, i], A, z₊[i]) 
                W[j, i] += EE_A
                Δplasticity[1] += EE_A

                EE_β = EE_hetero(W[j, i], z₋[j], β, E_in_tot, E_set)
                W[j, i] += EE_β
                Δplasticity[3] += EE_β
            else #IE
                IE_η = IE_pre(W[j, i], zᵢ₊[i-NE], ηᵢ)
                W[j, i] += IE_η
                Δplasticity[5] += IE_η

                IE_β = IE_hetero(W[j, i], zᵢ₋[j], βᵢ, I_in_tot, I_set)   
                W[j, i] += IE_β
                Δplasticity[7] += IE_β     
            end
            W[j, i] = clip_range(W[j, i], w_min, w_max)
        end
        for i in post
            if i <= NE #EE
                EE_B = EE_post(W[i, j], z₋[i], B)
                W[i, j] += EE_B + δ
                Δplasticity[2] += EE_B
                Δplasticity[4] += δ
                W[i, j] = clip_range(W[i, j], w_min, w_max)
            else
                # i refers to inhibitory cell can skip rest. 
                # Relies on findnz returning ordered indices for 1D array.
                break
            end
        end
    else # Inhibitory unit firing
        (post, post_vals) = findnz(W[:, j])
        for i in post
            if i <= NE #IE
                IE_α = IE_post(W[i, j], zᵢ₋[i], ηᵢ, αᵢ)
                W[i, j] += IE_α + δᵢ
                Δplasticity[6] += IE_α
                Δplasticity[8] += δᵢ
                W[i, j] = clip_range(W[i, j], w_min, w_max)
            else
                # i refers to inhibitory cell can skip rest. 
                # Relies on findnz returning ordered indices for 1D array.
                break
            end
        end
    end
    Δplasticity
end

# Perturbation protocols

function select_perturbation_set(p,cells,save_dir,perturbations::Dict)
    sorted_cells = [cell[1] for cell in cells[Symbol(perturbations["inactivation"]["type"])] ]

    #Remove input population
    input_cells = collect(range(1,length=p.N_in))
    filter!(x -> !(x in input_cells), sorted_cells)
    #Remove inhibitory population
    inhibitory_cells = collect(range(1+p.NE,length=p.NI))
    filter!(x -> !(x in inhibitory_cells), sorted_cells)

    silenced_cells = sorted_cells[range(1,length=perturbations["inactivation"]["number"])]
    return silenced_cells
end

# Functions to find next event
function find_next_spike(a::Array)
    where_pos = (a .>= 1.0)
    a_where_pos = a[where_pos]
    if length(a_where_pos) > 0
        val, j = findmin(a_where_pos)
        i = range(1, stop=length(a))[where_pos][j]
        return val, i
    else
        return Inf, missing
    end
end

function find_next_spike_arrival(spike_ar_queue::Queue, t::Real, τ::Real)
    if length(spike_ar_queue) > 0
        time, unit = first(spike_ar_queue)
        return exp((time - t) / τ), unit
    else
        return Inf, missing
    end
end

function copy_units(p,W,units_to_copy)
    n = length(units_to_copy)
    oldN = p.N
    oldNE = p.NE
    newN = p.N + n
    newNE = p.NE + n
    #Now there is a n cell wide open space between N_in and N_out

    #copy connections to the rest of the network
    newW = zeros(Float64, newN, newN)
    for post in range(1,length=oldN)
        for pre in range(1,length=oldN)
            newPost = post
            newPre = pre
            if(newPost > p.N_in)
                newPost += n
            end
            if(newPre > p.N_in)
                newPre += n
            end
            newW[newPost,newPre] = W[post,pre]
        end
    end

    #copy in the new units
    for unit in range(1,length=n)
        newUnit = p.N_in+unit   #destination in the new array
        oldUnit = units_to_copy[unit]   #Source in the old array
        for post in range(1,length=oldN)
            newPost = post
            if(newPost > p.N_in)
                newPost += n
            end
            newW[newPost,newUnit] = W[post,oldUnit]
        end
        for pre in range(1,length=oldN)
            newPre = pre
            if(newPre > p.N_in)
                newPre += n
            end
            newW[newUnit,newPre] = W[oldUnit,pre]
        end
    end

    #copy inter-population weights
    for unitpost in range(1,length=n)
        newUnitPost = p.N_in+unitpost
        oldUnitPost = units_to_copy[unitpost]
        for unitpre in range(1,length=n)
            newUnitPre = p.N_in+unitpre
            oldUnitPre = units_to_copy[unitpre]
            newW[newUnitPost,newUnitPre] = W[oldUnitPost,oldUnitPre]
        end
    end

    #remove connections between output dummies back to recurrent units
    for unitpost in range(1,length=n)
        newUnitPost = p.N_in+unitpost
        oldUnitPost = units_to_copy[unitpost]
        if oldUnitPost > p.N_in
            oldUnitPost += n
        end
        for unitpre in range(1,length=n)
            newUnitPre = p.N_in+unitpre
            oldUnitPre = units_to_copy[unitpre]
            if oldUnitPre > p.N_in
                oldUnitPre += n
            end
            newW[newUnitPost,oldUnitPre] = 0
            #newW[oldUnitPost,newUnitPre] = 0
        end
    end

    #TODO: This needs to copy all the other variables
    p = LIFv2Params(N=newN,NE=newNE)
    W = SparseMatrixCSC(newW)
    return p,W
end

function force_vector_manipulations!(p,save_dir,W_out::Array{Float64},eta::Array{Float64},perturbations::Dict)
    new_wout = Array{Float64}(undef, p.N_out)
    new_eta = Array{Float64}(undef, p.N_out)
    if perturbations["forcevectorreset"]["method"] == "regenerate"
        new_wout, new_eta = gen_output_weights(p)
    elseif perturbations["forcevectorreset"]["method"] == "replace"
        network_weights = (HDF5.h5read(joinpath(save_dir,perturbations["forcevectorreset"]["run"],"network.h5"),"/"))
        new_wout = network_weights["W_out"]
        new_eta = network_weights["eta"]
    elseif perturbations["forcevectorreset"]["method"] == "parallelToStim"
        cells = load_cells(perturbations["forcevectorreset"]["run"],save_dir=save_dir)
        #Generate the new direction
        for i in range(1,length=p.N_out)
            new_wout[i] = cells[Symbol(string(i+p.N_in))].s_mean
        end
        new_wout_mag = norm(new_wout)
        old_wout_mag = norm(W_out)
        #Normalize to the magnitude of the old weight vectr
        new_wout = new_wout * (old_wout_mag/new_wout_mag)
    elseif perturbations["forcevectorreset"]["method"] == "orthogonalToStim"
        cells = load_cells(perturbations["forcevectorreset"]["run"],save_dir=save_dir)
        #Generate the new direction
        stim_resp = Array{Float64}(undef, p.N_out)
        for i in range(1,length=p.N_out)
            stim_resp[i] = cells[Symbol(string(i+p.N_in))].s_mean
        end
        #Calculate norms
        stim_mag = norm(stim_resp)
        old_wout_mag = norm(W_out)
        ##Calculate cosine similarity
        #cosine = dot(stim_resp,w.W_out)/stim_mag/old_wout_mag
        #Calculate parallel component
        parallel = (dot(stim_resp,W_out)/stim_mag/stim_mag)*stim_resp
        #Calculate orthogonal component
        orthogonal = W_out - parallel
        ortho_mag = norm(orthogonal)
        #Normalize to the magnitude of the old weight vectr
        new_wout = orthogonal * (old_wout_mag/ortho_mag)
    elseif perturbations["forcevectorreset"]["method"] == "orthogonalToBoth"
        cells = load_cells(perturbations["forcevectorreset"]["run"],save_dir=save_dir)
        #Generate the new direction
        stim_resp = Array{Float64}(undef, p.N_out)
        for i in range(1,length=p.N_out)
            stim_resp[i] = cells[Symbol(string(i+p.N_in))].s_mean
        end
        choice_resp = Array{Float64}(undef, p.N_out)
        for i in range(1,length=p.N_out)
            choice_resp[i] = cells[Symbol(string(i+p.N_in))].c_mean
        end
        #Calculate norms
        stim_mag = norm(stim_resp)
        choice_mag = norm(choice_resp)
        old_wout_mag = norm(W_out)
        #Orthogonalize to stim
        parallel = (dot(stim_resp,w.W_out)/stim_mag/stim_mag)*stim_resp
        orthogonal = W_out - parallel
        #Orthogonalize to resp
        parallel = (dot(choice_resp,orthogonal)/choice_mag/choice_mag)*choice_resp
        orthogonal = orthogonal - parallel
        #Normalize size
        ortho_mag = norm(orthogonal)
        new_wout = orthogonal * (old_wout_mag/ortho_mag)
    end
    # Reset weights
    if perturbations["forcevectorreset"]["target"] == "W_out"
        W_out = new_wout
    elseif perturbations["forcevectorreset"]["target"] == "eta"
        eta = new_eta
    elseif perturbations["forcevectorreset"]["target"] == "W_out+eta"
        W_out = new_wout
        eta = new_eta
    end
end

function verify_keywords(perturbations::Dict)
    if haskey(perturbations,"dummyinput")
        #N_dummy is the number of units to send dummy input to
        if !haskey(perturbations["dummyinput"],"N_dummy")
            throw(KeyError("N_dummy"))
        end
        #v_dummy is the size of dummy inputs to send
        if !haskey(perturbations["dummyinput"],"v_dummy")
            throw(KeyError("v_dummy"))
        end
        #p_dummy is the probability of each trial sending dummy inputs
        if !haskey(perturbations["dummyinput"],"p_dummy")
            throw(KeyError("p_dummy"))
        end
    end
    if haskey(perturbations,"inputnoise")
        if !haskey(perturbations["inputnoise"],"noise_type")
            throw(KeyError("noise_type"))
        end
        if !haskey(perturbations["inputnoise"],"noise_scale")
            throw(KeyError("noise_scale"))
        end
    end
    if haskey(perturbations,"shuffle")
        if !haskey(perturbations["shuffle"],"shuffle_method")
            throw(KeyError("shuffle_method"))
        end
        if !haskey(perturbations["shuffle"],"shuffle_sets")
            throw(KeyError("shuffle_sets"))
        end
    end
    if haskey(perturbations,"inactivation")
        #run is the run in which the firing rates and responsiveness have been
        #calculated. These are used to determine the ordering of cells for the
        #inactivation. This should be a test run.
        if !haskey(perturbations["inactivation"],"run")
            throw(KeyError("run"))
        end
        #The number of cells to inactivate
        if !haskey(perturbations["inactivation"],"number")
            throw(KeyError("number"))
        end
        #The direction in which to traverse the sorted list for inactivation
        if !haskey(perturbations["inactivation"],"type")
            throw(KeyError("type"))
        end
        #The method by which to inactivate cells:
        #replacement: cells are replaced with a dummy unit that spikes according
            #to an exponential distribution with the same firing rate as the real unit.
            #No changes are made to the output weight vector
        #silencing: cells are completely silenced by setting their inter-spike
            #interval to infinity. Output weight is irrelevant.
        #nooutput: this manipulation is the same as the "replacement" method except
            #that in addition to replacing units with dummies, the corresponding
            #force output weight is also set to zero, preventing these dummies
            #from affecting the integrated output.
        #onlyoutput: only the output vector is modified. Recurrent units are left
            #exactly as is, but their corresponding force output weights are set
            #to zero.
        if !haskey(perturbations["inactivation"],"method")
            throw(KeyError("method"))
        end
    end
    if haskey(perturbations,"forcevectorreset")
        #The run from which to draw weight vectors from.
        if !haskey(perturbations["forcevectorreset"],"run")
            throw(KeyError("run"))
        end
        #The target vector to maniulate. Options are as follows:
        #W_out: modifies the force output weight vector
        #eta: modifies the force feedback weight vector
        #W_out+eta: modifies both force vectors
        if !haskey(perturbations["forcevectorreset"],"target")
            throw(KeyError("target"))
        end
        #The particular manipulation to perform on the target vector(s)
        #regenerate: regenerate a fresh random vector according to the same
            #distribution that the vectors are originally drawn from
        #replace: replace the target weight vector with that of the network from
            #the run specified in the "run" keyword
        #parallelToStim: this replaces the output vector with a vector constructed
            #to have the same L2-norm as the current output vector, but in a direction
            #parallel to the stimulus responsiveness vector.
        #orthogonalToStim: this replaces the output vector with a vector constructed
            #to have the same L2-norm as the current output vector, but in a direction
            #orthogonal to the stimulus responsiveness vector. This is done by subtracting
            #the parallel component, so it remains somewhat similar to the original vector.
        #orthogonalToBoth: this replaces the output vector with a vector constructed
            #to have the same L2-norm as the current output vector, but in a direction
            #orthogonal to both the stimulus responsiveness and choice responsiveness
            #vectors. This is done by subtracting the parallel components, so it
            #remains somewhat similar to the original vector.
        if !haskey(perturbations["forcevectorreset"],"method")
            throw(KeyError("method"))
        end
    end
end

# Main network function
function LIFv2spikingnet(n_trials::Int, p::LIFv2Params, w::LIFv2Weights; perturbations::Dict=Dict(), force_learn::Bool=true, force_feedback::Bool=true, force_delay::Integer=0, force_end::Integer=0, stdp::Bool=true, stdp_delay::Integer=0, stdp_end::Integer=0, show_figs::Bool=true, fig_render_period::Integer=10, save_figs::Bool=true, plot_units::Union{Array, Missing}=missing, plot_plasticity::Bool=false, auto_save::Bool=false, save_period::Integer=1000, notes::String="", previous_run=nothing, I_adjust::Bool=false, I_adjust_dur::Integer=0, I_bias₀::Real=0.0, verbose::Bool=false, save_dir::String=default_dir, debug=false, progress_bar=true)
    # Check that for all experiments that this dictionary includes, values are present
    ## TODO: include descriptions of each, remove ones we don't use, roll most of this code into functions, move all initializations related to perturbations into this top code, and make sure names are 100% clear.
    verify_keywords(perturbations)

    #Initialize perturbation-specific variables
    if haskey(perturbations,"dummyinput")
        dummy_tr = [Int(rand()<perturbations["dummyinput"]["p_dummy"]) for i = 1:n_trials]
    end
    if haskey(perturbations,"shuffle")
        shuffle_recurrent_weights!(w, p, perturbations)
    end
    silenced_cells=nothing
    if haskey(perturbations,"inactivation")
        cells = nothing
        inactivation_cells=nothing
        silenced_cells=nothing
        if perturbations["inactivation"]["type"]=="NNR_sorted" || perturbations["inactivation"]["type"]=="R_sorted" || perturbations["inactivation"]["type"]=="Alt_sorted"
            cells = load_cells(perturbations["inactivation"]["run"],save_dir=save_dir)
            inactivation_cells = select_perturbation_set(p,cells,save_dir,perturbations)
            silenced_cells = [cell for cell in inactivation_cells]
        elseif perturbations["inactivation"]["type"]=="ascending" || perturbations["inactivation"]["type"]=="descending"
            cells = load_informativity(perturbations["inactivation"]["run"],save_dir=save_dir)
            inactivation_cells = select_perturbation_set(p,cells,save_dir,perturbations)
            silenced_cells = [cell for cell in inactivation_cells]
            cells = load_cells(perturbations["inactivation"]["run"],save_dir=save_dir)
        else
            throw(ArgumentError(perturbations["inactivation"]["type"]))
        end
        #inactivation_cells = select_perturbation_set(p,cells,save_dir,perturbations)
        #silenced_cells = [cell for cell in inactivation_cells] #deep copy

        #inactivation_cells and silenced_cells need to be distinguished for the
        #purpose of the split dummy inactivations, where the number of units changes
        #and thus the unit numbers change.
        perturbations["inactivation"]["cells"] = inactivation_cells
    end

    # Turn live figures on/off
    if show_figs
        plt.pygui(true);
    else
        plt.pygui(false);
    end

    # Pulling commonly used variables from the paramaters and weights
    W = SparseMatrixCSC(w.W)
    W_out = w.W_out
    PW = w.PW
    eta = w.eta
    ΔV = p.V_th - p.Vᵣ
    N_non = p.NE - p.N_in - p.N_out

    # Reset output weights if using a nooutput network. Must be done after pulling
    #pulling commonly used variables so as not to corrupt upstream references to
    #said variables (ensure that pulling produces deep copies)

    if haskey(perturbations,"forcevectorreset")
        force_vector_manupulations!(p,save_dir,W_out,eta,perturbations)
    end

    #NOTE: Not currently sure if this actually works for inactivating inhibitory
    #cells. Luckily that's not a priority right now.
    if (haskey(perturbations,"inactivation") && (perturbations["inactivation"]["method"]=="normaloutputdummyrecurrent"||perturbations["inactivation"]["method"]=="dummyoutputnormalrecurrent"))
        p,W = copy_units(p,W,silenced_cells)
        if perturbations["inactivation"]["method"]=="normaloutputdummyrecurrent"
            #Remove all outputs from the output layer cells
            n = length(silenced_cells)
            for unit in silenced_cells
                for post in range(1,length=p.N)
                    newUnit = unit
                    if newUnit > p.N_in
                        newUnit += n
                    end
                    W[post,newUnit] = 0
                end
            end
            #Silence all recurrent layer cells
            silenced_cells = collect(range(p.N_in+1,length=n))
            #Add metadata
            perturbations["inactivation"]["cellIDs"] = silenced_cells
        elseif perturbations["inactivation"]["method"]=="dummyoutputnormalrecurrent"
            #Remove all outputs from the output layer cells
            n = length(silenced_cells)
            for unit in silenced_cells
                for post in range(1,length=p.N)
                    newUnit = unit
                    if newUnit > p.N_in
                        newUnit += n
                    end
                    W[post,newUnit] = 0
                end
            end
            #Silence all output layer cells
            for cellnum in range(1,length=n)
                if silenced_cells[cellnum] > p.N_in
                    silenced_cells[cellnum] += n
                end
            end
            #Add metadata
            perturbations["inactivation"]["cellIDs"] = silenced_cells
        else
            throw(KeyError("unexpected method"))
        end
    end

    if haskey(perturbations,"inactivation") && (perturbations["inactivation"]["method"]=="nooutput" || perturbations["inactivation"]["method"]=="onlyoutput")
        for cell in silenced_cells
            cellindex = cell-p.N_in
            if cellindex > 0 && cellindex <= p.NE-p.N_in #Must check that the cell has an output weight before setting to zero
                W_out[cellindex] = 0
            end
        end
    end

    # Setting STDP Functions
    if p.EE_stdp_type == "+"
        EE_pre = STDP_EE_pre_add
        EE_post = STDP_EE_post_add
    elseif p.EE_stdp_type == "*"
        EE_pre = STDP_EE_pre_mul
        EE_post = STDP_EE_post_mul
    end
    if p.IE_stdp_type == "+"
        IE_pre = STDP_IE_pre_add
        IE_post = STDP_IE_post_add
    elseif p.EE_stdp_type == "*"
        IE_pre = STDP_IE_pre_mul
        IE_post = STDP_IE_post_mul
    end
    if p.hetero_stdp_type == "rate"
        IE_hetero = STDP_hetero_rate
        EE_hetero = STDP_hetero_rate
    elseif p.hetero_stdp_type == "target"
        IE_hetero = STDP_hetero_target
        EE_hetero = STDP_hetero_target
    end
    E_set = p.jₑ₀ * sqrt(p.NE * p.p_con)
    I_set = p.jᵢ₀ * sqrt(p.NI * p.p_con)

    # Initializing task parameters
    weights = [f == p.target_freq ? p.p_target :
               (1.0 - p.p_target) / (length(p.freqs) - 1.0) for f in p.freqs]
    freqs_tr = [sample(p.freqs, Weights(weights)) for i = 1:n_trials]

    if haskey(perturbations,"dummyinput")
        
    end

    n_freq = length(p.freqs)
    f_out = gen_output_function(p.target_freq, p.resp_offset, p.resp_dur, p.out_bias, p.output_type)

    # Initializing random gaps between trials
    ITIs = (p.ITI_min + p.T) .+ ((p.ITI_max - p.ITI_min) .* rand(n_trials - 1))
    trial_offsets = cumsum(ITIs)
    prepend!(trial_offsets, 0.0)

    # Initialzing FORCE mechanisms
    S_out = fill(0.0, p.N_out)
    z_out = 0.0

    # Initializing PSPs
    epsp = zeros(p.N)
    ipsp = zeros(p.N)

    # Initializing STDP mechanisms
    z₊ = zeros(p.NE)
    z₋ = zeros(p.NE)
    zᵢ₊ = zeros(p.NI)
    zᵢ₋ = zeros(p.NE)

    #Initialize Plasticity Trackers
    if plot_plasticity
        plasticity_tracker = zeros(8)
    end

    # Initializing bias current to be above rheobase
    if I_bias₀ > ΔV
        I_bias = I_bias₀
    else
        I_bias = ΔV * 1.01
    end
    I₀ = fill(I_bias, p.N)
    I₁ = fill(I_bias, p.N)

    # Initializing feedback current
    I_feed = 0

    # Initializing time to spike
    if I_bias >= ΔV
        et = ones(p.N) + rand(p.N) # want et > 1
    else
        et = rand(p.N) # want et < 1
    end

    # Setting maximum time between events
    t_max = 0.1 # ms
    δet_max = exp(t_max / p.τ)

    # Initializing spiking queue and spike times
    spike_ar_queue = Queue{Tuple{Float64,Int64}}()
    δet_ar = Inf
    j_ar = missing

    δet_sp = Inf
    j_sp = missing

    js = Array{Union{Missing,Int64}}([missing, j_sp, j_ar])
    δets = Array{Union{Missing,Float64}}([δet_max, δet_sp, δet_ar])

    # Initializing time and save counter
    t = 0
    t_trial = 0
    activity_save_number = 0

    # Variables to track per trial
    stimulus_tr = Float64[]
    output_tr = Float64[]
    if show_figs | save_figs
        rates_tr = Array{Float64,1}[]
        plasticity_tr = Array{Float64,1}[]
        current_tr = Float64[]
        feedback_tr = Float64[]
        IDom_tr = Float64[]
        tar_tr = Int[]
        foil_tr = Int[]
        tar_out_tr = Float64[]
        foil_out_tr = Float64[]
        WEE_tr = Float64[]
        WIE_tr = Float64[]
        W_out_tr = Float64[]
    end
    activity_tr = Tuple{Int, Int, Float64}[]

    # Initializing figures
    plt.close("all")
    if show_figs | save_figs
        fig_rates, ax_rates = init_rate_figure(p)
        fig_current, ax_current = init_current_figure(p)
        fig_IDom, ax_IDom = init_IDom_figure(p)
        fig_raster, ax_raster = init_raster_figure(p)
        fig_output, ax_output = init_output_figure(p)
        fig_weights, ax_weights = init_weights_figure(p)
        if !ismissing(plot_units)
            N_ex = length(plot_units)
            fig_examples, ax_examples = init_example_unit_figure(p, plot_units)
        end
        if plot_plasticity
            fig_plasticity, ax_plasticity = init_plasticity_figure(p)
        end
    end

    run_name = "run " * timestamp()
    if auto_save | save_figs
        check_dir(run_name, save_dir=save_dir)
    end
    print("Starting LIFv2 $run_name...\n")
    if progress_bar
        trial_iter = ProgressBar(1:n_trials)
    else
        trial_iter = 1:n_trials
    end
    for trial in trial_iter
        # Resetting variables to track per moment
        spike_idx = Int64[] #initialize time
        spike_times = Float64[] # spike raster
        t_t = Float64[]
        z_out_t = Float64[]
        f_out_t = Float64[]
        nspike = [0, 0, 0, 0] # in, non, out, inh

        if show_figs | save_figs
            I_t = Array{Float64,1}[]
            I_minus_t = Array{Float64,1}[]
            I_plus_t = Array{Float64,1}[]
            V_t = Array{Float64,1}[]
        end


        # selecting stimulus
        freq = freqs_tr[trial]
        N_set, amp = place_code_stimulus(p.N_in, freq + p.ν_bias, n_freq, p.ν_scale; perturbations=perturbations)

        #add dummy inputs if necessary
        # TODO: move to top
        if haskey(perturbations,"dummyinput")
            dummy = dummy_tr[trial]
            N_set_dummy, amp_dummy = dummy_inputs(p.N_in, perturbations["dummyinput"]["N_dummy"], perturbations["dummyinput"]["v_dummy"] * dummy)
        end

        # Setting bias current
        if trial>=2
            I₀ = fill(I_bias, p.N)
        end

        # Creating description string to print to console next to progress bar
        descriptions = String[]
        
        # Setting trial time
        t_trial = t - trial_offsets[trial]
        while t_trial < p.T
            # find current
            I_in = input_current(t_trial, N_set, amp, p.stim_offset, p.stim_dur, p.N, perturbations=perturbations)

            if haskey(perturbations,"dummyinput")
                I_in += dummy_current(t_trial,N_set_dummy,amp_dummy,p.stim_offset,p.stim_dur,p.N)
            end

            if force_feedback
                I_temp = FORCE_feedback_current(z_out, eta, p.Q, p.N, p.NE, p.N_out)
                I_in += I_temp
                I_feed = 0
            end

            I₂ = (I₀ .+ I_in .+ epsp .- ipsp)

            # Adjusting the time to spike based on new current levels
            current_tf!(et, I₁, I₂, ΔV; perturbations=perturbations, silenced_cells=silenced_cells)

            # Finding next event
            δet_sp, j_sp = find_next_spike(et)
            δets[2] = δet_sp
            js[2] = j_sp

            δet_ar, j_ar = find_next_spike_arrival(spike_ar_queue, t, p.τ)
            δets[3] = δet_ar
            js[3] = j_ar

            δet, event = findmin(δets)
            j = js[event]

            # Advancing time to the event
            for (variable, timescale) in [
                (et, p.τ),
                (S_out, p.τ_out),
                (z₊, p.τ₊),
                (z₋, p.τ₋),
                (zᵢ₊, p.τᵢ),
                (zᵢ₋, p.τᵢ),
                (epsp, p.τ_epsp),
                (ipsp, p.τ_ipsp),
                ]
                advance_time!(variable, δet, p.τ, timescale)
            end
            dt = to_dt(δet, p.τ)
            t += dt
            t_trial += dt

            # Code for each event type
            if event == 1 # Advancing time w/o event
                nothing
            elseif event == 2 # Spike generated
                # TODO: change to `(j in silenced_cells)` 
                if haskey(perturbations,"inactivation") && perturbations["inactivation"]["method"]!="onlyoutput" && length(setdiff(Set(j),silenced_cells))==0
                    # default: poisson replacement
                    # TODO: make this the overall firing rate AND/OR the firing rate for the relevant period (pre, stimulus, response) . Probably should be two different methods.
                    dummyfreq = cells[Symbol(string(j))].fr_mean
                    # silenced: no firing
                    if perturbations["inactivation"]["method"]=="silencing"
                        #print("Silencing "*string(j))
                        dummyfreq=0
                    end
                    #print("Cell # "*string(j)*" dummied")
                    time = 1000*rand(Exponential(1/dummyfreq))
                    et[j] = exp(time / p.τ)
                else
                    et[j] = eT₊(I₂[j], ΔV) # reset to resting voltage
                end
                # Incrementing output rate
                if (j > p.NE - p.N_out) & (j <= p.NE)
                    S_out[j-(p.NE-p.N_out)] += 1.0
                end
                enqueue!(spike_ar_queue, (t + p.d, j))
            elseif event == 3 # Spike arrives
                dequeue!(spike_ar_queue)
                if j <= p.NE
                    z₊[j] += 1.0
                    z₋[j] += 1.0
                    zᵢ₋[j] += 1.0
                else
                    zᵢ₊[j-p.NE] += 1.0
                end
                spike_tf!(epsp, ipsp, j, W, p.NE)
                if stdp && (trial > stdp_delay) && (t_trial >= 0) && ((trial <= stdp_end) || stdp_end==0)
                    if plot_plasticity
                        Δplasticity = STDP_track!(W,
                            j, p.NE,
                            EE_pre, p.A,
                            EE_post, p.B,
                            EE_hetero, E_set, p.β, p.δ,
                            IE_pre, p.ηᵢ,
                            IE_post, p.αᵢ,
                            IE_hetero, I_set, p.βᵢ, p.δᵢ,
                            z₊, z₋, zᵢ₊, zᵢ₋,
                            p.w_min, p.w_max)
                        plasticity_tracker += Δplasticity
                    else
                        STDP!(W,
                            j, p.NE,
                            EE_pre, p.A,
                            EE_post, p.B,
                            EE_hetero, E_set, p.β, p.δ,
                            IE_pre, p.ηᵢ,
                            IE_post, p.αᵢ,
                            IE_hetero, I_set, p.βᵢ, p.δᵢ,
                            z₊, z₋, zᵢ₊, zᵢ₋,
                            p.w_min, p.w_max)
                    end
                end
            end

            # Calculating network output
            z_out = dot(W_out, S_out)
            f_out_now = f_out(freq, t_trial)

            # Force training
            if force_learn && (trial > force_delay) && (t_trial >= 0) && (rand() <= (dt / p.t_learn_dur)) && ((trial <= force_end) || force_end==0)
                err = z_out - f_out_now
                W_out, PW = FORCE_learn(W_out, PW, err, S_out)
            end

            # Variables to monitor on a moment basis during the trial
            if (t_trial >= 0)
                push!(t_t, t_trial)
                push!(z_out_t, z_out)
                push!(f_out_t, f_out_now)
                if event == 2 # A Neuron is spiking
                    push!(spike_times, t_trial) # store spike time
                    push!(spike_idx, j) # store spiking neuron
                    if j <= p.N_in # ex. input
                        nspike[1] += 1
                    elseif (j <= p.NE - p.N_out) #ex. non input non output
                        nspike[2] += 1
                    elseif (j <= p.NE) # ex. output
                        nspike[3] += 1
                    else # inh
                        nspike[4] += 1
                    end
                end

                if (show_figs | save_figs) & !ismissing(plot_units)
                    V = Voltage.(et[plot_units], I₂[plot_units], -p.Vᵣ, ΔV)
                    if !ismissing(j) & (event == 2) & (j ∈ plot_units)
                        i = findnext(plot_units .== j, 1)
                        V[i] = -20.0
                    end
                    push!(V_t, V)
                    push!(I_t, I_in[plot_units])
                    push!(I_plus_t, epsp[plot_units])
                    push!(I_minus_t, ipsp[plot_units])
                end
            end 
            # Setting past current based on present current
            I₁ = deepcopy(I₂)
        end
        
        if debug
        # Finding cells that have "jumped the shark"
            pos = I₁ .>= ΔV 
            num_jts = sum(pos .& (et .< 1.0)) + sum(.!pos .& (et .> 1.0))
            #num_jts2 = sum(Voltage.(et, I₁, -p.Vᵣ, ΔV) .>= p.V_th)
            push!(descriptions, @sprintf("JTS=%i", num_jts))
        end

        # Variables to monitor on a per trial basis
        inh_rate = (nspike[4] * 1000.0) / (p.NI * p.T)
        if show_figs | save_figs
            bl = trapz(t_t, z_out_t, 0, p.stim_offset)
            resp = trapz(t_t, z_out_t, p.resp_offset, p.resp_offset + p.resp_dur)
            rates = (nspike * 1000.0) ./ ([p.N_in, N_non, p.N_out, p.NI] .* p.T)
            exc_rate = ((nspike[1] + nspike[2] + nspike[3]) * 1000.0 / (p.NE * p.T))
            push!(descriptions, @sprintf("i-rate=%.2f", inh_rate))
            push!(descriptions, @sprintf("e-rate=%.2f", exc_rate))
            push!(stimulus_tr, freq)
            push!(output_tr, resp - bl)
            push!(activity_tr, zip(fill(trial, length(spike_times)), spike_idx, spike_times)...)#... are hcat?
        end

        # adjusting input current to match target rate
        if I_adjust & (trial <= I_adjust_dur)
            I_bias += p.η_rate * (p.inh_rate_target - inh_rate)
            if I_bias < 0
                I_bias = 0
            end
        elseif !I_adjust
            I_bias = I_bias₀
        end
        push!(descriptions, @sprintf("I=%.2f", I_bias))

        if show_figs | save_figs
            if (freq == p.target_freq)
                push!(tar_tr, trial)
                push!(tar_out_tr, resp - bl)
            else
                push!(foil_tr, trial)
                push!(foil_out_tr, resp - bl)
            end
            push!(rates_tr, rates)
            (out, in, vals) = findnz(W)
            avg_WEE = mean(vals[(in.<=p.NE).&(out.<= p.NE)])
            avg_WIE = mean(vals[(in.>p.NE).&(out.<= p.NE)])
            avg_W_out = mean(abs.(W_out))
            push!(WEE_tr, avg_WEE)
            push!(WIE_tr, avg_WIE)
            push!(W_out_tr, avg_W_out)
            push!(current_tr, I_bias)
            push!(feedback_tr, I_feed)
            push!(IDom_tr, ((p.NI*avg_WIE)/(p.NE*avg_WEE)) )
            if plot_plasticity
                if (trial == 1)
                    plasticity_tr = plasticity_tracker
                else
                    plasticity_tr = hcat(plasticity_tr, plasticity_tracker)
                end
            end

            # Render figures
            if (trial == 1) | (trial % fig_render_period == 0) | (trial == n_trials)
                update_rate_figure(fig_rates, ax_rates, p, trial, rates_tr; save_fig=save_figs, run=run_name, save_dir=save_dir)
                update_current_figure(fig_current, ax_current, p, trial, current_tr, feedback_tr; save_fig=save_figs, run=run_name, save_dir=save_dir)
                update_IDom_figure(fig_IDom, ax_IDom, p, trial, IDom_tr; save_fig=save_figs, run=run_name, save_dir=save_dir)
                update_raster_figure(fig_raster, ax_raster, p, trial, freq, spike_times, spike_idx; save_fig=save_figs, run=run_name, save_dir=save_dir)
                update_output_figure(fig_output, ax_output, p, trial, t_t, z_out_t, f_out_t, tar_tr, tar_out_tr, foil_tr, foil_out_tr; save_fig=save_figs, run=run_name, save_dir=save_dir)
                update_weights_figure(fig_weights, ax_weights, p, trial, W, WEE_tr, WIE_tr, W_out_tr; save_fig=save_figs, run=run_name, save_dir=save_dir)
                if !ismissing(plot_units)
                    update_example_unit_figure(fig_examples, ax_examples, p, plot_units, t_t, I_t, I_minus_t, I_plus_t, V_t; I_scale=1.0, save_fig=save_figs, run=run_name, save_dir=save_dir)
                end
                if plot_plasticity
                    update_plasticity_figure(fig_plasticity, ax_plasticity, p, trial, plasticity_tr; save_fig=save_figs, run=run_name, save_dir=save_dir)
                end
                push!(descriptions, "Rendered.")
            end
        end

        # Saving variables
        if (auto_save & ((trial == 1) | (trial % save_period == 0) | (trial == n_trials))) | (trial==n_trials) | (trial==stdp_delay) | (trial==stdp_end) | (trial==force_delay) | (trial==force_end)
            activity_save_number += 1
            #Must save out file
            save_temp_activity(activity_tr,activity_save_number,run_name,save_dir=save_dir)

            weights = LIFv2Weights(W, W_out, eta, PW);
            metadata = LIFv2Metadata("LIFv2", run_name, previous_run, notes, trial, stdp, stdp_delay, stdp_end, force_learn, force_feedback, force_delay, force_end, I_adjust, I_adjust_dur, I_bias, perturbations);

            if trial == 1
                save_weights(weights,run_name,save_dir=save_dir,filename="network_start.h5")
            end
            if trial == stdp_delay
                save_weights(weights,run_name,save_dir=save_dir,filename="network_STDP_start.h5")
            end
            if trial == stdp_end
                save_weights(weights,run_name,save_dir=save_dir,filename="network_STDP_end.h5")
            end
            if trial == force_delay
                save_weights(weights,run_name,save_dir=save_dir,filename="network_FORCE_start.h5")
            end
            if trial == force_end
                save_weights(weights,run_name,save_dir=save_dir,filename="network_FORCE_end.h5")
            end

            save_network(p, metadata, weights, stimulus_tr, output_tr; save_dir=save_dir)
            push!(descriptions, "Saved.")

            # Resetting activity variables
            activity = nothing
            activity_tr = Tuple{Int, Int, Float64}[]

            if (trial == n_trials)
                # Accumulate activity into single array and save it
                activity = consolidate_activity(activity_save_number,run_name,save_dir=save_dir)
                save_activity(activity,run_name,save_dir=save_dir)
                #Clean Directory
                clean_temp_activity(activity_save_number,run_name,save_dir=save_dir)
                #Return values
                return(metadata, weights, activity, stimulus_tr, output_tr)
            end
            activity_tr = Tuple{Int, Int, Float64}[]
        end
        # Update descriptions to progress bar
        progress_bar ? set_description(trial_iter, join(descriptions, "|")) : missing
    end
end
