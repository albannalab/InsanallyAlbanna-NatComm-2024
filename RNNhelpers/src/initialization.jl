module Initialization

using LinearAlgebra
using SparseArrays
using Random
using ..Types
import StatsBase: mean

export gen_recurrent_weights, gen_output_weights, gen_PW₀, shuffle_recurrent_weights!

function check_for_at_least_one_input(W::SparseMatrixCSC)
    n_rows, _ = size(W)
    for i in range(1,length=n_rows)
        n_nonzero = length(findnz(W[i,:])[1])
        if n_nonzero == 0
            return(false)
        end
    end
    return(true)
end

function gen_recurrent_weights(p::RecurrentNetworkModelParams)
    all_cells_have_inputs = false
    Wₑₑ = nothing
    Wᵢₑ = nothing
    Wₑᵢ = nothing
    Wᵢᵢ = nothing
    while !all_cells_have_inputs
        Wₑₑ = (2*p.jₑ₀ / sqrt(p.NE * p.p_con)) .* sprand(Float64, p.NE, p.NE, p.p_con) # E to E
        Wᵢₑ = (2*p.jᵢ₀ / sqrt(p.NI * p.p_con)) .* sprand(Float64, p.NE, p.NI, p.p_con) # I to E
        Wₑᵢ = (2*p.jₑ₀ / sqrt(p.NE * p.p_con)) .* sprand(Float64, p.NI, p.NE, p.p_con) # E to I
        Wᵢᵢ = (2*p.jᵢ₀ / sqrt(p.NI * p.p_con)) .* sprand(Float64, p.NI, p.NI, p.p_con) # I to I
        all_cells_have_inputs = all([check_for_at_least_one_input(w) for w in (Wₑₑ, Wᵢₑ, Wₑᵢ, Wᵢᵢ)])
    end
    W = vcat(hcat(Wₑₑ, Wᵢₑ), hcat(Wₑᵢ, Wᵢᵢ))
    W
end

function shuffle_recurrent_weights!(w::RecurrentNetworkModelWeights, p::RecurrentNetworkModelParams; perturbations::Dict=Dict())
    method = perturbations["shuffle"]["shuffle_method"]
    sets = perturbations["shuffle"]["shuffle_sets"]
    # all sets
    # sets = [(1:p.NE, 1:p.NE), (1:p.NE, p.NE+1:p.N), (p.NE+1:p.N, 1:p.NE), (p.NE+1:p.N, p.NE+1:p.N)]

    if sets == "both"
        # EE, IE
        sets = [(1:p.NE, 1:p.NE), (1:p.NE, p.NE+1:p.N)]    
    elseif sets == "EE"
        sets = [(1:p.NE, 1:p.NE)]    
    elseif sets == "IE"
        sets = [(1:p.NE, p.NE+1:p.N)]    
    end
    
    W = SparseMatrixCSC(w.W)
    for set in sets
        if method == "all"
            W[set...] = shuffle(W[set...])
        elseif method == "inputs"
            for i in set[1]
                W[i, set[2]] = shuffle(W[i, set[2]])
            end
        elseif method == "outputs"
            for j in set[2]
                W[set[1], j] = shuffle(W[set[1], j])
            end
        elseif method == "mean"
            m = collect(set[1])[end] - collect(set[1])[1] + 1
            n = collect(set[2])[end] - collect(set[2])[1] + 1
            mean_w = mean(findnz(W[set...])[3])
            sw = 2*mean_w .* sprand(Float64, m, n, p.p_con)
            W[set...] = sw
        end
    end
    dropzeros!(W)
    w.W = W
end

function gen_output_weights(p::RecurrentNetworkModelParams)
    eta = 2.0 .* rand(p.N_out) .- 1.0
    W_out = p.k₀ .* randn(p.N_out) # Output weights, only recieves imputs
    W_out, eta
end

function gen_PW₀(p::RecurrentNetworkModelParams)
    p.Pw₀ .* Matrix{Float64}(I, p.N_out, p.N_out)
end

end
