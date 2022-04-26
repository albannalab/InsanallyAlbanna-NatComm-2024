module Force

using LinearAlgebra

export FORCE_feedback_current, FORCE_learn

function FORCE_feedback_current(z_out::Real, eta::Array, Q::Real, N::Integer, NE::Integer, N_out::Integer)
    I_add = zeros(N)
    I_add[1+NE-N_out:NE] = Q * z_out .* eta
    I_add
end

function FORCE_learn(W_out::Array, PW::Matrix, err::Real, S_out::Array)
    kW = PW * S_out
    rPrW = dot(S_out, kW)
    cW = 1.0 / (1.0 + rPrW)
    PW -= cW .* kW * kW'
    W_out -= cW * err * kW
    W_out, PW
end

end
