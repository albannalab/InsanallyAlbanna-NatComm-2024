module Math

export pick_from_array, trapz, struct_is_same, clip_range

function pick_from_array(arr, n)
    [a[n] for a in arr]
end

function trapz(x, y, a, b)
    vals = (x.>=a) .& (x.<b)
    xp = x[vals]
    yp = y[vals]
    sum([(xp[i+1] - xp[i]) * (yp[i] + yp[i+1]) / 2.0  for i=1:(length(xp)-1)])
end

function struct_is_same(s1, s2)
    all(getfield(s1, f) == getfield(s2, f) for f in fieldnames(typeof(s1)))
end

function clip_range(w::Float64, w_min::Float64, w_max::Float64)
    if w < w_min
        return w_min
    elseif w > w_max
        return w_max
    else
        return w
    end
end

end
