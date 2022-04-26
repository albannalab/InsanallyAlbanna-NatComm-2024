module IO

using DataFrames
using Parameters
using ..Types
import JSON2
import HDF5
import JLD
import CSV
import Dates

export load_weights, save_weights, load_params, save_params, save_experiment_params, load_metadata, save_metadata, save_experiment_metadata, modify_metadata, save_network, load_network, load_sets, reinterpret_array_of_tuples, timestamp, modelof, default_dir, check_dir, save_activity, save_temp_activity, consolidate_activity, clean_temp_activity, load_cells, load_informativity

default_dir = "./results"

# Basic functions

function save_to_json(file_name::String, object::Union{RecurrentNetworkModelParams, RecurrentNetworkModelMetadata, NamedTuple})
    open(file_name, "w") do io
        write(io, JSON2.write(object));
    end
    nothing
end

function load_from_json(file_name::String)
    open(x -> JSON2.read(read(x, String)), file_name);
end

# TODO
# function save_arrays_to_jld(file_name::String, arrays::NamedTuple)
#     for (name, array) = arrays
#         JLD.save(file_name, name, array)
#     end
#     nothing
# end
#
# function load_arrays_from_jld(file_name::String)
#     # TODO
# end

function save_arrays_to_hdf5(file_name::String, variables::Union{RecurrentNetworkModelWeights, NamedTuple})
    variable_names = fieldnames(typeof(variables))
    if isfile(file_name)
        rm(file_name)
    end
    for name = variable_names
        array = getfield(variables, name)
        HDF5.h5write(file_name, String(name), array)
    end
    nothing
end

function load_arrays_from_hdf5(file_name::String)
    HDF5.h5read(file_name, "/")
end

function save_outputs_to_csv(file_name::String, cols)
    df = DataFrame(trial=1:length(cols[1]); cols...);
    CSV.write(file_name, df);
    nothing
end

# Higher level functions

function modelof(run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "metadata.json");
    metadata = open(x -> JSON2.read(read(x, String)), full_path);
    model_name = metadata[:model]
    return Meta.parse(model_name)
end


function load_weights(run_name::String, model::RecurrentNetworkModelTypes; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "network.h5");
    arrays = load_arrays_from_hdf5(full_path)
    model.Weights(values(arrays)...) # TODO: check variable names
end


function save_weights(weights::RecurrentNetworkModelWeights, run_name::String; save_dir=default_dir,filename="network.h5")
    full_path = joinpath(save_dir, run_name, filename);
    save_arrays_to_hdf5(full_path, weights);
    nothing
end

function load_params(run_name::String, model::RecurrentNetworkModelTypes; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "params.json");
    model.Params(load_from_json(full_path)...)
end


function save_params(params::RecurrentNetworkModelParams, run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "params.json");
    save_to_json(full_path, params);
    nothing
end


function save_experiment_params(params::NamedTuple, run_name::String, exp_name::String; save_dir=".")
    full_path = joinpath(save_dir, exp_name, run_name, "exp_params.json");
    save_to_json(full_path, params);
end


function load_metadata(run_name::String, model::RecurrentNetworkModelTypes; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "metadata.json");
    model.Metadata(load_from_json(full_path)...)
end


function save_metadata(metadata::RecurrentNetworkModelMetadata, run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "metadata.json");
    save_to_json(full_path, metadata);
    nothing
end

function save_experiment_metadata(meta::NamedTuple, exp_name::String; save_dir=".")
    check_dir(exp_name, save_dir=save_dir)
    full_path = joinpath(save_dir, exp_name, "exp_metadata.json");
    save_to_json(full_path, meta);
end

function modify_metadata(run_name::String, Struct::Type, key::Symbol, value::Any; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "metadata.json");
    metadata = Struct(load_from_json(full_path)...);
    setproperty!(metadata, key, value);
    save_to_json(full_path, metadata);
    nothing
end

function load_sets(run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "analysis_sets.json");
    sets = load_from_json(full_path)
    sets
end

function load_cells(run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "analysis_cells.json");
    cells = load_from_json(full_path)
    cells
end

function load_informativity(run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name, "analysis_informativity.json");
    informativity = load_from_json(full_path)
    informativity
end

function check_dir(run_name::String; save_dir=default_dir)
    full_path = joinpath(save_dir, run_name);
    if !isdir(full_path)
        mkpath(full_path);
    end
end


function save_network(params::RecurrentNetworkModelParams, metadata::RecurrentNetworkModelMetadata, weights::RecurrentNetworkModelWeights, inputs::Array, outputs::Array; save_dir::String=default_dir)
    run_name = metadata.run
    check_dir(run_name, save_dir=save_dir)
    full_path = joinpath(save_dir, run_name);
    save_to_json(joinpath(full_path, "params.json"), params)
    save_to_json(joinpath(full_path, "metadata.json"), metadata)
    save_arrays_to_hdf5(joinpath(full_path, "network.h5"), weights)
    save_outputs_to_csv(joinpath(full_path, "inputs_outputs.csv"), (inputs=inputs, outputs=outputs))
end


function load_network(run_name::String, model::RecurrentNetworkModelTypes; save_dir::String=default_dir)
    params = load_params(run_name, model; save_dir=save_dir);
    metadata = load_metadata(run_name, model; save_dir=save_dir);
    weights = load_weights(run_name, model; save_dir=save_dir);
    params, metadata, weights
end


function reinterpret_array_of_tuples(array)
    a = [[Float64(i) for i in a] for a in array]
    copy(transpose(hcat(a...)))
end

function timestamp()
    Dates.format(Dates.now(), Dates.DateFormat("yyyy-mm-dd HHMMSSsss"))
end

function save_temp_activity(activity_tr::Array, iteration::Int, run_name::String; save_dir::String=default_dir)
    full_path = joinpath(save_dir, run_name);
    activity = reinterpret_array_of_tuples(activity_tr)
    save_arrays_to_hdf5(joinpath(full_path,("activity_save_" * string(iteration) * ".h5")), (Activity=activity,))
end

function clean_temp_activity(segments::Int, run_name::String; save_dir::String=default_dir)
    full_path = joinpath(save_dir, run_name);
    #Load activity segments
    for i in range(1,length=segments)
        rm(joinpath(full_path,("activity_save_" * string(i) * ".h5")))
    end
end

function consolidate_activity(segments::Int, run_name::String; save_dir::String=default_dir)
    full_path = joinpath(save_dir, run_name);
    saved_activity = nothing
    activity = nothing
    #Load activity segments
    for i in range(1,length=segments)
        saved_activity = (HDF5.h5read(joinpath(full_path,("activity_save_" * string(i) * ".h5")), "/"))
        saved_activity = saved_activity["Activity"]
        if i == 1
            activity = saved_activity
        else
            activity = [saved_activity; activity]
        end
    end
    #Return activity matrix
    activity
end

function save_activity(activity::Array,run_name::String; save_dir::String=default_dir)
    full_path = joinpath(save_dir, run_name);
    save_arrays_to_hdf5(joinpath(full_path,"activity.h5"), (Activity=activity,))
end

end
