module RNNhelpers

include("types.jl")
include("math.jl")
include("IO.jl")
include("plots.jl")
include("force.jl")
include("initialization.jl")

using .Types
using .Math
using .IO
using .Plots
using .Force
using .Initialization

export init_rate_figure, update_rate_figure,
init_plasticity_figure, update_plasticity_figure,
init_current_figure, update_current_figure,
init_IDom_figure, update_IDom_figure,
init_raster_figure, update_raster_figure,
init_output_figure, update_output_figure,
init_example_unit_figure, update_example_unit_figure,
init_weights_figure, update_weights_figure,
pick_from_array, trapz, struct_is_same, clip_range,
RecurrentNetworkModelParams,
RecurrentNetworkModelMetadata,
RecurrentNetworkModelWeights,
@add_metadata_fields,
RecurrentNetworkModelTypes,
load_weights, save_weights,
load_params, save_params, save_experiment_params, load_informativity,
load_metadata, save_metadata, modify_metadata, save_experiment_metadata, load_sets, load_cells,
save_network, load_network, save_activity, save_temp_activity, consolidate_activity, clean_temp_activity,
default_dir, check_dir,
reinterpret_array_of_tuples, timestamp, modelof,
FORCE_feedback_current, FORCE_learn,
gen_output_weights, gen_recurrent_weights, gen_PW₀,
shuffle_recurrent_weights!

end
