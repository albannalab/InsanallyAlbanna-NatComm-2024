module Types

export RecurrentNetworkModelParams,
RecurrentNetworkModelWeights,
RecurrentNetworkModelMetadata, @add_metadata_fields,
RecurrentNetworkModelTypes

abstract type RecurrentNetworkModelParams end

abstract type RecurrentNetworkModelWeights end

macro add_metadata_fields()
    return esc(:(
        model::String;
        run::String;
        previous_run::Union{String, Nothing};
        notes::String;)
    )
end

abstract type RecurrentNetworkModelMetadata end

struct RecurrentNetworkModelTypes
    Params::Type where Params <: RecurrentNetworkModelParams
    Weights::Type where Weights <: RecurrentNetworkModelWeights
    Metadata::Type where Metadata <: RecurrentNetworkModelMetadata
end

end
