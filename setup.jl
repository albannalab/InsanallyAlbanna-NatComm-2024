using Pkg

# activate the environment defined by `Manifest.toml` and `Project.toml`
Pkg.activate(joinpath(@__DIR__))

# Download and precompile the dependencies
Pkg.instantiate()
Pkg.precompile()