module LimberJack

export Settings, CosmoPar, Cosmology, Ez, Hmpc, comoving_radial_distance
export growth_factor, growth_rate, fs8, sigma8
export lin_Pk, nonlin_Pk
export NumberCountsTracer, WeakLensingTracer, CMBLensingTracer
export angularCℓs, angularCℓ, angularCℓFast
export Theory, TheoryFast
export make_data

using Interpolations, LinearAlgebra, Statistics, QuadGK, Bolt
using NPZ, NumericalIntegration, PythonCall, LoopVectorization
#using OrdinaryDiffEq

# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

include("core.jl")
include("boltzmann.jl")
include("growth.jl")
include("halofit.jl")
include("tracers.jl")
include("spectra.jl")
include("theory.jl")
include("data_utils.jl")

end
