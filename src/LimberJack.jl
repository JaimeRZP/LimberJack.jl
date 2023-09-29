module LimberJack

export Settings, CosmoPar, Cosmology, Ez, Hmpc, comoving_radial_distance
export growth_factor, growth_rate, fs8, sigma8
export lin_Pk, nonlin_Pk
export NumberCountsTracer, WeakLensingTracer, CMBLensingTracer, get_IA
export angularCℓs, angularCℓ, angularCℓFast
export Theory
export make_data

using Interpolations, LinearAlgebra, Statistics, QuadGK
using NPZ, NumericalIntegration, PythonCall, Artifacts 
include("core.jl")
include("boltzmann.jl")
include("data_utils.jl")
include("growth.jl")
include("halofit.jl")
include("tracers.jl")
include("spectra.jl")
include("theory.jl")

# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    emupk_files = npzread(joinpath(artifact"emupk", "emupk.npz"))
    global emulator = Emulator(emupk_files)

    @static if !isdefined(Base, :get_extension)
        @require Bolt = "d94d39b4-3f51-4e5a-bfd8-f3b08e8f2b62" begin
            include("../ext/BoltExt.jl")
        end
    end
end       

end # module
