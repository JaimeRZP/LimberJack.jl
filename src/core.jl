"""
    Settings(;kwargs...)

Constructor of settings structure constructor. 

Kwargs:

- `nz::Int=300` : number of nodes in the general redshift array.
- `nz_chi::Int=1000` : number of nodes in the redshift array used to compute matter power spectrum grid.
- `nz_t::Int=350` : number of nodes in the general redshift array.
- `nk::Int=500`: number of nodes in the k-scale array used to compute matter power spectrum grid.
- `nℓ::Int=300`: number of nodes in the multipoles array.
- `using_As::Bool=false`: `True` if using the `As` parameter.
- `tk_mode::String=:EisHu` : choice of transfer function.
- `Dz_mode::String=:RK2` : choice of method to compute the linear growth factor.
- `Pk_mode::String=:linear` : choice of method to apply non-linear corrections to the matter power spectrum.

Returns:
```
mutable struct Settings
    nz::Int
    nz_chi::Int
    nz_t::Int
    nk::Int
    nℓ::Int

    xs
    zs
    zs_t
    ks
    ℓs
    logk
    dlogk

    using_As::Bool

    cosmo_type::DataType
    tk_mode::Symbol
    Dz_mode::Symbol
    Pk_mode::Symbol
end        
``` 
"""
mutable struct Settings
    nz::Int
    nz_chi::Int
    nz_t::Int
    nk::Int
    nℓ::Int

    xs
    zs
    zs_chi
    zs_t
    ks
    ℓs
    logk
    dlogk

    using_As::Bool
    tk_mode::Symbol
    Dz_mode::Symbol
    Pk_mode::Symbol
end

function Settings(;kwargs...)
    nz = get(kwargs, :nz, 300)
    nz_chi = get(kwargs, :nz_chi, 1000)
    nz_t = get(kwargs, :nz_t, 350)
    nk = get(kwargs, :nk, 500)
    nℓ = get(kwargs, :nℓ, 300)

    xs = LinRange(0, log(1+1100), nz)
    zs = @.(exp(xs) - 1)
    zs_chi = 10 .^ Vector(LinRange(-3, log10(1100), nz_chi))
    zs_t = range(0.00001, stop=3.0, length=nz_t)
    logk = range(log(0.0001), stop=log(100.0), length=nk)
    ks = exp.(logk)
    dlogk = log(ks[2]/ks[1])
    ℓs = range(0, stop=2000, length=nℓ)

    using_As = get(kwargs, :using_As, false)
    tk_mode = get(kwargs, :tk_mode, :EisHu)
    Dz_mode = get(kwargs, :Dz_mode, :RK2)
    Pk_mode = get(kwargs, :Pk_mode, :linear)

    if using_As && (tk_mode == :EisHu)
        @warn "Using As with the EisensteinHu transfer function is not possible."
        @warn "Using σ8 instead."
        using_As = false
    end   

    return Settings(nz, nz_chi, nz_t, nk, nℓ,
        xs, zs, zs_chi, zs_t, ks, ℓs, logk,  dlogk,
        using_As,
        tk_mode, Dz_mode, Pk_mode)
end

"""
    CosmoPar(kwargs...)

Cosmology parameters structure constructor.  

Kwargs:

- `Ωm::Dual=0.30` : cosmological matter density. 
- `Ωb::Dual=0.05` : cosmological baryonic density.
- `h::Dual=0.70` : reduced Hubble parameter.
- `ns::Dual=0.96` : Harrison-Zeldovich spectral index.
- `As::Dual=2.097e-9` : Harrison-Zeldovich spectral amplitude.
- `σ8::Dual=0.81`: variance of the matter density field in a sphere of 8 Mpc.
- `Y_p::Dual=0.24`: primordial helium fraction.
- `N_ν::Dual=3.046`: effective number of relativisic species (PDG25 value).
- `Σm_ν::Dual=0.0`: sum of neutrino masses (eV), Planck 15 default ΛCDM value.
- `θCMB::Dual=2.725/2.7`: CMB temperature over 2.7.
- `Ωg::Dual=2.38163816E-5*θCMB^4/h^2`: cosmological density of relativistic species.
- `Ωr::Dual=Ωg*(1.0 + N_ν * (7.0/8.0) * (4.0/11.0)^(4.0/3.0))`: cosmological radiation density.

Returns:

```
mutable struct CosmoPar{T}
    Ωm::T
    Ωb::T
    h::T
    ns::T
    As::T
    σ8::T
    θCMB::T
    Y_p::T
    N_ν::T
    Σm_ν::T
    Ωg::T
    Ωr::T
    Ωc::T
    ΩΛ::T
end     
``` 
"""
mutable struct CosmoPar{T}
    Ωm::T
    Ωb::T
    h::T
    ns::T
    As::T
    σ8::T
    θCMB::T
    Y_p::T
    N_ν::T
    Σm_ν::T
    Ωg::T
    Ωr::T
    Ωc::T
    ΩΛ::T
end

function CosmoPar(;kwargs...)
    kwargs = Dict(kwargs)

    Ωm = get(kwargs, :Ωm, 0.3)
    Ωb = get(kwargs, :Ωb, 0.05)
    h = get(kwargs, :h, 0.67)
    ns = get(kwargs, :ns, 0.96)
    As = get(kwargs, :As, 2.097e-9)
    σ8 = get(kwargs, :σ8, 0.81)
    cosmo_type = eltype([Ωm, Ωb, h, ns, As, σ8])

    Y_p = get(kwargs, :Y_p, 0.24)  # primordial helium fraction
    N_ν = get(kwargs, :N_ν, 3.046) # effective number of relativisic species (PDG25 value)
    Σm_ν = get(kwargs, :Σm_ν, 0.0) # sum of neutrino masses (eV), Planck 15 default ΛCDM value
    θCMB = get(kwargs, :θCMB, 2.725/2.7)
    prefac = 2.38163816E-5 # This is 4*sigma_SB*(2.7 K)^4/rho_crit(h=1)
    f_rel = 1.0 + N_ν * (7.0/8.0) * (4.0/11.0)^(4.0/3.0)
    Ωg = get(kwargs, :Ωr, prefac*θCMB^4/h^2)
    Ωr = get(kwargs, :Ωr, Ωg*f_rel)
    Ωc = Ωm-Ωb
    ΩΛ = 1-Ωm-Ωr
    return CosmoPar{cosmo_type}(Ωm, Ωb, h, ns, As, σ8,
        θCMB, Y_p, N_ν, Σm_ν,
        Ωg, Ωr, Ωc, ΩΛ)
end


function _get_cosmo_type(x::CosmoPar{T}) where{T}
    return T
end

mutable struct Cosmology
    settings::Settings
    cpar::CosmoPar
    cosmo_type::DataType
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
    t_of_z::AbstractInterpolation
    chi_max
    chi_LSS
    Dz::AbstractInterpolation
    fs8z::AbstractInterpolation
    PkLz0::AbstractInterpolation
    Pk::AbstractInterpolation
end

"""
    Cosmology(cpar::CosmoPar, settings::Settings)

Base cosmology structure constructor.

Calculates the LCDM expansion history based on the different \
species densities provided in `CosmoPar`.

The comoving distance is then calculated integrating the \
expansion history. 

Depending on the choice of transfer function in the settings, \
the primordial power spectrum is calculated using: 
- `tk_mode = :EisHu` : the Eisenstein & Hu formula (arXiv:astro-ph/9710252)
- `tk_mode = :EmuPk` : the Mootoovaloo et al 2021 emulator `EmuPk` (arXiv:2105.02256v2) 

Depending on the choice of power spectrum mode in the settings, \
the matter power spectrum is either: 
- `Pk_mode = :linear` : the linear matter power spectrum.
- `Pk_mode = :halofit` : the Halofit non-linear matter power spectrum (arXiv:astro-ph/0207664).

Arguments:
- `Settings::MutableStructure` : cosmology constructure settings. 
- `CosmoPar::Structure` : cosmological parameters.

Returns:

```
mutable struct Cosmology
    settings::Settings
    cpar::CosmoPar
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
    t_of_z::AbstractInterpolation
    chi_max
    chi_LSS
    Dz::AbstractInterpolation
    fs8z::AbstractInterpolation
    PkLz0::AbstractInterpolation
    Pk::AbstractInterpolation
end     
``` 
"""
function Cosmology(cpar::CosmoPar, settings::Settings; kwargs...)
    # Load settings
    cosmo_type = _get_cosmo_type(cpar)
    zs_chi, nz_chi = settings.zs_chi, settings.nz_chi
    zs, nz = settings.zs, settings.nz
    logk, nk = settings.logk, settings.nk
    ks = settings.ks
    dlogk = settings.dlogk
    pk0, pki = lin_Pk0(cpar, settings; kwargs...)
    # Compute redshift-distance relation
    chis = zeros(cosmo_type, nz_chi)
    ts = zeros(cosmo_type, nz_chi)
    for i in 1:nz_chi
        zz = zs_chi[i]
        chis[i] = quadgk(z -> 1.0/Ez(cpar, z), 0.0, zz, rtol=1E-5)[1]
        chis[i] *= CLIGHT_HMPC / cpar.h
        ts[i] = quadgk(z -> 1.0/((1+z)*Ez(cpar, z)), 0.0, zz, rtol=1E-5)[1]
        ts[i] *= cpar.h*(18.356164383561644*10^9)
    end
    # Distance to LSS
    chi_LSS = chis[end]
    # OPT: tolerances, interpolation method
    chii = linear_interpolation(zs_chi, Vector(chis), extrapolation_bc=Line())
    ti = linear_interpolation(zs_chi, Vector(ts), extrapolation_bc=Line())
    zi = linear_interpolation(Vector(chis), Vector(zs_chi), extrapolation_bc=Line())
    Dzs, Dzi, fs8zi = get_growth(cpar, settings; kwargs...)

    if settings.Pk_mode == :linear
        Pks = pk0 * (Dzs.^2)'
        Pki = linear_interpolation((logk, zs), log.(Pks);
                                   extrapolation_bc=Line())
    elseif settings.Pk_mode == :Halofit
        Pki = get_PKnonlin(cpar, zs, ks, pk0, Dzs)
    else 
        @error("Pk mode not implemented")
    end
    return Cosmology(
        settings,
        cpar,
        cosmo_type,
        chii,
        zi,
        ti,
        chis[end],
        chi_LSS,
        Dzi,
        fs8zi,
        pki,
        Pki)
end

"""
    Cosmology(;kwargs...)

Short form to call `Cosmology(cpar::CosmoPar, settings::Settings)`.
`kwargs` are passed to the constructors of `CosmoPar` and `Settings`.
Returns:

- `Cosmology` : cosmology structure.

"""
function Cosmology(;kwargs...)
    kwargs=Dict(kwargs)
    if :As ∈ keys(kwargs)
        using_As = true
    else
        using_As = false
    end     
    cpar = CosmoPar(;kwargs...)
    settings = Settings(; using_As=using_As, kwargs...)                  
    return Cosmology(cpar, settings)
end

function Ez(cpar::CosmoPar, z)
    E2 = @. (cpar.Ωm*(1+z)^3+cpar.Ωr*(1+z)^4+cpar.ΩΛ)
    return sqrt.(E2)
end

"""
    chi_to_z(cosmo::Cosmology, chi)

Given a `Cosmology` instance, converts from comoving distance to redshift.  

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `chi::Dual` : comoving distance

Returns:
- `z::Dual` : redshift

"""
function chi_to_z(cosmo::Cosmology, chi)
    return cosmo.z_of_chi(chi)
end

"""
    z_to_t(cosmo::Cosmology, z)

Given a `Cosmology` instance, converts from redshift to years.  

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `t::Dual` : years

"""
function z_to_t(cosmo::Cosmology, z)
    return cosmo.t_of_z(z)
end

"""
    Ez(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the expansion rate (H(z)/H0). 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Ez::Dual` : expansion rate 

"""
Ez(cosmo::Cosmology, z) = Ez(cosmo.cpar, z)

"""
    Hmpc(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the expansion history (H(z)) in Mpc. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Hmpc::Dual` : expansion rate 

"""
Hmpc(cosmo::Cosmology, z) = cosmo.cpar.h*Ez(cosmo, z)/CLIGHT_HMPC

"""
    comoving_radial_distance(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the comoving radial distance. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Chi::Dual` : comoving radial distance

"""
comoving_radial_distance(cosmo::Cosmology, z) = cosmo.chi(z)

"""
    growth_rate(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns growth rate. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `f::Dual` : f

"""
growth_rate(cosmo::Cosmology, z) = cosmo.fs8z(z) ./ (cosmo.cpar.σ8*cosmo.Dz(z)/cosmo.Dz(0))

"""
    fs8(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns fs8. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `fs8::Dual` : fs8

"""
fs8(cosmo::Cosmology, z) =  cosmo.fs8z(z)

"""
    growth_factor(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the growth factor (D(z) = log(δ)). 

Arguments:
- `cosmo::Cosmology` : cosmological structure
- `z::Dual` : redshift

Returns:
- `Dz::Dual` : comoving radial distance

"""
growth_factor(cosmo::Cosmology, z) = cosmo.Dz(z)

"""
    sigma8(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns s8. 

Arguments:
- `cosmo::Cosmology` : cosmological structure
- `z::Dual` : redshift

Returns:
- `s8::Dual` : comoving radial distance

"""
sigma8(cosmo::Cosmology, z) = cosmo.cpar.σ8*cosmo.Dz(z)/cosmo.Dz(0)

"""
    nonlin_Pk(cosmo::Cosmology, k, z)

Given a `Cosmology` instance, it returns the non-linear matter power spectrum (P(k,z)) \
using the Halofit fitting formula (arXiv:astro-ph/0207664). 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `k::Dual` : scale
- `z::Dual` : redshift

Returns:
- `Pk::Dual` : non-linear matter power spectrum

"""
function nonlin_Pk(cosmo::Cosmology, k, z)
    return @. exp(cosmo.Pk(log(k), z))
end

"""
    lin_Pk(cosmo::Cosmology, k, z)

Given a `Cosmology` instance, it returns the linear matter power spectrum (P(k,z))

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `k::Dual` : scale
- `z::Dual` : redshift

Returns:
- `Pk::Dual` : linear matter power spectrum

"""
function lin_Pk(cosmo::Cosmology, k, z)
    pk0 = @. exp(cosmo.PkLz0(log(k)))
    Dz2 = cosmo.Dz(z)^2
    return pk0 .* Dz2
end