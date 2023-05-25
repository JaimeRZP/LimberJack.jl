"""
    Settings(cosmo_type, nz, nz_pk, nk, tk_mode, pk_mode, custom_Dz)

Cosmology constructor settings structure. 

Arguments:

- `cosmo_type::Type` : type of cosmological parameters. 
- `nz::Int` : number of nodes in the general redshift array.
- `nz_pk::Int` : number of nodes in the redshift array used for matter power spectrum.
- `nk::Int`: number of nodes in the matter power spectrum.
- `tk_mode::String` : choice of transfer function.
- `Pk_mode::String` : choice of matter power spectrum.
- `custom_Dz::Any` : custom growth factor.

Returns:

- `Settings` : cosmology settings.

"""
mutable struct Settings
    nz::Int
    nz_pk::Int
    nz_t::Int
    nk::Int
    nℓ::Int

    zs
    zs_pk
    zs_t
    ks
    ℓs
    logk
    dlogk

    using_As::Bool

    cosmo_type::DataType
    tk_mode::String
    Dz_mode::String
    Pk_mode::String
    emul_files
end

Settings(;kwargs...) = begin
    nz = get(kwargs, :nz, 300)
    nz_pk = get(kwargs, :nz_pk, 70)
    nz_t = get(kwargs, :nz_t, 350)
    nk = get(kwargs, :nk, 500)
    nℓ = get(kwargs, :nℓ, 300)

    zs_pk = range(0., stop=3.0, length=nz_pk)
    zs = range(0.0, stop=3.0, length=nz)
    zs_t = range(0.00001, stop=3.0, length=nz_t)
    logk = range(log(0.0001), stop=log(100.0), length=nk)
    ks = exp.(logk)
    dlogk = log(ks[2]/ks[1])
    ℓs = range(0, stop=2000, length=nℓ)

    using_As = get(kwargs, :using_As, false)

    cosmo_type = get(kwargs, :cosmo_type, Float64)
    tk_mode = get(kwargs, :tk_mode, "EisHu")
    Dz_mode = get(kwargs, :Dz_mode, "RK2")
    Pk_mode = get(kwargs, :Pk_mode, "linear")
    emul_files = get(kwargs, :emul_files, nothing)
    Settings(nz, nz_pk, nz_t, nk, nℓ,
             zs, zs_pk, zs_t, ks, ℓs, logk,  dlogk,
             using_As,
             cosmo_type, tk_mode, Dz_mode, Pk_mode, emul_files)
end
"""
    CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB, Ωr, ΩΛ)

Cosmology parameters structure.  

Arguments:

- `Ωm::Dual` : cosmological matter density. 
- `Ωb::Dual` : cosmological baryonic density.
- `h::Dual` : reduced Hubble parameter.
- `n_s::Dual` : spectral index.
- `σ8::Dual`: variance of the matter density field in a sphere of 8 Mpc.
- `θCMB::Dual` : CMB temperature.
- `Ωr::Dual` : cosmological radiation density.
- `ΩΛ::Dual` : cosmological dark energy density.

Returns:

- `CosmoPar` : cosmology parameters structure.

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

"""
    CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)
Cosmology parameters structure constructor.
Arguments:
- `Ωm::Dual` : cosmological matter density.
- `Ωb::Dual` : cosmological baryonic density.
- `h::Dual` : reduced Hubble parameter.
- `n_s::Dual` : spectral index.
- `σ8::Dual`: variance of the matter density field in a sphere of 8 Mpc.
- `θCMB::Dual` : CMB temperature.
Returns:
- `CosmoPar` : cosmology parameters structure.
"""
CosmoPar(;kwargs...) = begin
    kwargs = Dict(kwargs)

    Ωm = get(kwargs, :Ωm, 0.3)
    Ωb = get(kwargs, :Ωb, 0.05)
    h = get(kwargs, :h, 0.67)
    ns = get(kwargs, :ns, 0.96)
    As = get(kwargs, :As, 2.097e-9)
    σ8 = get(kwargs, :σ8, 0.81)
    cosmo_type = eltype([Ωm, Ωb, h, ns, σ8])

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
    CosmoPar{cosmo_type}(Ωm, Ωb, h, ns, As, σ8,
                         θCMB, Y_p, N_ν, Σm_ν,
                         Ωg, Ωr, Ωc, ΩΛ)
end


function _get_cosmo_type(x::CosmoPar{T}) where{T}
    return T
end

"""
    Cosmology(Settings, CosmoPar,
              ks, pk0, logk, dlogk,
              zs, chi, z_of_chi, chi_max, chi_LSS, Dz, PkLz0, Pk)

Base cosmology structure.  

Arguments:
- `Settings::MutableStructure` : cosmology constructure settings. 
- `CosmoPar::Structure` : cosmological parameters.
- `ks::Dual` : scales array.
- `pk0::Dual`: primordial matter power spectrum.
- `logk::Dual` : log scales array.
- `dlogk::Dual` : increment in log scales.
- `zs::Dual` : redshift array.
- `chi::Dual` : comoving distance array.
- `z_of_chi::Dual` : redshift of comoving distance array.
- `chi_max::Dual` : upper bound of comoving distance array.
- `chi_LSS::Dual` : comoving distance to suface of last scattering.
- `Dz::Dual` : growth factor.
- `PkLz0::Dual` : interpolator of log primordial power spectrum over k-scales.
- `Pk::Dual` : matter power spectrum.

Returns:
- `CosmoPar` : cosmology parameters structure.

"""
struct Cosmology
    settings::Settings
    cpar::CosmoPar
    # Power spectrum
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
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
- `tk_mode = "BBKS"` : the BBKS fitting formula (https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B)
- `tk_mode = "EisHu"` : the Eisenstein & Hu formula (arXiv:astro-ph/9710252)
- `tk_mode = "emulator"` : the Mootoovaloo et al 2021 emulator (arXiv:2105.02256v2)

is `custom_Dz = nothing`, the growth factor is obtained either by solving the Jeans equation. \
Otherwise, provided custom growth factor is used.


Depending on the choice of power spectrum mode in the settings, \
the matter power spectrum is either: 
- `Pk_mode = "linear"` : the linear matter power spectrum.
- `Pk_mode = "halofit"` : the Halofit non-linear matter power spectrum (arXiv:astro-ph/0207664).

Arguments:
- `Settings::MutableStructure` : cosmology constructure settings. 
- `CosmoPar::Structure` : cosmological parameters.

Returns:

- `Cosmology` : cosmology structure.

"""
Cosmology(cpar::CosmoPar, settings::Settings; kwargs...) = begin
    # Load settings
    cosmo_type = settings.cosmo_type
    zs_pk, nz_pk = settings.zs_pk, settings.nz_pk
    zs, nz = settings.zs, settings.nz
    logk, nk = settings.logk, settings.nk
    ks = settings.ks
    dlogk = settings.dlogk
    pk0, pki = lin_Pk0(cpar, settings)
    # Compute redshift-distance relation
    norm = CLIGHT_HMPC / cpar.h
    chis = zeros(cosmo_type, nz)
    for i in 1:nz
        zz = zs[i]
        chis[i] = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
    end
    # OPT: tolerances, interpolation method
    chii = cubic_spline_interpolation(zs, Vector(chis), extrapolation_bc=Line())
    zi = linear_interpolation(chis, Vector(zs), extrapolation_bc=Line())
    # Distance to LSS
    chi_LSS = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, 1100., rtol=1E-5)[1] * norm

    Dzi, fs8zi = get_growth(cpar, settings; kwargs...)
    Dzs = Dzi(zs_pk)

    if settings.Pk_mode == "linear"
        Pks = [@. pk*Dzs^2 for pk in pk0]
        Pks = reduce(vcat, transpose.(Pks))
        Pk = linear_interpolation((logk, zs_pk), log.(Pks);
                                   extrapolation_bc=Line())
    elseif settings.Pk_mode == "Halofit"
        Pk = get_PKnonlin(cpar, zs_pk, ks, pk0, Dzs;
                          cosmo_type=cosmo_type)
    else 
        @error("Pk mode not implemented")
    end
    Cosmology(settings, cpar, chii, zi, chis[end],
              chi_LSS, Dzi, fs8zi, pki, Pk)
end

"""
    Cosmology(Ωm, Ωb, h, n_s, σ8;
              θCMB=2.725/2.7, nk=500, nz=500, nz_pk=100,
              tk_mode="BBKS", Pk_mode="linear", custom_Dz=nothing)

Simple cosmology structure constructor that calls the base constructure.
Fills the `CosmoPar` and `Settings` structure based on the given parameters.

Arguments:

- `Ωm::Dual` : cosmological matter density. 
- `Ωb::Dual` : cosmological baryonic density.
- `h::Dual` : reduced Hubble parameter.
- `n_s::Dual` : spectral index.
- `σ8::Dual`: variance of the matter density field in a sphere of 8 Mpc.

Kwargs:

- `θCMB::Dual` : CMB temperature.
- `nz::Int` : number of nodes in the general redshift array.
- `nz_pk::Int` : number of nodes in the redshift array used for matter power spectrum.
- `nk::Int`: number of nodes in the matter power spectrum.
- `tk_mode::String` : choice of transfer function.
- `Pk_mode::String` : choice of matter power spectrum.
- `custom_Dz::Any` : custom growth factor.

Returns:

- `Cosmology` : cosmology structure.

"""
Cosmology(;kwargs...) = begin

    kwargs=Dict(kwargs)
    if :As ∈ keys(kwargs)
        using_As = true
    else
        using_As = false
    end     
    cpar = CosmoPar(;kwargs...)
    cosmo_type = _get_cosmo_type(cpar)
    settings = Settings(;cosmo_type=cosmo_type,
                         using_As=using_As,
                         kwargs...)
    if using_As && (settings.tk_mode == "BBKS" || settings.tk_mode == "EisensteinHu")
        @warn "Using As with BBKS or EisensteinHu transfer function is not possible."
        @warn "Using σ8 instead."
        using_As = false
    end                     
    Cosmology(cpar, settings)
end

function _Ez(cpar::CosmoPar, z)
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
    Ez(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the expansion rate (H(z)/H0). 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Ez::Dual` : expansion rate 

"""
Ez(cosmo::Cosmology, z) = _Ez(cosmo.cpar, z)

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
