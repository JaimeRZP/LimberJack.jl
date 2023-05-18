# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

function _w_tophat(x::Real)
    x2 = x^2

    if x < 0.1
        w = 1. + x2*(-1.0/10.0 + x2*(1.0/280.0 +
            x2*(-1.0/15120.0 + x2*(1.0/1330560.0 +
            x2* (-1.0/172972800.0)))));
    else
        w = 3 * (sin(x)-x*cos(x))/(x2*x)
    end
    return w
end    

function σR2(ks, pk, dlogk, R)
    x = ks .* R
    wk = _w_tophat.(x)
    integrand = @. pk * wk^2 * ks^3
    σ = integrate(log.(ks), integrand, SimpsonEven())/(2*pi^2)
    return σ
end

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
    cosmo_type::DataType
    nz::Int
    nz_pk::Int
    nz_t::Int
    nk::Int
    tk_mode::String
    Pk_mode::String
    emul_path::String
    custom_Dz
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
struct CosmoPar{T}
    Ωm::T
    Ωb::T
    h::T
    n_s::T
    σ8::T
    θCMB::T
    Ωr::T
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
CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB) = begin
    # This is 4*sigma_SB*(2.7 K)^4/rho_crit(h=1)
    prefac = 2.38163816E-5
    Neff = 3.046
    f_rel = 1.0 + Neff * (7.0/8.0) * (4.0/11.0)^(4.0/3.0)
    Ωr = prefac*f_rel*θCMB^4/h^2
    ΩΛ = 1-Ωm-Ωr
    CosmoPar{Real}(Ωm, Ωb, h, n_s, σ8, θCMB, Ωr, ΩΛ)
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
    cosmo::CosmoPar
    # Power spectrum
    ks::Array
    pk0::Array
    logk
    dlogk
    # Redshift and background
    zs::Array
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
Cosmology(cpar::CosmoPar, settings::Settings) = begin
    # Load settings
    cosmo_type = settings.cosmo_type
    nk = settings.nk
    nz = settings.nz
    nz_pk = settings.nz_pk
    zs_pk = LinRange(0., 3.0, nz_pk)
    zs = range(0., stop=3.0, length=nz)
    # Compute linear power spectrum at z=0.
    logk = range(log(0.0001), stop=log(100.0), length=nk)
    ks = exp.(logk)
    dlogk = log(ks[2]/ks[1])
    if settings.tk_mode == "emulator"
        lks_emul, pk0_emul = get_emulated_log_pk0(cpar, settings)
        pki_emul = cubic_spline_interpolation(lks_emul, log.(pk0_emul),
                                        extrapolation_bc=Line())
        pk0 = exp.(pki_emul(logk))
    elseif settings.tk_mode == "EisHu"
        tk = TkEisHu(cpar, ks./ cpar.h)
        pk0 = @. ks^cpar.n_s * tk
    elseif settings.tk_mode == "BBKS"
        tk = TkBBKS(cpar, ks)
        pk0 = @. ks^cpar.n_s * tk
     else
        print("Transfer function not implemented")
    end
    #Renormalize Pk
    σ8_2_here = _σR2(ks, pk0, dlogk, 8.0/cpar.h)
    norm = cpar.σ8^2 / σ8_2_here
    pk0 *= norm
    # OPT: interpolation method
    pki = cubic_spline_interpolation(logk, log.(pk0);
                                     extrapolation_bc=Line())
    # Compute redshift-distance relation
    norm = CLIGHT_HMPC / cpar.h
    chis = zeros(cosmo_type, nz)
    for i in 1:nz
        zz = zs[i]
        chis[i] = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
    end
    # OPT: tolerances, interpolation method
    chii = cubic_spline_interpolation(zs, chis, extrapolation_bc=Line())
    zi = linear_interpolation(chis, zs, extrapolation_bc=Line())
    # Distance to LSS
    chi_LSS = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, 1100., rtol=1E-5)[1] * norm

    if settings.custom_Dz == nothing
        # ODE solution for growth factor
        x_Dz = LinRange(0, log(1+1100), nz_pk)
        dx_Dz = x_Dz[2]-x_Dz[1]
        z_Dz = @.(exp(x_Dz) - 1)
        a_Dz = @.(1/(1+z_Dz))
        aa = reverse(a_Dz)
        e = _Ez(cpar, z_Dz)
        ee = reverse(e)
        
        dd = zeros(settings.cosmo_type, nz_pk)
        yy = zeros(settings.cosmo_type, nz_pk)
        dd[1] = aa[1]
        yy[1] = aa[1]^3*ee[end]
        
        for i in 1:(nz_pk-1)
            A0 = -1.5 * cpar.Ωm / (aa[i]*ee[i])
            B0 = -1. / (aa[i]^2*ee[i])
            A1 = -1.5 * cpar.Ωm / (aa[i+1]*ee[i+1])
            B1 = -1. / (aa[i+1]^2*ee[i+1])
            yy[i+1] = (1+0.5*dx_Dz^2*A0*B0)*yy[i] + 0.5*(A0+A1)*dx_Dz*dd[i]
            dd[i+1] = 0.5*(B0+B1)*dx_Dz*yy[i] + (1+0.5*dx_Dz^2*A0*B0)*dd[i]
        end
        
        y = reverse(yy)
        d = reverse(dd)
        
        Dzi = linear_interpolation(z_Dz, d./d[1], extrapolation_bc=Line())
        fs8zi = linear_interpolation(z_Dz, -cpar.σ8 .* y./ (a_Dz.^2 .*e.*d[1]),
                                     extrapolation_bc=Line())
        Dzs = Dzi(zs_pk)
    else
        zs_c, Dzs_c = settings.custom_Dz
        d = zs_c[2]-zs_c[1]

        Dzi = cubic_spline_interpolation(zs_c, Dzs_c, extrapolation_bc=Line())
        dDzs_mid = (Dzs_c[2:end].-Dzs_c[1:end-1])/d
        zs_mid = (zs_c[2:end].+zs_c[1:end-1])./2
        dDzi = linear_interpolation(zs_mid, dDzs_mid, extrapolation_bc=Line())
        dDzs_c = dDzi(zs_c)
        fs8zi = cubic_spline_interpolation(zs_c, -cpar.σ8 .* (1 .+ zs_c) .* dDzs_c,
                                           extrapolation_bc=Line())

        Dzs = Dzi(zs_pk)
    end

    if settings.Pk_mode == "linear"
        Pks = [@. pk*Dzs^2 for pk in pk0]
        Pks = reduce(vcat, transpose.(Pks))
        Pk = linear_interpolation((logk, zs_pk), log.(Pks);
                                   extrapolation_bc=Line())
    elseif settings.Pk_mode == "Halofit"
        Pk = get_PKnonlin(cpar, zs_pk, ks, pk0, Dzs, cosmo_type)
    else 
        Pks = [@. pk*Dzs^2 for pk in pk0]
        Pks = reduce(vcat, transpose.(Pks))
        Pk = linear_interpolation((logk, zs_pk), log.(Pks);
                                   extrapolation_bc=Line())
        print("Pk mode not implemented. Using linear Pk.")
    end
    Cosmology(settings, cpar, ks, pk0, logk, dlogk,
              collect(zs), chii, zi, chis[end],
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
Cosmology(Ωm, Ωb, h, n_s, σ8; 
          θCMB=2.725/2.7, nk=500, nz=300, nz_pk=70,  nz_t=400,
          tk_mode="BBKS", Pk_mode="linear",
          emul_path= "../emulator/files.npz",
          custom_Dz=nothing) = begin
    
    cosmo_type = eltype([Ωm, Ωb, h, n_s, σ8, θCMB])
    if custom_Dz != nothing
        _, Dz = custom_Dz
        Dz_type = eltype(Dz)
        if !(Dz_type <: Float64)
            cosmo_type = Dz_type
        end
    end
    
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)
    settings = Settings(cosmo_type, nz, nz_pk, nz_t, nk,
                        tk_mode, Pk_mode, emul_path, custom_Dz)
    Cosmology(cpar, settings)
end

"""
    Cosmology()

Calls the simple Cosmology structure constructor for the \
paramters `[0.30, 0.05, 0.67, 0.96, 0.81]`.

Returns:

- `Cosmology` : cosmology structure.

"""
Cosmology() = Cosmology(0.30, 0.045, 0.67, 0.96, 0.81)

function σR2(cosmo::Cosmology, R)
    return _σR2(cosmo.ks, cosmo.pk0, cosmo.dlogk, R)
end

"""
    TkBBKS(cosmo::CosmoPar, k)

Computes the primordial power spectrum using the BBKS formula (https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B).  

Arguments:

- `cosmo::CosmoPar` : cosmological parameters structure 
- `k::Vector{Dual}` : scales array

Returns:

- `Tk::Vector{Dual}` : transfer function.

"""
function TkBBKS(cosmo::CosmoPar, k)
    q = @. (cosmo.θCMB^2 * k/(cosmo.Ωm * cosmo.h^2 * exp(-cosmo.Ωb*(1+sqrt(2*cosmo.h)/cosmo.Ωm))))
    return (@. (log(1+2.34q)/(2.34q))^2/sqrt(1+3.89q+(16.1q)^2+(5.46q)^3+(6.71q)^4))
end

function _T0(keq, k, ac, bc)
    q = @. (k/(13.41*keq))
    C = @. ((14.2/ac) + (386/(1+69.9*q^1.08)))
    T0 = @.(log(ℯ+1.8*bc*q)/(log(ℯ+1.8*bc*q)+C*q^2))
    return T0
end 

"""
    TkEisHu(cosmo::CosmoPar, k)

Computes the primordial power spectrum using the Einsentein & Hu formula (arXiv:astro-ph/9710252).  

Arguments:
- `cosmo::CosmoPar` : cosmological parameters structure
- `k::Vector{Dual}` : scales array

Returns:
- `Tk::Vector{Dual}` : transfer function.

"""
function TkEisHu(cosmo::CosmoPar, k)
    Ωc = cosmo.Ωm-cosmo.Ωb
    wm=cosmo.Ωm*cosmo.h^2
    wb=cosmo.Ωb*cosmo.h^2

    # Scales
    # k_eq
    keq = (7.46*10^-2)*wm/(cosmo.h * cosmo.θCMB^2)
    # z_eq
    zeq = (2.5*10^4)*wm*(cosmo.θCMB^-4)
    # z_drag
    b1 = 0.313*(wm^-0.419)*(1+0.607*wm^0.674)
    b2 = 0.238*wm^0.223
    zd = 1291*((wm^0.251)/(1+0.659*wm^0.828))*(1+b1*wb^b2)
    # k_Silk
    ksilk = 1.6*wb^0.52*wm^0.73*(1+(10.4*wm)^-0.95)/cosmo.h
    # r_s
    R_prefac = 31.5*wb*(cosmo.θCMB^-4)
    Rd = R_prefac*((zd+1)/10^3)^-1
    Rdm1 = R_prefac*(zd/10^3)^-1
    Req = R_prefac*(zeq/10^3)^-1
    rs = sqrt(1+Rd)+sqrt(Rd+Req)
    rs /= (1+sqrt(Req))
    rs =  (log(rs))
    rs *= (2/(3*keq))*sqrt(6/Req)

    # Tc
    q = @.(k/(13.41*keq))
    a1 = (46.9*wm)^0.670*(1+(32.1*wm)^-0.532)
    a2 = (12.0*wm)^0.424*(1+(45.0*wm)^-0.582)
    ac = (a1^(-cosmo.Ωb/cosmo.Ωm))*(a2^(-(cosmo.Ωb/cosmo.Ωm)^3))
    b1 = 0.944*(1+(458*wm)^-0.708)^-1
    b2 = (0.395*wm)^(-0.0266)
    bc = (1+b1*((Ωc/cosmo.Ωm)^b2-1))^-1
    f = @.(1/(1+(k*rs/5.4)^4))
    Tc1 = f.*_T0(keq, k, 1, bc)
    Tc2 = (1 .-f).*_T0(keq, k, ac, bc)
    Tc = Tc1 .+ Tc2

    # Tb
    y = (1+zeq)/(1+zd)
    Gy = y*(-6*sqrt(1+y)+(2+3y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)))
    ab = 2.07*keq*rs*(1+Rdm1)^(-3/4)*Gy
    bb =  0.5+(cosmo.Ωb/cosmo.Ωm)+(3-2*cosmo.Ωb/cosmo.Ωm)*sqrt((17.2*wm)^2+1)
    bnode = 8.41*(wm)^0.435
    ss = rs./(1 .+(bnode./(k.*rs)).^3).^(1/3)
    Tb1 = _T0(keq, k, 1, 1)./(1 .+(k.*rs/5.2).^2)
    Tb2 = (ab./(1 .+(bb./(k.*rs)).^3)).*exp.(-(k/ksilk).^ 1.4)
    Tb = (Tb1.+Tb2).*sin.(k.*ss)./(k.*ss)

    Tk = (cosmo.Ωb/cosmo.Ωm).*Tb.+(Ωc/cosmo.Ωm).*Tc
    return Tk.^2
end

function _Ez(cosmo::CosmoPar, z)
    E2 = @. (cosmo.Ωm*(1+z)^3+cosmo.Ωr*(1+z)^4+cosmo.ΩΛ)
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
    closest_chi, idx = findmin(abs(chi-cosmo.chis))
    if closest_chi >= chi
        idx += -1
    end

    return z
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
Ez(cosmo::Cosmology, z) = _Ez(cosmo.cosmo, z)

"""
    Hmpc(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the expansion history (H(z)) in Mpc. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Hmpc::Dual` : expansion rate 

"""
Hmpc(cosmo::Cosmology, z) = cosmo.cosmo.h*Ez(cosmo, z)/CLIGHT_HMPC

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
growth_rate(cosmo::Cosmology, z) = cosmo.fs8z(z) ./ (cosmo.cosmo.σ8*cosmo.Dz(z)/cosmo.Dz(0))

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
sigma8(cosmo::Cosmology, z) = cosmo.cosmo.σ8*cosmo.Dz(z)/cosmo.Dz(0)

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
