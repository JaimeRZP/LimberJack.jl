abstract type Tracer end

"""
    NumberCountsTracer(warr, chis, wint, b, lpre)
Number counts tracer structure. 
Arguments:
- `wint::Interpolation` : interpolation of the radial kernel over comoving distance.
- `F::Function` : prefactor.
Returns:
- `NumberCountsTracer::NumberCountsTracer` : Number counts tracer structure.
"""
struct NumberCountsTracer <: Tracer
    wint::AbstractInterpolation
    F::Function
end

"""
    NumberCountsTracer(cosmo::Cosmology, z_n, nz; kwargs...)
Number counts tracer structure constructor. 
Arguments:
- `cosmo::Cosmology` : cosmology structure. 
- `z_n::Vector{Dual}` : redshift array.
- `nz::Interpolation` : distribution of sources over redshift.
Kwargs:
- `b::Dual = 1` : matter-galaxy bias. 
 
Returns:
- `NumberCountsTracer::NumberCountsTracer` : Number counts tracer structure.
"""
NumberCountsTracer(cosmo::Cosmology, z_n, nz; 
    b=1.0, res=1000, nz_interpolation="linear") = begin
    z_w, nz_w = nz_interpolate(z_n, nz, res)
    nz_norm = integrate(z_w, nz_w, SimpsonEven())
    
    chi = cosmo.chi(z_w)
    hz = Hmpc(cosmo, z_w)
    
    w_arr = @. (nz_w*hz/nz_norm)
    wint = linear_interpolation(chi, b .* w_arr, extrapolation_bc=Line())
    F::Function = ℓ -> 1
    NumberCountsTracer(wint, F)
end

"""
    WeakLensingTracer(cosmo::Cosmology, z_n, nz; kwargs...)
Weak lensing tracer structure constructor. 
Arguments:
- `cosmo::Cosmology` : cosmology structure. 
- `z_n::Vector{Dual}` : redshift array.
- `nz::Interpolation` : distribution of sources over redshift.
Kwargs:
- `mb::Dual = 1` : multiplicative bias. 
- `IA_params::Vector{Dual} = [A_IA, alpha_IA]`: instrinsic aligment parameters.
 
Returns:
- `WeakLensingTracer::WeakLensingTracer` : Weak lensing tracer structure.
"""
struct WeakLensingTracer <: Tracer
    wint::AbstractInterpolation
    F::Function
end

WeakLensingTracer(cosmo::Cosmology, z_n, nz;
    IA_params = [0.0, 0.0], m=0.0,
    res=350, nz_interpolate="linear") = begin
    cosmo_type = cosmo.settings.cosmo_type
    z_w, nz_w = nz_interpolate(z_n, nz, res)
    res = length(z_w)
    nz_norm = integrate(z_w, nz_w, SimpsonEven())
    chi = cosmo.chi(z_w)

    # Calculate chis at which to precalculate the lensing kernel
    # OPT: perhaps we don't need to sample the lensing kernel
    #      at all zs.
    # Calculate integral at each chi
    w_itg(chii) = @.(nz_w*(1-chii/chi))
    w_arr = zeros(cosmo_type, res)
    @inbounds for i in 1:res-3
        w_arr[i] = integrate(z_w[i:res], w_itg(chi[i])[i:res], SimpsonEven())
    end
    
    # Normalize
    H0 = cosmo.cpar.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cpar.Ωm*H0^2
    # Fix first element
    chi[1] = 0.0
    w_arr = @. w_arr * chi * lens_prefac * (1+z_w) / nz_norm
    
    if IA_params != [0.0, 0.0]
        hz = Hmpc(cosmo, z_w)
        As = get_IA(cosmo, z_w,IA_params)
        corr =  @. As * (nz_w * hz / nz_norm)
        w_arr = @. w_arr - corr
    end

    # Interpolate
    b = m+1.0
    wint = linear_interpolation(chi, b.*w_arr, extrapolation_bc=Line())
    F::Function = ℓ -> @.(sqrt((ℓ+2)*(ℓ+1)*ℓ*(ℓ-1))/(ℓ+0.5)^2)
    WeakLensingTracer(wint, F)
end

"""
    CMBLensingTracer(warr, chis, wint, lpre)
CMB lensing tracer structure. 
Arguments:
- `wint::Interpolation` : interpolation of the radial kernel over comoving distance.
- `F::Function` : prefactor.
Returns:
- `CMBLensingTracer::CMBLensingTracer` : CMB lensing tracer structure.
"""
struct CMBLensingTracer <: Tracer
    wint::AbstractInterpolation
    F::Function
end

"""
    CMBLensingTracer(cosmo::Cosmology; nchi=100)
CMB lensing tracer structure. 
Arguments:
- `cosmo::Cosmology` : cosmology structure.
Returns:
- `CMBLensingTracer::CMBLensingTracer` : CMB lensing tracer structure.
"""
CMBLensingTracer(cosmo::Cosmology; res=350) = begin
    # chi array
    chis = range(0.0, stop=cosmo.chi_max, length=res)
    zs = cosmo.z_of_chi(chis)
    # Prefactor
    H0 = cosmo.cpar.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cpar.Ωm*H0^2
    # Kernel
    w_arr = @. lens_prefac*chis*(1-chis/cosmo.chi_LSS)*(1+zs)

    # Interpolate
    wint = linear_interpolation(chis, w_arr, extrapolation_bc=0.0)
    F::Function = ℓ -> @.(((ℓ+1)*ℓ)/(ℓ+0.5)^2) 
    CMBLensingTracer(wint, F)
end

"""
    get_IA(cosmo::Cosmology, zs, IA_params)
CMB lensing tracer structure. 
Arguments:
- `cosmo::Cosmology` : cosmology structure.
- `zs::Vector{Dual}` : redshift array.
- `IA_params::Vector{Dual}` : Intrinsic aligment parameters.
Returns:
- `IA_corr::Vector{Dual}` : Intrinsic aligment correction to radial kernel.
"""
function get_IA(cosmo::Cosmology, zs, IA_params)
    A_IA = IA_params[1]
    alpha_IA = IA_params[2]
    return @. A_IA*((1 + zs)/1.62)^alpha_IA * (0.0134 * cosmo.cpar.Ωm / cosmo.Dz(zs))
end

function nz_interpolate(z, nz; mode="linear", res=nothing)
    if mode!="none"
        if mode=="linear"
            nz_int = linear_interpolation(z, nz, extrapolation_bc=Line())
        if mode=="cubic"
            dz = mean(z[2:end] - z[1:end-1])
            z_range = z[1]:dz:z[end]
            nz_int = cubic_spline_interpolation(z_range, nz)
        end
        if res==nothing
            dzz = dz/10
        else
            L = z[end] - z[1]
            dzz = L/res
        end
        zz_range = z[1]:dzz:z[end]
        nzz = nz_int(zz_range)
        return zz_range, nzz
    else
        return z, nz
    end
end
