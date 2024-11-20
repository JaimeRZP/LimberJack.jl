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
NumberCountsTracer(cosmo::Cosmology, z, nz; b=1.0) = begin
    sel = @. (z > 0.0)
    z = z[sel]
    nz = nz[sel]
    nz_norm = integrate(z, nz, SimpsonEven())  
    chi = cosmo.chi(z)
    hz = Hmpc(cosmo, z)
    w_arr = @. (nz*hz/nz_norm)
    wint = linear_interpolation(chi, b .* w_arr, extrapolation_bc=0.0)
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

WeakLensingTracer(cosmo::Cosmology, z, nz;
    A_IA=0.0, alpha_IA=0.0, m=0.0) = begin
    sel = @. (z > 0.0)
    z = z[sel]
    nz = nz[sel]
    cosmo_type = cosmo.settings.cosmo_type
    res = length(z)
    nz_norm = integrate(z, nz, SimpsonEven())
    chi = cosmo.chi(z)
    # Compute kernels
    w_itg(chii) = @.(nz*(1-chii/chi))
    w_arr = zeros(cosmo_type, res)
    @inbounds for i in 1:res-3
        w_arr[i] = integrate(z[i:res], w_itg(chi[i])[i:res], SimpsonEven())
    end
    # Normalize
    H0 = cosmo.cpar.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cpar.Ωm*H0^2
    chi[1] = 0.0
    w_arr = @. w_arr * chi * lens_prefac * (1+z) / nz_norm
    # IA correction
    hz = Hmpc(cosmo, z)
    As = @. A_IA*((1 + z)/1.62)^alpha_IA * (0.0134 * cosmo.cpar.Ωm / cosmo.Dz(z))
    corr =  @. As * (nz * hz / nz_norm)
    w_arr_ia = @. w_arr - corr
    # Interpolate
    b = m+1.0
    wint = linear_interpolation(chi, b.*w_arr_ia, extrapolation_bc=0.0)
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
    z = cosmo.z_of_chi(chis)
    # Prefactor
    H0 = cosmo.cpar.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cpar.Ωm*H0^2
    # Kernel
    w_arr = @. lens_prefac*chis*(1-chis/cosmo.chi_LSS)*(1+z)
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

function nz_interpolate(z, nz, res; mode="linear")
    if mode!="none"
        if mode=="linear"
            nz_int = linear_interpolation(z, nz;
                extrapolation_bc=Line())
        end
        if mode=="cubic"
            dz = mean(z[2:end] - z[1:end-1])
            z_range = z[1]:dz:z[end]
            nz_int = cubic_spline_interpolation(z_range, nz;
                extrapolation_bc=Line())
        end
        zz_range = range(0.00001, stop=z[end], length=res)
        nzz = nz_int(zz_range)
        return zz_range, nzz
    else
        return z, nz
    end
end
