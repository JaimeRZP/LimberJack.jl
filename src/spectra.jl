"""
    Cℓintegrand(cosmo::Cosmology, t1::Tracer, t2::Tracer, logk, ℓ)
Returns the integrand of the angular power spectrum. 
Arguments:
- `cosmo::Cosmology` : cosmology structure.
- `t1::Tracer` : tracer structure.
- `t2::Tracer` : tracer structure.
- `logk::Vector{Float}` : log scale array.
- `ℓ::Float` : multipole.
Returns:
- `integrand::Vector{Real}` : integrand of the angular power spectrum.
"""
function Cℓintegrand(cosmo::Cosmology,
                     t1::AbstractInterpolation,
                     t2::AbstractInterpolation,
                     ℓ::Number)
    sett = cosmo.settings
    chis = zeros(sett.cosmo_type, sett.nk)
    chis[1:sett.nk] = (ℓ+0.5) ./ sett.ks
    chis .*= (chis .< cosmo.chi_max)
    z = cosmo.z_of_chi(chis)

    w1 = t1(chis)
    w2 = t2(chis)
    pk = nonlin_Pk(cosmo, sett.ks, z)

    return @. (sett.ks*w1*w2*pk)
end

"""
    angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs)
Returns the angular power spectrum. 
Arguments:
- `cosmo::Cosmology` : cosmology structure.
- `t1::Tracer` : tracer structure.
- `t2::Tracer` : tracer structure.
- `ℓs::Vector{Float}` : multipole array.
Returns:
- `Cℓs::Vector{Real}` : angular power spectrum.
"""
function angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs::Vector)
    # OPT: we are not optimizing the limits of integration
    sett = cosmo.settings
    Cℓs = [integrate(sett.logk, Cℓintegrand(cosmo, t1.wint, t2.wint, ℓ)/(ℓ+0.5), SimpsonEven()) for ℓ in ℓs]
    return t1.F(ℓs) .* t2.F(ℓs) .* Cℓs
end