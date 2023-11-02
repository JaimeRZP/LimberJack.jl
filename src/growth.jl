function _dgrowth!(dd, d, cosmo::CosmoPar, a)
    ez = Ez(cosmo, 1.0/a-1.0)
    dd[1] = d[2] * 1.5 * cosmo.Ωm / (a^2*ez)
    dd[2] = d[1] / (a^3*ez)
end

get_growth(mode::Symbol, cpar::CosmoPar, settings::Settings; kwargs...) = get_growth(Val(mode), cpar, settings; kwargs...)
get_growth(@nospecialize(mode), cpar::CosmoPar, settings::Settings; kwargs...) = error("Dz mode $(typeof(i)) not supported.")

function get_growth(mode::Val{:RK2}, cpar::CosmoPar, settings::Settings; kwargs...)
    cosmo_type = _get_cosmo_type(cpar)
    x = settings.xs
    z = settings.zs
    a = @.(1/(1+z))
    dx = x[2]-x[1]
    
    aa = reverse(a)
    e = Ez(cpar, z)
    ee = reverse(e)
    dd = zeros(cosmo_type, settings.nz)
    yy = zeros(cosmo_type, settings.nz)
    dd[1] = aa[1]
    yy[1] = aa[1]^3*ee[end]
    
    for i in 1:(settings.nz-1)
        A0 = -1.5 * cpar.Ωm / (aa[i]*ee[i])
        B0 = -1. / (aa[i]^2*ee[i])
        A1 = -1.5 * cpar.Ωm / (aa[i+1]*ee[i+1])
        B1 = -1. / (aa[i+1]^2*ee[i+1])
        yy[i+1] = (1+0.5*dx^2*A0*B0)*yy[i] + 0.5*(A0+A1)*dx*dd[i]
        dd[i+1] = 0.5*(B0+B1)*dx*yy[i] + (1+0.5*dx^2*A0*B0)*dd[i]
    end
    
    y = reverse(yy)
    d = reverse(dd)
    
    Dzs = d./d[1] 
    Dzi = linear_interpolation(z, Dzs, extrapolation_bc=Line())
    fs8zi = linear_interpolation(z, -cpar.σ8 .* y./ (a.^2 .*e.*d[1]),
                                 extrapolation_bc=Line())
    return Dzs, Dzi, fs8zi
end

function get_growth(mode::Val{:Custom}, cpar::CosmoPar, settings::Settings; kwargs...)
    zs_c, Dzs_c = kwargs[:Dz_custom]
    d = zs_c[2]-zs_c[1]
    Dzi = cubic_spline_interpolation(zs_c, Dzs_c, extrapolation_bc=Line())
    Dzs = Dzi(settings.zs)
    dDzs_mid = (Dzs_c[2:end].-Dzs_c[1:end-1])/d
    zs_mid = (zs_c[2:end].+zs_c[1:end-1])./2
    dDzi = linear_interpolation(zs_mid, dDzs_mid, extrapolation_bc=Line())
    dDzs_c = dDzi(zs_c)
    fs8zi = cubic_spline_interpolation(zs_c, -cpar.σ8 .* (1 .+ zs_c) .* dDzs_c,
                                       extrapolation_bc=Line())
    return Dzs, Dzi, fs8zi
end

#=
function _get_growth(mode::Val{:OrdDiffEq}, cpar::CosmoPar, settings::Settings; kwargs...)
        z_ini = 1000.0
        a_ini = 1.0/(1.0+z_ini)
        ez_ini = Ez(cpar, z_ini)
        d0 = [a_ini^3*ez_ini, a_ini]
        a_s = reverse(@. 1.0 / (1.0 + zs))
        prob = ODEProblem(_dgrowth!, d0, (a_ini, 1.0), cpar)
        sol = solve(prob, Tsit5(), reltol=1E-6,
                    abstol=1E-8, saveat=a_s)
        # OPT: interpolation (see below), ODE method, tolerances
        # Note that sol already includes some kind of interpolation,
        # so it may be possible to optimize this by just using
        # sol directly.
        s = vcat(sol.u'...)
        Dzs = reverse(s[:, 2] / s[end, 2])
        # OPT: interpolation method
        Dzi = LinearInterpolation(zs, Dzs, extrapolation_bc=Line())
end
=#

function get_growth(cpar::CosmoPar, settings::Settings; kwargs...)
        return get_growth(settings.Dz_mode, cpar, settings; kwargs...)
end