#=
halofit:
- Julia version: 
- Author: andrina
- Date: 2021-09-17
- Modified by: damonge and JaimeRZP
- Date: 2022-02-19
=#

function _get_σ2(lks, ks, pks, R, kind)
    nlks = length(lks)
    k3 = ks .^ 3
    x2 = @. (ks*R)^2
    if kind == 2
        pre = x2
    elseif kind == 4
        pre = @. x2*(1-x2)
    else
        pre = 1
    end
    integrand = @. k3 * pre * exp(-x2) * pks
    #integral = sum(@.(0.5*(integrand[2:nlks]+integrand[1:nlks-1])*(lks[2:nlks]-lks[1:nlks-1])))/(2*pi^2)
    integral = integrate(lks, integrand, SimpsonEven())/(2*pi^2)
    return integral
end

function get_PKnonlin(cpar::CosmoPar, z, k, PkLz0, Dzs; cosmo_type::DataType=Real)
    nk = length(k)
    nz = length(z)
    logk = log.(k)

    Dz2s = Dzs .^ 2
    # OPT: hard-coded range and number of points
    lR0 = log(0.01)
    lR1 = log(10.0)
    lRs = range(lR0, stop=lR1, length=100)
    σ2s = zeros(cosmo_type, length(lRs))
    @inbounds for i in 1:length(lRs)
        lR = lRs[i]
        σ2s[i] = _get_σ2(logk, k, PkLz0, exp(lR), 0)
    end
    lσ2s = log.(σ2s)
    # When s8<0.6 lσ2i interpolation fails --> Extrapolation needed
    lσ2i = cubic_spline_interpolation(lRs, lσ2s, extrapolation_bc=Line())
    pk_NLs = zeros(cosmo_type, nk, nz)
    @inbounds for i in 1:nz
        Dz2 =  Dz2s[i]
        rsig =  _get_rsigma(lσ2i, Dz2, lR0, lR1)
        onederiv_int = _get_σ2(logk, k, PkLz0, rsig, 2) * Dz2
        twoderiv_int = _get_σ2(logk, k, PkLz0, rsig, 4) * Dz2
        neff = 2*onederiv_int - 3.0
        C = 4*(twoderiv_int + onederiv_int^2)
        pk_NLs[1:nk, i] = _power_spectrum_nonlin(cpar, PkLz0 .* Dz2, k, z[i], rsig, neff, C)
    end
    
    pk_NL = linear_interpolation((logk, z), log.(pk_NLs),
                                 extrapolation_bc=Line())

    return pk_NL
end 

function _secant(f, x0, x1, tol)
    # TODO: fail modes
    x_nm1 = x0
    x_nm2 = x1
    f_nm1 = f(x_nm1)
    f_nm2 = f(x_nm2)
    while abs(x_nm1 - x_nm2) > tol
        x_n = (x_nm2*f_nm1-x_nm1*f_nm2)/(f_nm1-f_nm2)
        x_nm2 = x_nm1
        x_nm1 = x_n
        f_nm2 = f_nm1
        f_nm1 = f(x_nm1)
    end
    return 0.5*(x_nm2+x_nm1)
end

function _get_rsigma(lσ2i, Dz2, lRmin, lRmax)
    lDz2 = log(Dz2)
    lRsigma = _secant(lR -> lσ2i(lR)+lDz2, lRmin, lRmax, 1E-4)
    return exp(lRsigma)
end

function _power_spectrum_nonlin(cpar::CosmoPar, PkL, k, z, rsig, neff, C)
    # DAM: note that below I've commented out anything to do with
    # neutrinos or non-Lambda dark energy.
    opz = 1.0+z
    #weffa = -1.0
    Ez2 = cpar.Ωm*opz^3+cpar.Ωr*opz^4+cpar.ΩΛ
    omegaMz = cpar.Ωm*opz^3 / Ez2
    omegaDEwz = cpar.ΩΛ/Ez2

    ksigma = @. 1.0 / rsig
    neff2 = @. neff * neff
    neff3 = @. neff2 * neff
    neff4 = @. neff3 * neff

    delta2_norm = @. k*k*k/2.0/pi/pi

    # compute the present day neutrino massive neutrino fraction
    # uses all neutrinos even if they are moving fast
    # fnu = 0.0

    # eqns A6 - A13 of Takahashi et al.
    an = @. 10.0^(1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3 +
        0.2250*neff4 - 0.6038*C)# + 0.1749*omegaDEwz*(1.0 + weffa))
    bn = @. 10.0^(-0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C)# + 0.2279*omegaDEwz*(1.0 + weffa))
    cn = @. 10.0^(0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C)
    gamman = @. 0.1971 - 0.0843*neff + 0.8460*C
    alphan = @. abs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C)
    betan = @. 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 - 0.1682*C
    #mun = 0.0
    nun = @. 10.0^(5.2105 + 3.6902*neff)

    # eqns C17 and C18 for Smith et al.
    if abs(1.0 - omegaMz) > 0.01
        f1a = omegaMz ^ -0.0732
        f2a = omegaMz ^ -0.1423
        f3a = omegaMz ^ 0.0725
        f1b = omegaMz ^ -0.0307
        f2b = omegaMz ^ -0.0585
        f3b = omegaMz ^ 0.0743
        fb_frac = omegaDEwz / (1.0 - omegaMz)
        f1 = fb_frac * f1b + (1.0 - fb_frac) * f1a
        f2 = fb_frac * f2b + (1.0 - fb_frac) * f2a
        f3 = fb_frac * f3b + (1.0 - fb_frac) * f3a
    else
        f1 = 1.0
        f2 = 1.0
        f3 = 1.0
    end

    # correction to betan from Bird et al., eqn A10
    #betan += (fnu * (1.081 + 0.395*neff2))

    # eqns A1 - A3
    y = k ./ ksigma
    y2 = @. y * y
    fy = @. y/4.0 + y2/8.0
    DeltakL =  PkL .* delta2_norm

    # correction to DeltakL from Bird et al., eqn A9
    #kh = @. k / cpar.h
    #kh2 = @. kh * kh
    #DeltakL_tilde_fac = @. fnu * (47.48 * kh2) / (1.0 + 1.5 * kh2)
    #DeltakL_tilde = @. DeltakL * (1.0 + DeltakL_tilde_fac)
    #DeltakQ = @. DeltakL * (1.0 + DeltakL_tilde)^betan / (1.0 + DeltakL_tilde*alphan) * exp(-fy)
    DeltakQ = @. DeltakL * (1.0 + DeltakL)^betan / (1.0 + DeltakL*alphan) * exp(-fy)

    DeltakHprime = @. an * y^(3.0*f1) / (1.0 + bn*y^f2 + (cn*f3*y)^(3.0 - gamman))
    #DeltakH = @. DeltakHprime / (1.0 + mun/y + nun/y2)
    #
    # correction to DeltakH from Bird et al., eqn A6-A7
    #Qnu = @. fnu * (0.977 - 18.015 * (cpar.Ωm - 0.3))
    #DeltakH *= @. (1.0 + Qnu)
    DeltakH = @. DeltakHprime / (1.0 + nun/y2)

    DeltakNL = @. DeltakQ + DeltakH
    PkNL = @. DeltakNL / delta2_norm

    return PkNL
end
