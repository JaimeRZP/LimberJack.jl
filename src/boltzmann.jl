function _T0(keq, k, ac, bc)
    q = @. (k/(13.41*keq))
    C = @. ((14.2/ac) + (386/(1+69.9*q^1.08)))
    T0 = @.(log(ℯ+1.8*bc*q)/(log(ℯ+1.8*bc*q)+C*q^2))
    return T0
end

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

struct Emulator
    alphas
    hypers
    trans_cosmos
    training_karr
    inv_chol_cosmos
    mean_cosmos
    mean_log_Pks
    std_log_Pks
end

Emulator(files) = begin
    trans_cosmos = files["trans_cosmos"]
    karr = files["training_karr"]
    training_karr = range(log(karr[1]), stop=log(karr[end]), length=length(karr))
    hypers = files["hypers"]
    alphas = files["alphas"]
    inv_chol_cosmos = files["inv_chol_cosmos"]
    mean_cosmos = files["mean_cosmos"]
    mean_log_Pks = files["mean_log_Pks"]
    std_log_Pks = files["std_log_Pks"]
    #Note: transpose python arrays
    Emulator(alphas, hypers, 
             trans_cosmos', training_karr,
             inv_chol_cosmos, mean_cosmos,
             mean_log_Pks, std_log_Pks)
end

function _x_transformation(emulator::Emulator, point)
    return emulator.inv_chol_cosmos * (point .- emulator.mean_cosmos)'
end

function _y_transformation(emulator::Emulator, point)
    return ((point .- emulator.mean_log_Pks) ./ emulator.std_log_Pks)'
end

function _inv_y_transformation(emulator::Emulator, point)
    return exp.(emulator.std_log_Pks .* point .+ emulator.mean_log_Pks)
end

function _get_kernel(arr1, arr2, hyper)
    arr1_w = @.(arr1/exp(hyper[2:6]))
    arr2_w = @.(arr2/exp(hyper[2:6]))
    
    # compute the pairwise distance
    term1 = sum(arr1_w.^2, dims=1)
    term2 = 2 * (arr1_w' * arr2_w)'
    term3 = sum(arr2_w.^2, dims=1)
    dist = @.(term1-term2+term3)
    # compute the kernel
    kernel = @.(exp(hyper[1])*exp(-0.5*dist))
    return kernel
end

function get_emulated_log_pk0(cpar::CosmoPar, settings::Settings)
    cosmotype = settings.cosmo_type
    wc = cpar.Ωc*cpar.h^2
    wb = cpar.Ωb*cpar.h^2
    ln1010As = log((10^10)*cpar.As)
    params = [wc, wb, ln1010As, cpar.ns, cpar.h]
    params_t = _x_transformation(emulator, params')
    
    nk = length(emulator.training_karr)
    pk0s_t = zeros(cosmotype, nk)
    @inbounds for i in 1:nk
        kernel = _get_kernel(emulator.trans_cosmos, params_t, emulator.hypers[i, :])
        pk0s_t[i] = dot(vec(kernel), vec(emulator.alphas[i,:]))
    end
    pk0s = vec(_inv_y_transformation(emulator, pk0s_t))
    return emulator.training_karr, pk0s
end

function get_Bolt_pk0(cpar::CosmoPar, settings::Settings)
    𝕡 = Bolt.CosmoParams(h = cpar.h,
        Ω_r = cpar.Ωg,
        Ω_b = cpar.Ωb,
        Ω_c = cpar.Ωc,
        A = cpar.As,
        n = cpar.ns,
        Y_p = cpar.Y_p,
        N_ν = cpar.N_ν,
        Σm_ν = cpar.Σm_ν)
    bg = Background(𝕡)
    𝕣 = Bolt.RECFAST(bg; Yp=𝕡.Y_p, OmegaB=𝕡.Ω_b,OmegaG=𝕡.Ω_r)
    ih = IonizationHistory(𝕣, 𝕡, bg)
    ks_Bolt = LinRange(-4, 2, settings.nk)
    ks_Bolt = 10 .^(k_grid) 
    pL = [plin(k,𝕡,bg,ih) for k in ks_Bolt]
    return ks_Bolt, pL
end

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
    # OPT: proper integration instead?
    σ = integrate(log.(ks), integrand, SimpsonEven())/(2*pi^2)
    #σ = sum(integrand)*dlogk/(2*pi^2)
    return σ
end

function σR2(cosmo::Cosmology, R)
    return σR2(cosmo.ks, cosmo.pk0, cosmo.dlogk, R)
end

function lin_Pk0(cpar::CosmoPar, settings::Settings)
    if settings.tk_mode == "emupk"
        lks_emul, pk0_emul = get_emulated_log_pk0(cpar, settings)
        pki_emul = cubic_spline_interpolation(lks_emul, log.(pk0_emul),
                                              extrapolation_bc=Line())
        pk0 = exp.(pki_emul(settings.logk))
    elseif settings.tk_mode == "Bolt"
        ks_Bolt, pk0_Bolt = get_Bolt_pk0(cpar, settings)
        pki_Bolt = linear_interpolation(ks_Bolt, log.(pk0_Bolt);
                                        extrapolation_bc=Line())
        pk0 = exp.(pki_Bolt(settings.ks))
    elseif settings.tk_mode == "EisHu"
        tk = TkEisHu(cpar, settings.ks./ cpar.h)
        pk0 = @. settings.ks^cpar.ns * tk
     else
        @error("Transfer function not implemented")
    end

    #Renormalize Pk
    _σ8 = σR2(settings.ks, pk0, settings.dlogk, 8.0/cpar.h)
    if settings.using_As
        cpar.σ8 = _σ8 
    else    
        norm = cpar.σ8^2 / _σ8
        pk0 *= norm
    end
    pki = cubic_spline_interpolation(settings.logk, log.(pk0);
                                     extrapolation_bc=Line())
    return pk0, pki
end