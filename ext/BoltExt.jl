module BoltExt

if isdefined(Base, :get_extension)
    using Bolt
else
    using ..Bolt
end

function lin_Pk0(mode::Val{:Bolt}, cpar::CosmoPar, settings::Settings)
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
    pk0_Bolt = [plin(k,𝕡,bg,ih) for k in ks_Bolt]
    pki_Bolt = linear_interpolation(ks_Bolt, log.(pk0_Bolt);
                                    extrapolation_bc=Line())
    pk0 = exp.(pki_Bolt(settings.ks))
    return pk0
end

end # module