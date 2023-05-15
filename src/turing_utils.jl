function Loglike(model, MAP_names)

    function LL(MAP_vals;
                model=model,
                MAP_names=MAP_names)
        MAP_namedT = NamedTuple(Symbol(name)=>val for (name,val) in zip(MAP_names, MAP_vals))
        return Turing.loglikelihood(model, MAP_namedT)
    end

    return LL
end

function get_MAP(loglike::Function,
                 lower_bound::Vector{Float64},
                 upper_bound::Vector{Float64})

    start_value = (upper_bound .+ lower_bound) ./ 2
    opt = optimize((v)->-loglike(v), lower_bound, upper_bound, start_value, Fminbox())
    return Optim.minimizer(opt)
end

function get_mass_matrix(loglike::Function, MAP)

    # Get the Hessian
    hess = ForwardDiff.hessian(loglike, values(MAP))
    inv_hess = inv(hess)
    # Turn the Hessian into more of a covariance Matrix
    w, v = eigen(inv_hess)
    hess_cov = v * (diagm(abs.(w)) * v')
    hess_cov = tril(hess_cov) + triu(hess_cov', 1)
    hess_cov = Hermitian(hess_cov)
    hess_cov = convert(Matrix{Float64}, hess_cov)
    return hess_cov
end

function Theory(cosmology::Cosmology,
                names, types, pairs,
                idx, files;
                Nuisances=Dict())
    
    nui_type =  eltype(valtype(Nuisances))
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    tracers =  Dict{String}{Tracer}()
    ntracers = length(names)
    @inbounds for i in 1:ntracers
        name = names[i]
        t_type = types[i]
        if t_type == "galaxy_density"
            zs_mean, nz_mean = files[string("nz_", name)]
            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = NumberCountsTracer(cosmology, zs .+ dz, nz;
                                        b=b)
        elseif t_type == "galaxy_shear"
            zs_mean, nz_mean = files[string("nz_", name)]
            m = get(Nuisances, string(name, "_", "m"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = WeakLensingTracer(cosmology, zs .+ dz, nz;
                                       m=m, IA_params=IA_params)
            
        elseif t_type == "cmb_convergence"
            tracer = CMBLensingTracer(cosmology)

        else
            print("Not implemented")
            tracer = nothing
        end
        merge!(tracers, Dict(name => tracer))
    end

    npairs = length(pairs)
    total_len = last(idx)
    cls = zeros(cosmology.settings.cosmo_type, total_len)
    @inbounds Threads.@threads :static for i in 1:npairs
        name1, name2 = pairs[i]
        ls = files[string("ls_", name1, "_", name2)]
        tracer1 = tracers[name1]
        tracer2 = tracers[name2]
        cls[idx[i]+1:idx[i+1]] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end
    
    return cls
end

function Theory(cosmology::Cosmology,
                instructions, files;
                Nuisances=Dict())
    
    names = instructions.names
    types = instructions.types
    pairs = instructions.pairs
    idx = instructions.idx
    
    return Theory(cosmology::Cosmology,
                  names, types, pairs,
                  idx, files;
                  Nuisances=Nuisances)
 end

function Theory_st(cosmology::Cosmology,
                names, types, pairs,
                idx, files;
                Nuisances=Dict())
    
    nui_type =  eltype(valtype(Nuisances))
    if !(nui_type <: Float64) & (nui_type != Any)
        if nui_type != Real
            cosmology.settings.cosmo_type = nui_type
        end
    end
    
    tracers =  Dict{String}{Tracer}()
    ntracers = length(names)
    @inbounds for i in 1:ntracers
        name = names[i]
        t_type = types[i]
        if t_type == "galaxy_density"
            zs_mean, nz_mean = files[string("nz_", name)]
            b = get(Nuisances, string(name, "_", "b"), 1.0)
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = NumberCountsTracer(cosmology, zs .- dz, nz;
                                        b=b)
        elseif t_type == "galaxy_shear"
            zs_mean, nz_mean = files[string("nz_", name)]
            m = get(Nuisances, string(name, "_", "m"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = WeakLensingTracer(cosmology, zs .- dz, nz;
                                       m=m, IA_params=IA_params)
            
        elseif t_type == "cmb_convergence"
            tracer = CMBLensingTracer(cosmology)

        else
            print("Not implemented")
            tracer = nothing
        end
        merge!(tracers, Dict(name => tracer))
    end

    npairs = length(pairs)
    total_len = last(idx)
    cls = zeros(cosmology.settings.cosmo_type, total_len)
    @inbounds for i in 1:npairs
        name1, name2 = pairs[i]
        ls = files[string("ls_", name1, "_", name2)]
        tracer1 = tracers[name1]
        tracer2 = tracers[name2]
        cls[idx[i]+1:idx[i+1]] = angularCℓs(cosmology, tracer1, tracer2, ls)
    end
    
    return cls
end

function Theory_st(cosmology::Cosmology,
                instructions, files;
                Nuisances=Dict())
    
    names = instructions.names
    types = instructions.types
    pairs = instructions.pairs
    idx = instructions.idx
    
    return Theory_st(cosmology::Cosmology,
                  names, types, pairs,
                  idx, files;
                  Nuisances=Nuisances)
 end
