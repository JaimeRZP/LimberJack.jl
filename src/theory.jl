function Theory(cosmology::Cosmology,
                names, types, pairs,
                idx, files;
                Nuisances=Dict(),
                res_wl=350, res_gc=1000,
                int_wl="linear", int_gc="linear",
                smooth_gc=0)
    
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
                                        b=b, res=res_gc, 
                                        nz_interpolation=int_gc,
                                        smooth=smooth_gc)
        elseif t_type == "galaxy_shear"
            zs_mean, nz_mean = files[string("nz_", name)]
            m = get(Nuisances, string(name, "_", "m"), 0.0)
            IA_params = [get(Nuisances, "A_IA", 0.0),
                         get(Nuisances, "alpha_IA", 0.0)]
            nz = get(Nuisances, string(name, "_", "nz"), nz_mean)
            zs = get(Nuisances, string(name, "_", "zs"), zs_mean)
            dz = get(Nuisances, string(name, "_", "dz"), 0.0)
            tracer = WeakLensingTracer(cosmology, zs .+ dz, nz;
                                       m=m, IA_params=IA_params,
                                       res=res_wl,
                                       nz_interpolation=int_wl)
            
        elseif t_type == "cmb_convergence"
            tracer = CMBLensingTracer(cosmology)

        else
            @error("Tracer not implemented")
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
        lss = [[i for i in l-3:l+3] for l in ls]
        clss = [mean(angularCℓs(cosmology, tracer1, tracer2, _lls)) for _lls in lss]
        cls[idx[i]+1:idx[i+1]] = clss
    end
    
    return cls
end

"""
    Theory(cosmology::Cosmology,
           instructions::Instructions, files;
           Nuisances=Dict())

Composes a theory vector given a `Cosmology` object, \
a `Meta` objectm, a `files` npz file and \
a dictionary of nuisance parameters.

Arguments:
- `cosmology::Cosmology` : `Cosmology` object.
- `meta::Meta` : `Meta` object.
- `files` : `files` `npz` file.
- `Nuisances::Dict` : dictonary of nuisace parameters. 

Returns:
- Meta: structure
```
struct Meta
    names : names of tracers.
    pairs : pairs of tracers to compute angular spectra for.
    types : types of the tracers.
    idx : positions of cls in theory vector.
    data : data vector.
    cov : covariance of the data.
    inv_cov : inverse covariance of the data.
end
```
- files: npz file
"""
function Theory(cosmology::Cosmology,
                instructions::Instructions, files;
                Nuisances=Dict(),
                kwargs...)
    
    names = instructions.names
    types = instructions.types
    pairs = instructions.pairs
    idx = instructions.idx
    
    return Theory(cosmology::Cosmology,
                  names, types, pairs,
                  idx, files;
                  Nuisances=Nuisances,
                  kwargs...)
 end
