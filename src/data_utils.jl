mutable struct Instructions
    names
    pairs
    types
    idx
    data
    cov 
    inv_cov
end

function _get_type(sacc_file, tracer_name)
    typ = sacc_file.tracers[tracer_name].quantity
    return pyconvert(String, typ)
end

function _apply_scale_cuts!(s, yaml_file)
    indices = Vector{Int}([])
    for cl in yaml_file["order"]
        t1, t2 = cl["tracers"]
        cls = cl["cls"]
        if "ell_cuts" in keys(cl)
            lmin, lmax = cl["ell_cuts"]
            for cl_name in cls
                ind = s.indices(cl_name, (t1, t2),
                                ell__gt=lmin, ell__lt=lmax)
                append!(indices, pyconvert(Vector{Int}, ind))
            end
        end
    end
    if length(indices) != 0
        s.keep_indices(indices)
    end
end


"""
    make_data(sacc_file, yaml_file)

Process `sacc` and `yaml` files into a `Meta` structure \
containing the instructions of how to compose the theory \
vector and a `npz` file with the neccesary redshift distributions \
of the tracers involved. 

Arguments:
- `sacc_file` : sacc file
- `yaml_file` : yaml_file

Returns:
```
struct Instructions
    names : names of tracers.
    pairs : pairs of tracers to compute angular spectra for.
    types : types of the tracers.
    idx : positions of cls in theory vector.
    data : data vector.
    cov : covariance of the data.
    inv_cov : inverse covariance of the data.
end
```
-files: npz file
"""
function make_data(sacc_file, yaml_file; kwargs...)

    kwargs=Dict(kwargs)
    kwargs_keys = [string(i) for i in collect(keys(kwargs))]

    #cut
    _apply_scale_cuts!(sacc_file, yaml_file)
        
    #build quantities of interest
    data = Vector{Float64}([])
    ls = []
    indices = Vector{Int}([])
    pairs = []
    for cl in yaml_file["order"]
        t1, t2 = cl["tracers"]
        cls = cl["cls"]
        for cl_name in cls
            l, c_ell, ind = sacc_file.get_ell_cl(
                cl_name,
                string(t1),
                string(t2),
                return_cov=false,
                return_ind=true)
            append!(indices, pyconvert(Vector{Int}, ind))
            append!(data, pyconvert(Vector{Float64}, c_ell))
            push!(ls, pyconvert(Vector{Float64}, l))
            push!(pairs, pyconvert(Vector{String}, [t1, t2]))
        end
    end
    
    names = unique(vcat(pairs...))
    cov = pyconvert(Vector{Vector{Float64}}, sacc_file.covariance.dense)
    cov = permutedims(hcat(cov...))[indices.+1, :][:, indices.+1]
    w, v = eigen(cov)
    cov = v * (diagm(abs.(w)) * v')
    cov = tril(cov) + triu(cov', 1)
    cov = Hermitian(cov)
    inv_cov = inv(cov)
    lengths = [length(l) for l in ls]
    lengths = vcat([0], lengths)
    idx  = cumsum(lengths)
    types = [_get_type(sacc_file, name) for name in names]
    
    # build struct
    instructions = Instructions(names, pairs, types, idx,
                                data, cov, inv_cov)
    
    # Initialize
    files = Dict{String}{Vector}()
    # Load in l's
    for (pair, l) in zip(pairs, ls)
        t1, t2 = pair
        println(t1, " ", t2, " ", length(l))
        merge!(files, Dict(string("ls_", t1, "_", t2)=> l))
    end
    
    # Load in nz's
    for (name, tracer) in sacc_file.tracers.items()
        if string(name) in names
            if string("nz_", name) in kwargs_keys
                println(string("using custom nz for ", string("nz_", name)))
                nzs = kwargs[Symbol("nz_", name)]
                z= pyconvert(Vector{Float64}, nzs["z"])
                nz=pyconvert(Vector{Float64}, nzs["dndz"])
                merge!(files, Dict(string("nz_", name)=>[z, nz]))
            else
                if string(tracer.quantity) != "cmb_convergence"
                    z=pyconvert(Vector{Float64}, tracer.z)
                    nz=pyconvert(Vector{Float64}, tracer.nz)
                    merge!(files, Dict(string("nz_", name)=>[z, nz]))
                end
            end
        end
    end
    
    return instructions, files
end
