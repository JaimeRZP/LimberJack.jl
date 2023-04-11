using Distributed

@everywhere begin
    using LinearAlgebra
    using Turing
    using LimberJack
    using GaussianProcess
    using CSV
    using NPZ
    using YAML
    using PythonCall
    sacc = pyimport("sacc");

    println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

    sacc_path = "../../data/LSST/cls_FD_covG.fits"
    yaml_path = "../../data/LSST/gcgc_Nzs_40.yml"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    meta, files = make_data(sacc_file, yaml_file)

@everywhere fol = "LSST"
@everywhere data_set = "gcgc_Nzs_40"
@everywhere meta = np.load(string("../../data/", fol, "/", data_set, "_meta.npz"))
@everywhere files = npzread(string("../../data/", fol, "/", data_set, "_files.npz"))

@everywhere names = pyconvert(Vector{String}, meta["names"])
@everywhere pairs = pyconvert(Vector{Vector{String}}, meta["pairs"]);
@everywhere idx = pyconvert(Vector{Int}, meta["idx"])
@everywhere types = pyconvert(Vector{String}, meta["types"])
@everywhere data_vector = pyconvert(Vector{Float64}, meta["cls"])
@everywhere cov_tot = npzread("../../data/DESY1/binned_40_nzs/gcgc_cov_marg_lsst.npz")["cov_marg"]
@everywhere errs = sqrt.(diag(cov_tot))
@everywhere fake_data = data_vector ./ errs
@everywhere fake_cov = Hermitian(cov_tot ./ (errs * errs'));

@everywhere @model function model(data;
                                  names=names,
                                  types=types,
                                  pairs=pairs,
                                  idx=idx,
                                  cov=fake_cov, 
                                  files=files)
    
    Ωm ~ Uniform(0.2, 0.6)
    s8 = 0.81 #~ Uniform(0.6, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    DESwl__0_b ~ Uniform(0.8, 3.0) # = 1.21
    DESwl__1_b ~ Uniform(0.8, 3.0) # = 1.30
    DESwl__2_b ~ Uniform(0.8, 3.0) # = 1.48
    DESwl__3_b ~ Uniform(0.8, 3.0) # = 1.64

    nuisances = Dict("DESwl__0_b" => DESwl__0_b,
                     "DESwl__1_b" => DESwl__1_b,
                     "DESwl__2_b" => DESwl__2_b,
                     "DESwl__3_b" => DESwl__3_b)
    
    theory = Theory(cosmology, names, types, pairs,
                    idx, files; Nuisances=nuisances)
    data ~ MvNormal(theory ./ errs, cov)
end;

cycles = 6
steps = 50
iterations = 100
TAP = 0.60
adaptation = 100
init_ϵ = 0.05
nchains = nprocs()
println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("init_ϵ ", init_ϵ)
println("nchains ", nchains)

# Start sampling.
folpath = "../../chains/Nzs_chains/"
folname = string("Nzs40_LSST_gcgc_b_analytical_", "TAP_", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    println("Found existing file")
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        println("Restarting chain")
    else
        last_n = 0
    end
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

for i in (1+last_n):(last_n+cycles)
    if i == 1
        chain = sample(model(fake_data), NUTS(adaptation, TAP), MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(model(fake_data), NUTS(adaptation, TAP), MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true, resume_from=old_chain)
    end  
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
