using Distributed

@everywhere begin
    using Turing
    using LimberJack
    using CSV
    using NPZ
    using YAML
    using LinearAlgebra
    using PythonCall
    sacc = pyimport("sacc");

    println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

    sacc_path = "../../data/LSST/cls_covG_lsst.fits"
    yaml_path = "../../data/DESY1/wlwl.yml"
    nz_path = "../../data/DESY1/binned_40_nzs/"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    nz_DESwl__0 = npzread(string(nz_path, "nz_DESwl__0.npz"))
    nz_DESwl__1 = npzread(string(nz_path, "nz_DESwl__1.npz"))
    nz_DESwl__2 = npzread(string(nz_path, "nz_DESwl__2.npz"))
    nz_DESwl__3 = npzread(string(nz_path, "nz_DESwl__3.npz"))
    meta, files = make_data(sacc_file, yaml_file;
                            nz_DESwl__0=nz_DESwl__0,
                            nz_DESwl__1=nz_DESwl__1,
                            nz_DESwl__2=nz_DESwl__2,
                            nz_DESwl__3=nz_DESwl__3)

    data_vector = meta.data
    cov_tot = npzread("../../data/DESY1/binned_40_nzs/wlwl_cov_marg_lsst.npz")["cov_marg"]

    errs = sqrt.(diag(cov_tot))
    fake_data = data_vector ./ errs
    fake_cov = Hermitian(cov_tot ./ (errs * errs'))
end

@everywhere @model function model(data;
                                  meta=meta,
                                  files=files,
                                  cov=fake_cov)
    Ωm ~ Uniform(0.2, 0.6)
    s8 ~ Uniform(0.6, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)
    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    A_IA = 0.0 #~ Uniform(-5, 5)
    alpha_IA = 0.0 #~ Uniform(-5, 5)
    DESwl__0_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__1_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__2_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__3_m = 0.012 #~ Normal(0.012, 0.023)


    nuisances = Dict("A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,
                     "DESwl__0_m" => DESwl__0_m,
                     "DESwl__1_m" => DESwl__1_m,
                     "DESwl__2_m" => DESwl__2_m,
                     "DESwl__3_m" => DESwl__3_m)
    
    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory ./ errs, cov)
end;

cycles = 6
iterations = 250
nchains = nprocs()

adaptation = 250
TAP = 0.65

stats_model = model(fake_data)
sampler = NUTS(adaptation, TAP)

println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("nchains ", nchains)

# Start sampling.
folpath = "../../chains/Nzs_chains/"
folname = string("Nzs40_LSST_analytical_lite_", "TAP_", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    println("Found existing file ", folname)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        println("Restarting chain")
    else
        println("Starting new chain")
        last_n = 0
    end
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

for i in (1+last_n):(cycles+last_n)
    if i == 1
        chain = sample(stats_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(stats_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true, resume_from=old_chain)
    end
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
