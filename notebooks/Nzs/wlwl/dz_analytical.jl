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

    sacc_path = "../../data/FD/cls_FD_covG.fits"
    yaml_path = "../../data/DESY1/wlwl.yml"
    nz_path = "../../data/DESY1/binned_40_nzs/"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    nz_DESwl__0 = npzread(string(nz_path, "nz_DESwl__0.npz"))
    nz_DESwl__1 = npzread(string(nz_path, "nz_DESwl__1.npz"))
    nz_DESwl__2 = npzread(string(nz_path, "nz_DESwl__2.npz"))
    nz_DESwl__3 = npzread(string(nz_path, "nz_DESwl__3.npz"))
    zs_k0, nz_k0, cov_k0 = nz_DESwl__0["z"], nz_DESwl__0["dndz"], nz_DESwl__0["cov"]
    zs_k1, nz_k1, cov_k1 = nz_DESwl__1["z"], nz_DESwl__1["dndz"], nz_DESwl__1["cov"]
    zs_k2, nz_k2, cov_k2 = nz_DESwl__2["z"], nz_DESwl__2["dndz"], nz_DESwl__2["cov"]
    zs_k3, nz_k3, cov_k3 = nz_DESwl__3["z"], nz_DESwl__3["dndz"], nz_DESwl__3["cov"]
    meta, files = make_data(sacc_file, yaml_file;
                            nz_DESwl__0=nz_DESwl__0,
                            nz_DESwl__1=nz_DESwl__1,
                            nz_DESwl__2=nz_DESwl__2,
                            nz_DESwl__3=nz_DESwl__3)

    data_vector = meta.data
    cov_tot = npzread("../../data/DESY1/binned_40_nzs/wlwl_dz_cov_marg.npz")["cov_marg"]

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

    DESwl__0_dz = 0 #~ TruncatedNormal(0.0, 4*0.017, -0.8, 0.8)
    DESwl__1_dz = 0 #~ TruncatedNormal(0.0, 4*0.017, -0.8, 0.8)
    DESwl__2_dz = 0 #~ TruncatedNormal(0.0, 4*0.013, -0.8, 0.8)
    DESwl__3_dz = 0 #~ TruncatedNormal(0.0, 4*0.015, -0.8, 0.8)
    DESwl__0_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__1_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__2_m = 0.012 #~ Normal(0.012, 0.023)
    DESwl__3_m = 0.012 #~ Normal(0.012, 0.023)
    A_IA = 0.0 #~ Uniform(-5, 5)
    alpha_IA = 0.0 #~ Uniform(-5, 5)

    nuisances = Dict("A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,
                     "DESwl__0_dz" => DESwl__0_dz,
                     "DESwl__1_dz" => DESwl__1_dz,
                     "DESwl__2_dz" => DESwl__2_dz,
                     "DESwl__3_dz" => DESwl__3_dz,
                     "DESwl__0_m" => DESwl__0_m,
                     "DESwl__1_m" => DESwl__1_m,
                     "DESwl__2_m" => DESwl__2_m,
                     "DESwl__3_m" => DESwl__3_m)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory ./ errs, cov)
end

cycles = 4
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
folpath = "../../chains/Nzs_chains/lite_runs/"
folname = string("dz_analytical_custom_nz_lite_", "TAP_", TAP)
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
