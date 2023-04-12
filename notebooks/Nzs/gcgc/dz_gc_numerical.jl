using Distributed

@everywhere begin
    using LinearAlgebra
    using Turing
    using LimberJack
    using CSV
    using NPZ
    using YAML
    using PythonCall
    sacc = pyimport("sacc");

    println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

    sacc_path = "../../../data/FD/cls_FD_covG.fits"
    yaml_path = "../../../data/DESY1/gcgc.yml"
    nz_path = "../../../data/DESY1/nzs/"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    nz_DESgc__0 = npzread(string(nz_path, "nz_DESgc__0.npz"))
    nz_DESgc__1 = npzread(string(nz_path, "nz_DESgc__1.npz"))
    nz_DESgc__2 = npzread(string(nz_path, "nz_DESgc__2.npz"))
    nz_DESgc__3 = npzread(string(nz_path, "nz_DESgc__3.npz"))
    nz_DESgc__4 = npzread(string(nz_path, "nz_DESgc__4.npz"))
    zs_k0, nz_k0, cov_k0 = nz_DESgc__0["z"], nz_DESgc__0["dndz"], nz_DESgc__0["cov"]
    zs_k1, nz_k1, cov_k1 = nz_DESgc__1["z"], nz_DESgc__1["dndz"], nz_DESgc__1["cov"]
    zs_k2, nz_k2, cov_k2 = nz_DESgc__2["z"], nz_DESgc__2["dndz"], nz_DESgc__2["cov"]
    zs_k3, nz_k3, cov_k3 = nz_DESgc__3["z"], nz_DESgc__3["dndz"], nz_DESgc__3["cov"]
    zs_k4, nz_k4, cov_k4 = nz_DESgc__4["z"], nz_DESgc__4["dndz"], nz_DESgc__4["cov"]
    meta, files = make_data(sacc_file, yaml_file;
                            nz_DESgc__0=nz_DESgc__0,
                            nz_DESgc__1=nz_DESgc__1,
                            nz_DESgc__2=nz_DESgc__2,
                            nz_DESgc__3=nz_DESgc__3,
                            nz_DESgc__4=nz_DESgc__4)
    data = meta.data
    cov = meta.cov
end

@everywhere @model function model(data;
                                  meta=meta,
                                  files=files)
    Ωm ~ Uniform(0.2, 0.6)
    s8 ~ Uniform(0.6, 0.9)
    Ωb ~ Uniform(0.03, 0.07)
    h ~ Uniform(0.55, 0.91)
    ns ~ Uniform(0.87, 1.07)

    DESgc__0_b ~ Uniform(0.8, 3.0)
    DESgc__1_b ~ Uniform(0.8, 3.0)
    DESgc__2_b ~ Uniform(0.8, 3.0)
    DESgc__3_b ~ Uniform(0.8, 3.0)
    DESgc__4_b ~ Uniform(0.8, 3.0)
    DESgc__0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__1_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__2_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DESgc__3_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESgc__4_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)

    nuisances = Dict("DESgc__0_b" => DESgc__0_b,
                     "DESgc__1_b" => DESgc__1_b,
                     "DESgc__2_b" => DESgc__2_b,
                     "DESgc__3_b" => DESgc__3_b,
                     "DESgc__4_b" => DESgc__4_b,
                     "DESgc__0_dz" => DESgc__0_dz,
                     "DESgc__1_dz" => DESgc__1_dz,
                     "DESgc__2_dz" => DESgc__2_dz,
                     "DESgc__3_dz" => DESgc__3_dz,
                     "DESgc__4_dz" => DESgc__4_dz)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory, cov)
end

cycles = 4
iterations = 250
nchains = nprocs()

adaptation = 250
TAP = 0.65

cond_model = model(data)
sampler = NUTS(adaptation, TAP)

println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("nchains ", nchains)

# Start sampling.
folpath = "../../../chains/Nzs_chains/gcgc_runs/"
folname = string("dz_gc_numerical_", "TAP_", TAP)
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
        chain = sample(cond_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true)
    else
        old_chain = read(joinpath(folname, string("chain_", i-1,".jls")), Chains)
        chain = sample(cond_model, sampler, MCMCDistributed(),
                       iterations, nchains, progress=true; save_state=true, resume_from=old_chain)
    end
    write(joinpath(folname, string("chain_", i,".jls")), chain)
    CSV.write(joinpath(folname, string("chain_", i,".csv")), chain)
    CSV.write(joinpath(folname, string("summary_", i,".csv")), describe(chain)[1])
end
