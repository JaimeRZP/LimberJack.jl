using Distributed
@everywhere begin
    using LinearAlgebra
    using Turing
    using LimberJack
    using CSV
    using YAML
    using PythonCall
    sacc = pyimport("sacc");

    println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

    sacc_path = "../../data/FD/cls_FD_covG.fits"
    yaml_path = "../../data/DESY1/gcgc.yml"
    nz_path = "../../data/DESY1/nzs"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    #nz_DESwl__0 = npzread(string(nz_path, "nz_DESwl__0.npz"))
    #nz_DESwl__1 = npzread(string(nz_path, "nz_DESwl__1.npz"))
    #nz_DESwl__2 = npzread(string(nz_path, "nz_DESwl__2.npz"))
    #nz_DESwl__3 = npzread(string(nz_path, "nz_DESwl__3.npz"))
    meta, files = make_data(sacc_file, yaml_file)
                            #nz_DESwl__0=nz_DESwl__0,
                            #nz_DESwl__1=nz_DESwl__1,
                            #nz_DESwl__2=nz_DESwl__2,
                            #nz_DESwl__3=nz_DESwl__3)

    data = meta.data
    cov = meta.cov
end 

@everywhere @model function model(data;
                                  meta=meta, 
                                  files=files)
    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    s8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)

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
    #DESwl__0_dz ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    #DESwl__1_dz ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    #DESwl__2_dz ~ TruncatedNormal(0.009, 0.011, -0.2, 0.2)
    #DESwl__3_dz ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    #DESwl__0_m ~ Normal(0.012, 0.023)
    #DESwl__1_m ~ Normal(0.012, 0.023)
    #DESwl__2_m ~ Normal(0.012, 0.023)
    #DESwl__3_m ~ Normal(0.012, 0.023)
    #A_IA ~ Uniform(-5, 5) 
    #alpha_IA ~ Uniform(-5, 5)

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

                    # "DESwl__0_dz" => DESwl__0_dz,
                    # "DESwl__1_dz" => DESwl__1_dz,
                    # "DESwl__2_dz" => DESwl__2_dz,
                    # "DESwl__3_dz" => DESwl__3_dz,
                    # "DESwl__0_m" => DESwl__0_m,
                    # "DESwl__1_m" => DESwl__1_m,
                    # "DESwl__2_m" => DESwl__2_m,
                    # "DESwl__3_m" => DESwl__3_m,
                    # "A_IA" => A_IA,
                    # "alpha_IA" => alpha_IA,)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8;
                          nz_t=300
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory, cov)
end

cycles = 6
iterations = 500
nchains = nprocs()

adaptation = 500
TAP = 0.65
init_ϵ = 0.01

cond_model = model(data)
sampler = NUTS(adaptation, TAP)

println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("nchains ", nchains)

# Start sampling.
folpath = "../../chains/NUTS/standard_runs/"
folname = string("DESY1_gcgc_EisHu_high_res")
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
