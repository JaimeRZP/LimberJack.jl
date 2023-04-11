using Distributed

@everywhere using Turing
@everywhere using LimberJack
@everywhere using CSV
@everywhere using NPZ
@everywhere using FITSIO
@everywhere using LinearAlgebra
@everywhere using PythonCall
@everywhere np = pyimport("numpy")

@everywhere println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

@everywhere fol = "DESY1"
@everywhere data_set = "gcgc_Nzs_40"
@everywhere meta = np.load(string("../../data/", fol, "/", data_set, "_meta.npz"))
@everywhere files = npzread(string("../../data/", fol, "/", data_set, "_files.npz"))

@everywhere tracers_names = pyconvert(Vector{String}, meta["tracers"])
@everywhere pairs = pyconvert(Vector{Vector{String}}, meta["pairs"]);
@everywhere idx = pyconvert(Vector{Int}, meta["idx"])
@everywhere data_vector = pyconvert(Vector{Float64}, meta["cls"])
@everywhere cov_tot = pyconvert(Matrix{Float64}, meta["cov"])
@everywhere errs = sqrt.(diag(cov_tot))
@everywhere fid_cosmo = LimberJack.Cosmology(0.3, 0.05, 0.67, 0.96, 0.81, 
                                             tk_mode="EisHu",
                                             Pk_mode="Halofit")
@everywhere fid_nui = Dict("DESgc__0_0_b" => 1.21,
                           "DESgc__1_0_b" => 1.30,
                           "DESgc__2_0_b" => 1.48,
                           "DESgc__3_0_b" => 1.64,
                           "DESgc__4_0_b" => 1.84)
@everywhere fake_data = Theory(fid_cosmo, tracers_names, pairs,
                               idx, files; Nuisances=fid_nui) ./ errs
@everywhere fake_cov = Hermitian(cov_tot ./ (errs * errs'));

@everywhere nz_path = "../../data/DESY1/binned_40_nzs/"
@everywhere zs_k0, nz_k0, cov_k0 = get_nzs(nz_path, "DESgc__0_0")
@everywhere zs_k1, nz_k1, cov_k1 = get_nzs(nz_path, "DESgc__1_0")
@everywhere zs_k2, nz_k2, cov_k2 = get_nzs(nz_path, "DESgc__2_0")
@everywhere zs_k3, nz_k3, cov_k3 = get_nzs(nz_path, "DESgc__3_0")
@everywhere zs_k4, nz_k4, cov_k4 = get_nzs(nz_path, "DESgc__4_0")

@everywhere @model function model(data;
                                  tracers_names=tracers_names,
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
    
    n = length(nz_k0)
    DESgc__0_0_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__1_0_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__2_0_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__3_0_nz = zeros(cosmology.settings.cosmo_type, n)
    DESgc__4_0_nz = zeros(cosmology.settings.cosmo_type, n)
    for i in 1:n
        DESgc__0_0_nz[i] ~ TruncatedNormal(nz_k0[i], sqrt.(diag(cov_k0))[i], 0.0, 1) 
        DESgc__1_0_nz[i] ~ TruncatedNormal(nz_k1[i], sqrt.(diag(cov_k1))[i], 0.0, 1) 
        DESgc__2_0_nz[i] ~ TruncatedNormal(nz_k2[i], sqrt.(diag(cov_k2))[i], 0.0, 1) 
        DESgc__3_0_nz[i] ~ TruncatedNormal(nz_k3[i], sqrt.(diag(cov_k3))[i], 0.0, 1) 
        DESgc__4_0_nz[i] ~ TruncatedNormal(nz_k4[i], sqrt.(diag(cov_k4))[i], 0.0, 1) 
    end

    DESgc__0_0_b ~ Uniform(0.8, 3.0) # = 1.21
    DESgc__1_0_b ~ Uniform(0.8, 3.0) # = 1.30
    DESgc__2_0_b ~ Uniform(0.8, 3.0) # = 1.48
    DESgc__3_0_b ~ Uniform(0.8, 3.0) # = 1.64
    DESgc__4_0_b ~ Uniform(0.8, 3.0) # = 1.84

    nuisances = Dict("DESgc__0_0_nz" => DESgc__0_0_nz,
                     "DESgc__1_0_nz" => DESgc__1_0_nz,
                     "DESgc__2_0_nz" => DESgc__2_0_nz,
                     "DESgc__3_0_nz" => DESgc__3_0_nz,
                     "DESgc__4_0_nz" => DESgc__4_0_nz,
        
                     "DESgc__0_0_b" => DESgc__0_0_b,
                     "DESgc__1_0_b" => DESgc__1_0_b,
                     "DESgc__2_0_b" => DESgc__2_0_b,
                     "DESgc__3_0_b" => DESgc__3_0_b,
                     "DESgc__4_0_b" => DESgc__4_0_b)
    
    theory = Theory(cosmology, tracers_names, pairs,
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
folname = string("Nzs40_gcgc_b_numerical_", "TAP_", TAP)
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
