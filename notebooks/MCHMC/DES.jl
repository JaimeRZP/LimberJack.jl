using Pkg
Pkg.activate("../../../MicroCanonicalHMC.jl/")

using Base.Threads
using LinearAlgebra
using Turing
using LimberJack
using DataFrames
using CSV
using YAML
using NPZ
using PythonCall
sacc = pyimport("sacc");

using MicroCanonicalHMC

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/DESY1/gcgc_gcwl_wlwl.yml"
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

cov = meta.cov
data = meta.data
    
@model function model(data; files=files)
    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ TruncatedNormal(72, 5, 0.64, 0.82)
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
    DESwl__0_dz ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    DESwl__1_dz ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    DESwl__2_dz ~ TruncatedNormal(0.009, 0.011, -0.2, 0.2)
    DESwl__3_dz ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    DESwl__0_m ~ Normal(0.012, 0.023)
    DESwl__1_m ~ Normal(0.012, 0.023)
    DESwl__2_m ~ Normal(0.012, 0.023)
    DESwl__3_m ~ Normal(0.012, 0.023)
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("DESgc__0_b" => DESgc__0_b,
                     "DESgc__1_b" => DESgc__1_b,
                     "DESgc__2_b" => DESgc__2_b,
                     "DESgc__3_b" => DESgc__3_b,
                     "DESgc__4_b" => DESgc__4_b,
                     "DESgc__0_dz" => DESgc__0_dz,
                     "DESgc__1_dz" => DESgc__1_dz,
                     "DESgc__2_dz" => DESgc__2_dz,
                     "DESgc__3_dz" => DESgc__3_dz,
                     "DESgc__4_dz" => DESgc__4_dz,
                     "DESwl__0_dz" => DESwl__0_dz,
                     "DESwl__1_dz" => DESwl__1_dz,
                     "DESwl__2_dz" => DESwl__2_dz,
                     "DESwl__3_dz" => DESwl__3_dz,
                     "DESwl__0_m" => DESwl__0_m,
                     "DESwl__1_m" => DESwl__1_m,
                     "DESwl__2_m" => DESwl__2_m,
                     "DESwl__3_m" => DESwl__3_m,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory_st(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory, cov)
end

d = 25
eps = 0.07
L = round(sqrt(d), digits=2)
sigma = ones(d)
nchains = nthreads()

stats_model = model(data)
target = TuringTarget(stats_model)
spl = MCHMC(eps, L; sigma=sigma)

# Start sampling.
folpath = "../../chains/MCHMC"
folname = string("DES_eps_", eps, "_L_", L, "_t_", nchains)
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

@threads :static for i in 1:nchains    
    file_name = string("chain_", i)
    samples= Sample(spl, target, 10_000;
                    burn_in=200, fol_name=folname, file_name=file_name, dialog=true)
end      
