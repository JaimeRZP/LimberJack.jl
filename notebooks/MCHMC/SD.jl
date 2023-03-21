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
sacc = pyimport("sacc")

using MicroCanonicalHMC

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/SD/SD.yml"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
meta, files = make_data(sacc_file, yaml_file)
    
cov = meta.cov
data = meta.data

@model function model(data; files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Uniform(0.64, 0.82)
    s8 ~ Uniform(0.6, 0.9)
    ns ~ Uniform(0.84, 1.1)
    
    DECALS__0_0_b ~ Uniform(0.8, 3.0)
    DECALS__1_0_b ~ Uniform(0.8, 3.0)
    DECALS__2_0_b ~ Uniform(0.8, 3.0)
    DECALS__3_0_b ~ Uniform(0.8, 3.0)
    DECALS__0_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DECALS__1_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DECALS__2_0_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DECALS__3_0_dz ~ TruncatedNormal(0.0, 0.010, -0.2, 0.2)
    KiDS1000__0_e_dz ~ TruncatedNormal(0.0, 0.0106, -0.2, 0.2)
    KiDS1000__1_e_dz ~ TruncatedNormal(0.0, 0.0113, -0.2, 0.2)
    KiDS1000__2_e_dz ~ TruncatedNormal(0.0, 0.0118, -0.2, 0.2)
    KiDS1000__3_e_dz ~ TruncatedNormal(0.0, 0.0087, -0.2, 0.2)
    KiDS1000__4_e_dz ~ TruncatedNormal(0.0, 0.0097, -0.2, 0.2)
    KiDS1000__0_e_m ~ Normal(0.0, 0.019)
    KiDS1000__1_e_m ~ Normal(0.0, 0.020)
    KiDS1000__2_e_m ~ Normal(0.0, 0.017)
    KiDS1000__3_e_m ~ Normal(0.0, 0.012)
    KiDS1000__4_e_m ~ Normal(0.0, 0.010)
    eBOSS__0_0_b ~ Uniform(0.8, 5.0)
    eBOSS__1_0_b ~ Uniform(0.8, 5.0)
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    nuisances = Dict("DECALS__0_0_b" => DECALS__0_0_b,
                     "DECALS__1_0_b" => DECALS__1_0_b,
                     "DECALS__2_0_b" => DECALS__2_0_b,
                     "DECALS__3_0_b" => DECALS__3_0_b,
                     "DECALS__0_0_dz" => DECALS__0_0_dz,
                     "DECALS__1_0_dz" => DECALS__1_0_dz,
                     "DECALS__2_0_dz" => DECALS__2_0_dz,
                     "DECALS__3_0_dz" => DECALS__3_0_dz,
                     "KiDS1000__0_e_dz" => KiDS1000__0_e_dz,
                     "KiDS1000__1_e_dz" => KiDS1000__1_e_dz,
                     "KiDS1000__2_e_dz" => KiDS1000__2_e_dz,
                     "KiDS1000__3_e_dz" => KiDS1000__3_e_dz,
                     "KiDS1000__4_e_dz" => KiDS1000__4_e_dz,
                     "KiDS1000__0_e_m" => KiDS1000__0_e_m,
                     "KiDS1000__1_e_m" => KiDS1000__1_e_m,
                     "KiDS1000__2_e_m" => KiDS1000__2_e_m,
                     "KiDS1000__3_e_m" => KiDS1000__3_e_m,
                     "KiDS1000__4_e_m" => KiDS1000__4_e_m,
                     "eBOSS__0_0_b" => eBOSS__0_0_b,
                     "eBOSS__1_0_b" => eBOSS__1_0_b,
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA)

    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="emulator",
                                     Pk_mode="Halofit",
                                     emul_path="../../emulator/files.npz")
    
    theory = Theory_st(cosmology, meta, files; Nuisances=nuisances)
    data ~ MvNormal(theory, cov)
end;

d = 27
eps = 0.05
L = round(sqrt(d), digits=2)
sigma = ones(d)
nchains = nthreads()

stats_model = model(data)
target = TuringTarget(stats_model)
spl = MCHMC(eps, L; sigma=sigma)

# Start sampling.
folpath = "../../chains/MCHMC"
folname = string("SD_eps_", eps, "_L_", L, "_t_", nchains)
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
