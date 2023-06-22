using LinearAlgebra
using Turing
using LimberJack
using GaussianProcess
using CSV
using NPZ
using JLD2
using YAML
using PythonCall
sacc = pyimport("sacc");

#println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

sacc_path = "../../data/FD/cls_FD_covG.fits"
yaml_path = "../../data/ND/ND.yml"
sacc_file = sacc.Sacc().load_fits(sacc_path)
yaml_file = YAML.load_file(yaml_path)
meta, files = make_data(sacc_file, yaml_file)

cls_data = meta.data
cls_cov = meta.cov

fs8_meta = npzread("../../data/fs8s/fs8s.npz")
fs8_zs = fs8_meta["z"]
fs8_data = fs8_meta["data"]
fs8_cov = fs8_meta["cov"]

cov = zeros(Float64, length(fs8_data)+length(cls_data), length(fs8_data)+length(cls_data))
cov[1:length(fs8_data), 1:length(fs8_data)] = fs8_cov
cov[length(fs8_data)+1:(length(fs8_data)+length(cls_data)),
    length(fs8_data)+1:(length(fs8_data)+length(cls_data))] = cls_cov
data = [fs8_data; cls_data];

errs = sqrt.(diag(cov))
data = data ./ errs
cov = Hermitian(cov ./ (errs * errs')) 

@model function model(data;
    meta=meta,
    files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Uniform(0.64, 0.82)
    s8 ~ Uniform(0.6, 0.9)
    ns ~ Uniform(0.84, 1.1)
    
    DESgc__0_0_b ~ Uniform(0.8, 3.0)
    DESgc__1_0_b ~ Uniform(0.8, 3.0)
    DESgc__2_0_b ~ Uniform(0.8, 3.0)
    DESgc__3_0_b ~ Uniform(0.8, 3.0)
    DESgc__4_0_b ~ Uniform(0.8, 3.0)
    DESgc__0_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__1_0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__2_0_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DESgc__3_0_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESgc__4_0_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    
    A_IA ~ Uniform(-5, 5) 
    alpha_IA ~ Uniform(-5, 5)

    DESwl__0_e_dz ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    DESwl__1_e_dz ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    DESwl__2_e_dz ~ TruncatedNormal(-0.009, 0.011, -0.2, 0.2)
    DESwl__3_e_dz ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    DESwl__0_e_m ~ Normal(0.012, 0.023)
    DESwl__1_e_m ~ Normal(0.012, 0.023)
    DESwl__2_e_m ~ Normal(0.012, 0.023)
    DESwl__3_e_m ~ Normal(0.012, 0.023)

    eBOSS__0_0_b ~ Uniform(0.8, 5.0)
    eBOSS__1_0_b ~ Uniform(0.8, 5.0)

    nuisances = Dict("DESgc__0_0_b" => DESgc__0_0_b,
                     "DESgc__1_0_b" => DESgc__1_0_b,
                     "DESgc__2_0_b" => DESgc__2_0_b,
                     "DESgc__3_0_b" => DESgc__3_0_b,
                     "DESgc__4_0_b" => DESgc__4_0_b,
                     "DESgc__0_0_dz" => DESgc__0_0_dz,
                     "DESgc__1_0_dz" => DESgc__1_0_dz,
                     "DESgc__2_0_dz" => DESgc__2_0_dz,
                     "DESgc__3_0_dz" => DESgc__3_0_dz,
                     "DESgc__4_0_dz" => DESgc__4_0_dz,
        
                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,

                     "DESwl__0_e_dz" => DESwl__0_e_dz,
                     "DESwl__1_e_dz" => DESwl__1_e_dz,
                     "DESwl__2_e_dz" => DESwl__2_e_dz,
                     "DESwl__3_e_dz" => DESwl__3_e_dz,
                     "DESwl__0_e_m" => DESwl__0_e_m,
                     "DESwl__1_e_m" => DESwl__1_e_m,
                     "DESwl__2_e_m" => DESwl__2_e_m,
                     "DESwl__3_e_m" => DESwl__3_e_m,
        
                     "eBOSS__0_0_b" => eBOSS__0_0_b,
                     "eBOSS__1_0_b" => eBOSS__1_0_b)

    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, s8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    cls = Theory(cosmology, meta, files; Nuisances=nuisances)
    fs8s = fs8(cosmology, fs8_zs)
    theory = [fs8s; cls]

    data_vector ~ MvNormal(theory ./ errs, cov)
end;

iterations = 300
adaptation = 300
TAP = 0.65
init_ϵ = 0.01

println("sampling settings: ")
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
#println("nchains ", nchains)

# Start sampling.
folpath = "../../chains/NUTS/18_runs/"
folname = string("ND_RSD_TAP_", TAP)
folname = joinpath(folpath, folname)

if isdir(folname)
    fol_files = readdir(folname)
    println("Found existing file ", folname)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[7])
        #println("Restarting chain")
    else
        #println("Starting new chain")
        last_n = 0
    end
else
    mkdir(folname)
    println(string("Created new folder ", folname))
    last_n = 0
end

# Sample
cond_model = model(data)
sampler = NUTS(adaptation, TAP)
chain = sample(cond_model, sampler, iterations;
                progress=true, save_state=true,
                call_back=Turing.Inference.SaveCSV,
                chain_name=string("chain_", last_n+1,".csv"))

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])