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

Γ = sqrt(cov)
iΓ = inv(Γ)
data = iΓ * data

@model function model(data;
    meta=meta,
    files=files)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)
    
    DESgc__0_b = 1.48 #~ Uniform(0.8, 3.0)
    DESgc__1_b = 1.81 #~ Uniform(0.8, 3.0)
    DESgc__2_b = 1.78 #~ Uniform(0.8, 3.0)
    DESgc__3_b = 2.17 #~ Uniform(0.8, 3.0)
    DESgc__4_b = 2.21 #~ Uniform(0.8, 3.0)
    eBOSS__0_b = 2.444 #~ Uniform(0.8, 5.0)
    eBOSS__1_b = 2.630 #~ Uniform(0.8, 5.0)

    nuisances = Dict("DESgc__0_b" => DESgc__0_b,
                     "DESgc__1_b" => DESgc__1_b,
                     "DESgc__2_b" => DESgc__2_b,
                     "DESgc__3_b" => DESgc__3_b,
                     "DESgc__4_b" => DESgc__4_b,
                     "eBOSS__0_b" => eBOSS__0_b,
                     "eBOSS__1_b" => eBOSS__1_b)

    
    cosmology = LimberJack.Cosmology(Ωm, Ωb, h, ns, σ8,
                                     tk_mode="EisHu",
                                     Pk_mode="Halofit")
    
    cls = Theory(cosmology, meta, files; Nuisances=nuisances)
    fs8s = fs8(cosmology, fs8_zs)
    theory = [fs8s; cls]
    
    data ~ MvNormal(iΓ * theory, I)
end;

iterations = 500
adaptation = 300
TAP = 0.65
init_ϵ = 0.005

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

# Create a placeholder chain file.
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), Dict("params"=>[]), append=true)

# Sample
cond_model = model(data)
sampler = NUTS(adaptation, TAP; init_ϵ=init_ϵ)
chain = sample(cond_model, sampler, iterations;
                progress=true, save_state=true,
                callback=Turing.Inference.SaveCSV,
                chain_name=joinpath(folname, string("chain_temp_", last_n+1)))

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])