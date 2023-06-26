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

data = meta.data
cov = meta.cov

errs = sqrt.(diag(cov))
data = data ./ errs
cov = Hermitian(cov ./ (errs * errs')) 

fid_cosmo = Cosmology()
n = 101
N = 201
latent_x = range(0., stop=3., length=n)
x = range(0., stop=3., length=N)
            
@model function model(data;
    meta=meta,
    files=files,
    fid_cosmo=fid_cosmo,
    latent_x=latent_x,
    x=x)

    #KiDS priors
    Ωm ~ Uniform(0.2, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ Truncated(Normal(0.72, 0.05), 0.64, 0.82)
    σ8 = 0.81
    ns ~ Uniform(0.84, 1.1)
    
    DESgc__0_b = 1.48 #~ Uniform(0.8, 3.0)
    DESgc__1_b = 1.81 #~ Uniform(0.8, 3.0)
    DESgc__2_b = 1.78 #~ Uniform(0.8, 3.0)
    DESgc__3_b = 2.17 #~ Uniform(0.8, 3.0)
    DESgc__4_b = 2.21 #~ Uniform(0.8, 3.0)
    DESgc__0_dz = -0.005 #~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__1_dz = -0.008 #~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__2_dz = -0.0001 #~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DESgc__3_dz = 0.001 #~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESgc__4_dz = -0.004 #~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)

    A_IA = 0.27 #~ Uniform(-5, 5) 
    alpha_IA = -2.41 #~ Uniform(-5, 5)

    DESwl__0_dz = -0.018 #~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    DESwl__1_dz = 0.001 #~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    DESwl__2_dz = 0.004 #~ TruncatedNormal(0.009, 0.011, -0.2, 0.2)
    DESwl__3_dz = 0.014 #~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    DESwl__0_m = 0.049 #~ Normal(0.012, 0.023)
    DESwl__1_m = 0.026 #~ Normal(0.012, 0.023)
    DESwl__2_m = 0.026 #~ Normal(0.012, 0.023)
    DESwl__3_m = -0.008 #~ Normal(0.012, 0.023)

    eBOSS__0_b = 2.444 #~ Uniform(0.8, 5.0)
    eBOSS__1_b = 2.630 #~ Uniform(0.8, 5.0)

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

                     "A_IA" => A_IA,
                     "alpha_IA" => alpha_IA,

                     "DESwl__0_dz" => DESwl__0_dz,
                     "DESwl__1_dz" => DESwl__1_dz,
                     "DESwl__2_dz" => DESwl__2_dz,
                     "DESwl__3_dz" => DESwl__3_dz,
                     "DESwl__0_m" => DESwl__0_m,
                     "DESwl__1_m" => DESwl__1_m,
                     "DESwl__2_m" => DESwl__2_m,
                     "DESwl__3_m" => DESwl__3_m,

                     "eBOSS__0_b" => eBOSS__0_b,
                     "eBOSS__1_b" => eBOSS__1_b)

    eta = 0.2
    l = 0.3
    latent_N = length(latent_x)
    v ~ filldist(truncated(Normal(0, 1), -2, 2), latent_N)
    
    mu = fid_cosmo.Dz(vec(latent_x))
    K = sqexp_cov_fn(latent_x; eta=eta, l=l)
    latent_gp = latent_GP(mu, v, K)
    gp = conditional(latent_x, x, latent_gp, sqexp_cov_fn;
                     eta=1.0, l=l)
    
    cosmology = Cosmology(Ωm, Ωb, h, ns, σ8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit", 
                          custom_Dz=[x, gp])
    
    theory = Theory(cosmology, meta, files; Nuisances=nuisances)
    
    data ~ MvNormal(theory ./ errs, cov)
end;

iterations = 100
adaptation = 300
TAP = 0.60
init_ϵ = 0.005

println("sampling settings: ")
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)

# Start sampling.
folpath = "../../chains/NUTS/18_runs/"
folname = string("ND_super_gp_EisHu_Gibbs_TAP_", TAP)
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
sampler = Gibbs(NUTS(adaptation, TAP, :Ωm, :Ωb, :h, :ns; init_ϵ=init_ϵ),
                NUTS(adaptation, TAP, :v; init_ϵ=init_ϵ))
chain = sample(cond_model, sampler, iterations;
                progress=true, save_state=true,
                callback=Turing.Inference.SaveCSV,
                chain_name=string("chain_temp_", last_n+1,".csv"))

# Save the actual chain.       
@save joinpath(folname, string("chain_", last_n+1,".jls")) chain
CSV.write(joinpath(folname, string("chain_", last_n+1,".csv")), chain)
CSV.write(joinpath(folname, string("summary_", last_n+1,".csv")), describe(chain)[1])