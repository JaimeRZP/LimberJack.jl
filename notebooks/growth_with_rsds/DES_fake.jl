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
    yaml_path = "../../data/DESY1/gcgc_gcwl_wlwl.yml"
    sacc_file = sacc.Sacc().load_fits(sacc_path)
    yaml_file = YAML.load_file(yaml_path)
    meta, files = make_data(sacc_file, yaml_file)
    
    pars = [0.29292871, 0.7881468, 0.04007301, 0.7817957, 0.9117439,
            1.477675, 1.805091, 1.783029, 2.173222, 2.215141,
           -0.005197634, -0.008097385, -0.0008775838, 0.001324318, -0.004479689,
            0.0, 0.0, #0.272602, -2.410019,
            0.04939203, 0.02605237, 0.0261737, -0.007550238,
           -0.01824749, 0.001053441, 0.004397685, 0.01410571]
                          #Ωm,      Ωb,       h,     ns,        s8,
    fid_cosmo = Cosmology(pars[1], pars[3], pars[4], pars[5], pars[2], 
                          tk_mode="EisHu", Pk_mode="Halofit")
    
    fid_nui = Dict("DESgc__0_b" => pars[6],
                     "DESgc__1_b" => pars[7],
                     "DESgc__2_b" => pars[8],
                     "DESgc__3_b" => pars[9],
                     "DESgc__4_b" => pars[10],
                     "DESgc__0_dz" => pars[11],
                     "DESgc__1_dz" => pars[12],
                     "DESgc__2_dz" => pars[13],
                     "DESgc__3_dz" => pars[14],
                     "DESgc__4_dz" => pars[15],
                     "A_IA" => pars[16],
                     "alpha_IA" => pars[17],
                     "DESwl__0_m" => pars[18],
                     "DESwl__1_m" => pars[19],
                     "DESwl__2_m" => pars[20],
                     "DESwl__3_m" => pars[21],
                     "DESwl__0_dz" => pars[22],
                     "DESwl__1_dz" => pars[23],
                     "DESwl__2_dz" => pars[24],
                     "DESwl__3_dz" => pars[25])

    data_vector = Theory(fid_cosmo, meta, files;
                         Nuisances=fid_nui)
    cov_tot = meta.cov
    errs = sqrt.(diag(cov_tot))
    fake_data = data_vector ./ errs
    fake_cov = Hermitian(cov_tot ./ (errs * errs'));
end 

@everywhere @model function model(data;
                                  cov=fake_cov,
                                  meta=meta, 
                                  files=files)
    #KiDS priors
    Ωm ~ Uniform(0.1, 0.6)
    Ωb ~ Uniform(0.028, 0.065)
    h ~ TruncatedNormal(72, 5, 0.64, 0.82)
    s8 ~ Uniform(0.4, 1.2)
    ns ~ Uniform(0.84, 1.1)
    
    DESgc__0_b ~ Truncated(LogNormal(0.5, 0.3), 0.0, 3.0)
    DESgc__1_b ~ Truncated(LogNormal(0.5, 0.3), 0.0, 3.0)
    DESgc__2_b ~ Truncated(LogNormal(0.5, 0.3), 0.0, 3.0)
    DESgc__3_b ~ Truncated(LogNormal(0.5, 0.3), 0.0, 3.0)
    DESgc__4_b ~ Truncated(LogNormal(0.5, 0.3), 0.0, 3.0)
    DESgc__0_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__1_dz ~ TruncatedNormal(0.0, 0.007, -0.2, 0.2)
    DESgc__2_dz ~ TruncatedNormal(0.0, 0.006, -0.2, 0.2)
    DESgc__3_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    DESgc__4_dz ~ TruncatedNormal(0.0, 0.01, -0.2, 0.2)
    A_IA ~ TruncatedNormal(0.0, 1.0, -5, 5) 
    alpha_IA ~ TruncatedNormal(0.0, 1.0, -5, 5)
    DESwl__0_dz ~ TruncatedNormal(-0.001, 0.016, -0.2, 0.2)
    DESwl__1_dz ~ TruncatedNormal(-0.019, 0.013, -0.2, 0.2)
    DESwl__2_dz ~ TruncatedNormal(0.009, 0.011, -0.2, 0.2)
    DESwl__3_dz ~ TruncatedNormal(-0.018, 0.022, -0.2, 0.2)
    DESwl__0_m ~ Normal(0.012, 0.023)
    DESwl__1_m ~ Normal(0.012, 0.023)
    DESwl__2_m ~ Normal(0.012, 0.023)
    DESwl__3_m ~ Normal(0.012, 0.023)

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
                     "DESwl__3_m" => DESwl__3_m)

    cosmology = Cosmology(Ωm, Ωb, h, ns, s8,
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    theory = Theory(cosmology, meta, files; Nuisances=fid_nui)
    data ~ MvNormal(theory ./ errs, cov)
end

cycles = 6
iterations = 500
nchains = nprocs()

adaptation = 500
TAP = 0.65
init_ϵ = 0.01

stats_model = model(fake_data)
sampler = NUTS(adaptation, TAP)

println("sampling settings: ")
println("cycles ", cycles)
println("iterations ", iterations)
println("TAP ", TAP)
println("adaptation ", adaptation)
println("nchains ", nchains)

# Start sampling.
folpath = "../../chains"
folname = string("DESY1_EisHu_fake_data_priors2_TAP_", TAP)
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
