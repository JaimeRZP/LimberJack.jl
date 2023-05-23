using LinearAlgebra
using LimberJack
using CSV
using YAML
using BenchmarkTools
using PythonCall
using ForwardDiff
sacc = pyimport("sacc");

#println("My id is ", myid(), " and I have ", Threads.nthreads(), " threads")

sacc_path = "../data/FD/cls_FD_covG.fits"
yaml_path = "../data/DESY1/gcgc_gcwl_wlwl.yml"
nz_path = "../data/DESY1/nzs"
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

p = [ 2.80864713e-01,  3.98941817e-02,  8.03805162e-01,  8.15499178e-01,
9.67896094e-01,  1.45433358e+00,  1.77905433e+00,  1.70409115e+00,
2.08994246e+00,  2.27404915e+00,  2.48932691e-03,  5.69699152e-03,
-5.19871207e-03,  4.53661141e-03, -9.45028216e-03,  5.72751378e-03,
-4.12309149e-02,  1.28279270e-03, -3.74374413e-02,  2.53967236e-02,
1.90603567e-02,  2.52821784e-02,  2.57635754e-02,  2.73739640e-01,
-6.99330759e-01]

function make_cls(p)
    #KiDS priors
    Ωm = p[1]
    Ωb = p[2]
    h = p[3]
    σ8 = p[4]
    ns = p[5]

    DESgc__0_b = p[6]
    DESgc__1_b = p[7]
    DESgc__2_b = p[8]
    DESgc__3_b = p[9]
    DESgc__4_b = p[10]
    DESgc__0_dz = p[11]
    DESgc__1_dz = p[12]
    DESgc__2_dz = p[13]
    DESgc__3_dz = p[15]
    DESgc__4_dz = p[16]
    DESwl__0_dz = p[17]
    DESwl__1_dz = p[18]
    DESwl__2_dz = p[19]
    DESwl__3_dz = p[20]
    DESwl__0_m = p[21]
    DESwl__1_m = p[22]
    DESwl__2_m = p[23]
    DESwl__3_m = p[24]
    A_IA = p[25]
    alpha_IA = p[26]

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

    cosmology = Cosmology(Ωm, Ωb, h, ns, σ8;
                          tk_mode="EisHu",
                          Pk_mode="Halofit")

    return Theory(cosmology, meta, files; Nuisances=nuisances)
end

@benchmark make_cls(p)

@benchmark ForwardDiff.gradient(make_cls, p)