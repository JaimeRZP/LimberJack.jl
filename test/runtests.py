import numpy as np
import pyccl as ccl

test_results = {}
extensive = True

cosmo_eishu = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu", 
                                       matter_power_spectrum="linear")
cosmo_camb = ccl.Cosmology(Omega_c=0.12/0.75**2, Omega_b=0.022/0.75**2, h=0.75, n_s=1.0, sigma8=0.81,
                           transfer_function="boltzmann_class",
                           matter_power_spectrum="linear")

cosmo_camb_As = ccl.Cosmology(Omega_c=0.224, Omega_b=0.046, h=0.70, n_s=1.0,  A_s = 2.10058e-9, 
                              transfer_function="boltzmann_class",
                              matter_power_spectrum="linear")
cosmo_bolt_As = ccl.Cosmology(Omega_c=0.224, Omega_b=0.046, h=0.70, n_s=1.0, A_s = 2.10058e-9,   
                              Omega_g=5.0469e-5, 
                              transfer_function="boltzmann_class",
                              matter_power_spectrum="linear")

cosmo_eishu_nonlin = ccl.CosmologyVanillaLCDM(transfer_function="eisenstein_hu",
                                              matter_power_spectrum="halofit",
                                              Omega_g=0, Omega_k=0)
cosmo_camb_nonlin = ccl.Cosmology(Omega_c=0.12/0.75**2, Omega_b=0.022/0.75**2, h=0.75, n_s=1.0, sigma8=0.81,
                                  transfer_function="boltzmann_class",
                                  matter_power_spectrum="halofit")
cosmo_bolt_nonlin = ccl.Cosmology(Omega_c=0.224, Omega_b=0.046, h=0.7, n_s=1.0, sigma8=0.81,
                                  Omega_g=5.0469e-5, 
                                  transfer_function="boltzmann_class",
                                  matter_power_spectrum="halofit")
# =====
ztest = np.logspace(-2, np.log10(1100), num=300)
test_results["Hz"] = 100*cosmo_eishu['h']*ccl.h_over_h0(cosmo_eishu,  1 / (1 + ztest))
test_results["Chi"] = ccl.comoving_radial_distance(cosmo_eishu,  1 / (1 + ztest))
test_results["Chi_LSS"] = ccl.comoving_radial_distance(cosmo_eishu,  1 / (1 + 1100))
# =====
ztest = np.linspace(0.01, 3., num=100)
test_results["Dz"] = ccl.growth_factor(cosmo_eishu, 1 / (1 + ztest))
test_results["fz"] = ccl.growth_rate(cosmo_eishu, 1 / (1 + ztest))
# =====
ks = np.logspace(-4, np.log10(7), 40)
test_results["pk_EisHu"] = ccl.linear_matter_power(cosmo_eishu, ks, 1.)
test_results["pk_emul"] = ccl.linear_matter_power(cosmo_camb, ks, 1.)
# =====
ks = np.logspace(-4, np.log10(7), 40)
test_results["pk_emul_As"] = ccl.linear_matter_power(cosmo_camb_As, ks, 1.)
test_results["pk_Bolt_As"] = ccl.linear_matter_power(cosmo_bolt_As, ks, 1.)
# =====
ks = np.logspace(-4, np.log10(7), 40)
test_results["pk_EisHu_nonlin_z0"] = ccl.nonlin_matter_power(cosmo_eishu_nonlin, ks, 1.)
test_results["pk_EisHu_nonlin_z05"] = ccl.nonlin_matter_power(cosmo_eishu_nonlin, ks, 0.666666)
test_results["pk_EisHu_nonlin_z1"] = ccl.nonlin_matter_power(cosmo_eishu_nonlin, ks, 0.5)
test_results["pk_EisHu_nonlin_z2"] = ccl.nonlin_matter_power(cosmo_eishu_nonlin, ks, 0.333333)
test_results["pk_emul_nonlin_z0"] = ccl.nonlin_matter_power(cosmo_camb_nonlin, ks, 1.)
test_results["pk_emul_nonlin_z05"] = ccl.nonlin_matter_power(cosmo_camb_nonlin, ks, 0.666666)
test_results["pk_emul_nonlin_z1"] = ccl.nonlin_matter_power(cosmo_camb_nonlin, ks, 0.5)
test_results["pk_emul_nonlin_z2"] = ccl.nonlin_matter_power(cosmo_camb_nonlin, ks, 0.333333)
test_results["pk_Bolt_nonlin"] = ccl.nonlin_matter_power(cosmo_bolt_nonlin, ks, 1.)
# =====
z = np.linspace(0., 2., num=1000)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
clustering_chis, clustering_warr = ccl.tracers.get_density_kernel(cosmo_eishu,(z,nz))
test_results["clustering_chis"] = clustering_chis
test_results["clustering_warr"] = clustering_warr
lensing_chis, lensing_warr = ccl.tracers.get_lensing_kernel(cosmo_eishu, (z,nz))
test_results["lensing_chis"] = lensing_chis
test_results["lensing_warr"] = lensing_warr
CMBLensing_chis, CMBLensing_warr = ccl.tracers.get_kappa_kernel(cosmo_eishu, 1100, 100)
test_results["CMBLensing_chis"] = CMBLensing_chis
test_results["CMBLensing_warr"] = CMBLensing_warr
# =====
z = np.linspace(0., 2., num=1000)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
if extensive:
    ℓs = np.logspace(1,3,100)
else:
    ℓs = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
tg = ccl.NumberCountsTracer(cosmo_eishu, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_eishu, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_eishu, z_source=1100)
test_results["cl_gg_eishu"] = ccl.angular_cl(cosmo_eishu, tg, tg, ℓs)
test_results["cl_gs_eishu"] = ccl.angular_cl(cosmo_eishu, tg, ts, ℓs)
test_results["cl_ss_eishu"] = ccl.angular_cl(cosmo_eishu, ts, ts, ℓs)
test_results["cl_gk_eishu"] = ccl.angular_cl(cosmo_eishu, tg, tk, ℓs)
test_results["cl_sk_eishu"] = ccl.angular_cl(cosmo_eishu, ts, tk, ℓs)
test_results["cl_kk_eishu"] = ccl.angular_cl(cosmo_eishu, tk, tk, ℓs)
# =====
z = np.linspace(0., 2., num=1000)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
if extensive:
    ℓs = np.logspace(1,3,100)
else:
    ℓs = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
tg = ccl.NumberCountsTracer(cosmo_eishu_nonlin, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_eishu_nonlin, z_source=1100)
test_results["cl_gg_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, tg, ℓs)
test_results["cl_gs_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, ts, ℓs)
test_results["cl_ss_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, ts, ts, ℓs)
test_results["cl_gk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, tk, ℓs)
test_results["cl_sk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, ts, tk, ℓs)
test_results["cl_kk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tk, tk, ℓs)
# =====
z = np.linspace(0., 2., num=1000)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
if extensive:
    ℓs = np.logspace(1,3,100)
else:
    ℓs = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
tg = ccl.NumberCountsTracer(cosmo_camb, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_camb, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_camb, z_source=1100)
test_results["cl_gg_camb"] = ccl.angular_cl(cosmo_camb, tg, tg, ℓs)
test_results["cl_gs_camb"] = ccl.angular_cl(cosmo_camb, tg, ts, ℓs)
test_results["cl_ss_camb"] = ccl.angular_cl(cosmo_camb, ts, ts, ℓs)
test_results["cl_gk_camb"] = ccl.angular_cl(cosmo_camb, tg, tk, ℓs)
test_results["cl_sk_camb"] = ccl.angular_cl(cosmo_camb, ts, tk, ℓs)
test_results["cl_kk_camb"] = ccl.angular_cl(cosmo_camb, tk, tk, ℓs)
# =====
z = np.linspace(0., 2., num=1000)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
if extensive:
    ℓs = np.logspace(1,3,100)
else:
    ℓs = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
tg = ccl.NumberCountsTracer(cosmo_eishu_nonlin, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_eishu_nonlin, z_source=1100)
test_results["cl_gg_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, tg, ℓs)
test_results["cl_gs_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, ts, ℓs)
test_results["cl_ss_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, ts, ts, ℓs)
test_results["cl_gk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tg, tk, ℓs)
test_results["cl_sk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, ts, tk, ℓs)
test_results["cl_kk_eishu_nonlin"] = ccl.angular_cl(cosmo_eishu_nonlin, tk, tk, ℓs)
# =====
z = np.linspace(0.01, 2., num=1024)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
if extensive:
    ℓs = np.logspace(1,3,100)
else:
    ℓs = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
tg = ccl.NumberCountsTracer(cosmo_camb_nonlin, False, dndz=(z, nz), bias=(z, 1 * np.ones_like(z)))
ts = ccl.WeakLensingTracer(cosmo_camb_nonlin, dndz=(z, nz))
tk = ccl.CMBLensingTracer(cosmo_camb_nonlin, z_source=1100)
test_results["cl_gg_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tg, tg, ℓs)
test_results["cl_gs_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tg, ts, ℓs)
test_results["cl_ss_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, ts, ts, ℓs)
test_results["cl_gk_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tg, tk, ℓs)
test_results["cl_sk_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, ts, tk, ℓs)
test_results["cl_kk_camb_nonlin"] = ccl.angular_cl(cosmo_camb_nonlin, tk, tk, ℓs)
# =====
z = np.linspace(0.01, 2., num=256)
nz = np.exp(-0.5*((z-0.5)/0.05)**2)
if extensive:
    ℓs = np.logspace(1,3,100)
else:
    ℓs = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
Dz = ccl.comoving_radial_distance(cosmo_eishu,  1 / (1 + z))
IA_corr = (0.1*((1 + z)/1.62)**0.1 * (0.0134*0.3/Dz))
tg_b = ccl.NumberCountsTracer(cosmo_eishu_nonlin, False, dndz=(z, nz), bias=(z, 2 * np.ones_like(z)))
ts_m = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz))
ts_IA = ccl.WeakLensingTracer(cosmo_eishu_nonlin, dndz=(z, nz), ia_bias=(z, IA_corr))
test_results["cl_gg_b"] = ccl.angular_cl(cosmo_eishu_nonlin, tg_b, tg_b, ℓs)
test_results["cl_ss_m"] = (1.0 + 1.0)**2  * ccl.angular_cl(cosmo_eishu_nonlin, ts_m, ts_m, ℓs)
test_results["cl_ss_IA"] = ccl.angular_cl(cosmo_eishu_nonlin, ts_IA, ts_IA, ℓs)
# =====
if extensive:
    np.savez("test_extensive_results.npz", **test_results)
else:
    np.savez("test_results.npz", **test_results)