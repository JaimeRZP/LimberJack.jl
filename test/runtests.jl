using Test
using LimberJack
using ForwardDiff
using NPZ
using Statistics

test_main=true
if test_main
    println("testing main functions")
end  
test_Bolt = false
if test_Bolt
    println("testing Bolt.jl")
end    
extensive= false
if extensive
    println("extensive")
    test_results = npzread("test_extensive_results.npz")
else
    test_results = npzread("test_results.npz")
end 


test_output = Dict{String}{Vector}()

if test_main
    cosmo_EisHu = Cosmology(nz=1000, nz_t=1000, nz_pk=1000, nk=1000, tk_mode=:EisHu)
    cosmo_emul = Cosmology(Ωm=(0.12+0.022)/0.75^2, Ωb=0.022/0.75^2, h=0.75, ns=1.0, σ8=0.81,
        nz=1000, nz_t=1000, nz_pk=1000, nk=1000, tk_mode=:EmuPk)
    cosmo_emul_As = Cosmology(Ωm=0.27, Ωb=0.046, h=0.7, ns=1.0, As=2.097e-9,
        nz=1000, nz_t=1000, nz_pk=1000, nk=1000, tk_mode=:EmuPk)
    cosmo_EisHu_nonlin = Cosmology(nk=1000, nz=1000, nz_t=1000, nz_pk=1000, tk_mode=:EisHu, Pk_mode=:Halofit)
    cosmo_emul_nonlin = Cosmology(Ωm=(0.12+0.022)/0.75^2, Ωb=0.022/0.75^2, h=0.75, ns=1.0, σ8=0.81,
    nk=1000, nz=1000, nz_t=1000, nz_pk=1000, tk_mode=:EmuPk, Pk_mode=:Halofit)

    @testset "Main tests" begin
        @testset "CreateCosmo" begin
            @test cosmo_EisHu.cpar.Ωm == 0.3
        end

        @testset "BMHz" begin
            c = 299792458.0
            ztest = 10 .^ Vector(LinRange(-2, log10(1100), 300))
            Hz = cosmo_EisHu.cpar.h*100*Ez(cosmo_EisHu, ztest)
            Hz_bm = test_results["Hz"]
            merge!(test_output, Dict("Hz"=> Hz))
            @test all(@. (abs(Hz/Hz_bm-1.0) < 0.0005))
        end

        @testset "BMChi" begin
            ztest = 10 .^ Vector(LinRange(-2, log10(1100), 300))
            chi = comoving_radial_distance(cosmo_EisHu, ztest)
            chi_LSS = cosmo_EisHu.chi_LSS
            chi_bm = test_results["Chi"]
            chi_LSS_bm = test_results["Chi_LSS"]
            merge!(test_output, Dict("Chi"=> chi))
            merge!(test_output, Dict("Chi_LSS"=> [chi_LSS]))
            @test all(@. (abs(chi/chi_bm-1.0) < 0.0005))
            @test all(@. (abs(chi_LSS/chi_LSS_bm-1.0) < 0.0005))
        end

        @testset "BMGrowth" begin
            ztest = LinRange(0.01, 3.0, 100)   
            Dz = growth_factor(cosmo_EisHu, ztest)
            fz = growth_rate(cosmo_EisHu, ztest)
            fs8z = fs8(cosmo_EisHu, ztest)
            Dz_bm = test_results["Dz"]
            fz_bm = test_results["fz"]
            fs8z_bm = 0.81 .* Dz_bm .* fz_bm
            merge!(test_output, Dict("Dz"=> Dz))
            merge!(test_output, Dict("fz"=> fz))
            merge!(test_output, Dict("fs8z"=> fs8z))
            @test all(@. (abs(Dz/Dz_bm-1.0) < 0.005))
            @test all(@. (abs(fz/fz_bm-1.0) < 0.005))
            @test all(@. (abs(fs8z/fs8z_bm-1.0) < 0.005))
        end

        @testset "linear_Pk_σ8" begin
            ks = exp.(LimberJack.emulator.training_karr)
            pk_EisHu = nonlin_Pk(cosmo_EisHu, ks, 0.0)
            pk_emul = nonlin_Pk(cosmo_emul, ks, 0.0)
            pk_EisHu_bm = test_results["pk_EisHu"]
            pk_emul_bm = test_results["pk_emul"]
            merge!(test_output, Dict("pk_EisHu"=> pk_EisHu))
            merge!(test_output, Dict("pk_emul"=> pk_emul))
            @test all(@. (abs(pk_EisHu/pk_EisHu_bm-1.0) <  0.005))
            @test all(@. (abs(pk_emul/pk_emul_bm-1.0) <  0.05))
        end

        @testset "linear_Pk_As" begin
            ks = exp.(LimberJack.emulator.training_karr)
            pk_emul = nonlin_Pk(cosmo_emul_As, ks, 0.0)
            pk_emul_bm = test_results["pk_emul_As"]
            merge!(test_output, Dict("pk_emul_As"=> pk_emul))
            @test all(@. (abs(pk_emul/pk_emul_bm-1.0) <  0.05))
        end             

        @testset "nonlinear_Pk" begin
            ks = exp.(LimberJack.emulator.training_karr)
            pk_EisHu_z0 = nonlin_Pk(cosmo_EisHu_nonlin, ks, 0.0)
            pk_EisHu_z05 = nonlin_Pk(cosmo_EisHu_nonlin, ks, 0.5)
            pk_EisHu_z1 = nonlin_Pk(cosmo_EisHu_nonlin, ks, 1.0)
            pk_EisHu_z2 = nonlin_Pk(cosmo_EisHu_nonlin, ks, 2.0)
            pk_emul_z0 = nonlin_Pk(cosmo_emul_nonlin, ks, 0)
            pk_emul_z05 = nonlin_Pk(cosmo_emul_nonlin, ks, 0.5)
            pk_emul_z1 = nonlin_Pk(cosmo_emul_nonlin, ks, 1.0)
            pk_emul_z2 = nonlin_Pk(cosmo_emul_nonlin, ks, 2.0)
            pk_EisHu_z0_bm = test_results["pk_EisHu_nonlin_z0"]
            pk_EisHu_z05_bm = test_results["pk_EisHu_nonlin_z05"]
            pk_EisHu_z1_bm = test_results["pk_EisHu_nonlin_z1"]
            pk_EisHu_z2_bm = test_results["pk_EisHu_nonlin_z2"]
            pk_emul_z0_bm = test_results["pk_emul_nonlin_z0"]
            pk_emul_z05_bm = test_results["pk_emul_nonlin_z05"]
            pk_emul_z1_bm = test_results["pk_emul_nonlin_z1"]
            pk_emul_z2_bm = test_results["pk_emul_nonlin_z2"]
            merge!(test_output, Dict("pk_EisHu_nonlin_z0"=> pk_EisHu_z0))
            merge!(test_output, Dict("pk_EisHu_nonlin_z05"=> pk_EisHu_z05))
            merge!(test_output, Dict("pk_EisHu_nonlin_z1"=> pk_EisHu_z1))
            merge!(test_output, Dict("pk_EisHu_nonlin_z2"=> pk_EisHu_z2))
            merge!(test_output, Dict("pk_emul_nonlin_z0"=> pk_emul_z0))
            merge!(test_output, Dict("pk_emul_nonlin_z05"=> pk_emul_z05))
            merge!(test_output, Dict("pk_emul_nonlin_z1"=> pk_emul_z1))
            merge!(test_output, Dict("pk_emul_nonlin_z2"=> pk_emul_z2))
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(pk_EisHu_z0/pk_EisHu_z0_bm-1.0) < 0.005))
            @test all(@. (abs(pk_EisHu_z05/pk_EisHu_z05_bm-1.0) < 0.005))
            @test all(@. (abs(pk_EisHu_z1/pk_EisHu_z1_bm-1.0) < 0.005))
            @test all(@. (abs(pk_EisHu_z2/pk_EisHu_z2_bm-1.0) < 0.005))
            @test all(@. (abs(pk_emul_z0/pk_emul_z0_bm-1.0) < 0.05))
            @test all(@. (abs(pk_emul_z05/pk_emul_z05_bm-1.0) < 0.05))
            @test all(@. (abs(pk_emul_z1/pk_emul_z1_bm-1.0) < 0.05))
            @test all(@. (abs(pk_emul_z2/pk_emul_z2_bm-1.0) < 0.05))
        end
    
        @testset "Traces" begin
            z = Vector(range(0., stop=2., length=1000))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            clustering_t = NumberCountsTracer(cosmo_EisHu, z, nz)
            lensing_t = WeakLensingTracer(cosmo_EisHu, z, nz)
            CMBLensing_t = CMBLensingTracer(cosmo_EisHu)

            clustering_chis = test_results["clustering_chis"]
            clustering_warr_bm = test_results["clustering_warr"]

            lensing_chis = test_results["lensing_chis"] 
            lensing_warr_bm = test_results["lensing_warr"]

            CMBLensing_chis = test_results["CMBLensing_chis"]
            CMBLensing_warr_bm = test_results["CMBLensing_warr"]
            
            clustering_warr = clustering_t.wint(clustering_chis)
            lensing_warr = lensing_t.wint(lensing_chis)
            CMBLensing_warr = CMBLensing_t.wint(CMBLensing_chis)

            clustering_zarr = cosmo_EisHu.z_of_chi(clustering_chis)
            lensing_zarr = cosmo_EisHu.z_of_chi(lensing_chis)
            CMBLensing_zarr = cosmo_EisHu.z_of_chi(CMBLensing_chis)

            merge!(test_output, Dict("clustering_zarr"=> clustering_zarr))
            merge!(test_output, Dict("lensing_zarr"=> lensing_zarr))
            merge!(test_output, Dict("CMBLensing_zarr"=> CMBLensing_zarr))

            merge!(test_output, Dict("clustering_warr"=> clustering_warr))
            merge!(test_output, Dict("lensing_warr"=> lensing_warr))
            merge!(test_output, Dict("CMBLensing_warr"=> CMBLensing_warr))

            @test all(@. (abs(clustering_warr/clustering_warr_bm-1.0) < 0.05))
            @test all(@. (abs(lensing_warr/lensing_warr_bm-1.0)[1:-10] < 0.1))
            @test all(@. (abs(CMBLensing_warr/CMBLensing_warr_bm-1.0)[2:end-1] < 0.01))
        end

        @testset "EisHu_Cℓs" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(1, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end
            z = Vector(range(0., stop=2., length=1000))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo_EisHu, z, nz; b=1.0)
            ts = WeakLensingTracer(cosmo_EisHu, z, nz;
                                m=0.0,
                                IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo_EisHu)
            Cℓ_gg = angularCℓs(cosmo_EisHu, tg, tg, ℓs)
            Cℓ_gs = angularCℓs(cosmo_EisHu, tg, ts, ℓs)
            Cℓ_ss = angularCℓs(cosmo_EisHu, ts, ts, ℓs)
            Cℓ_gk = angularCℓs(cosmo_EisHu, tg, tk, ℓs)
            Cℓ_sk = angularCℓs(cosmo_EisHu, ts, tk, ℓs)
            Cℓ_kk = angularCℓs(cosmo_EisHu, tk, tk, ℓs)
            Cℓ_gg_bm = test_results["cl_gg_eishu"]
            Cℓ_gs_bm = test_results["cl_gs_eishu"]
            Cℓ_ss_bm = test_results["cl_ss_eishu"]
            Cℓ_gk_bm = test_results["cl_gk_eishu"]
            Cℓ_sk_bm = test_results["cl_sk_eishu"]
            Cℓ_kk_bm = test_results["cl_kk_eishu"]
            merge!(test_output, Dict("cl_gg_eishu"=> Cℓ_gg))
            merge!(test_output, Dict("cl_gs_eishu"=> Cℓ_gs))
            merge!(test_output, Dict("cl_ss_eishu"=> Cℓ_ss))
            merge!(test_output, Dict("cl_gk_eishu"=> Cℓ_gk))
            merge!(test_output, Dict("cl_sk_eishu"=> Cℓ_sk))
            merge!(test_output, Dict("cl_kk_eishu"=> Cℓ_kk))
            
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.005))
            # The ℓ=10 point is a bit inaccurate for some reason
            @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_kk/Cℓ_kk_bm-1.0) < 0.005))
        end

        @testset "nonlin_EisHu_Cℓs" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(1, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end
            z = Vector(range(0., stop=2., length=1000))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo_EisHu, z, nz; b=1.0)
            ts = WeakLensingTracer(cosmo_EisHu, z, nz;
                                m=0.0,
                                IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo_EisHu)
            Cℓ_gg = angularCℓs(cosmo_EisHu_nonlin, tg, tg, ℓs)
            Cℓ_gs = angularCℓs(cosmo_EisHu_nonlin, tg, ts, ℓs)
            Cℓ_ss = angularCℓs(cosmo_EisHu_nonlin, ts, ts, ℓs)
            Cℓ_gk = angularCℓs(cosmo_EisHu_nonlin, tg, tk, ℓs)
            Cℓ_sk = angularCℓs(cosmo_EisHu_nonlin, ts, tk, ℓs)
            Cℓ_kk = angularCℓs(cosmo_EisHu_nonlin, tk, tk, ℓs)
            Cℓ_gg_bm = test_results["cl_gg_eishu_nonlin"]
            Cℓ_gs_bm = test_results["cl_gs_eishu_nonlin"]
            Cℓ_ss_bm = test_results["cl_ss_eishu_nonlin"]
            Cℓ_gk_bm = test_results["cl_gk_eishu_nonlin"]
            Cℓ_sk_bm = test_results["cl_sk_eishu_nonlin"]
            Cℓ_kk_bm = test_results["cl_kk_eishu_nonlin"]
            merge!(test_output, Dict("cl_gg_eishu_nonlin"=> Cℓ_gg))
            merge!(test_output, Dict("cl_gs_eishu_nonlin"=> Cℓ_gs))
            merge!(test_output, Dict("cl_ss_eishu_nonlin"=> Cℓ_ss))
            merge!(test_output, Dict("cl_gk_eishu_nonlin"=> Cℓ_gk))
            merge!(test_output, Dict("cl_sk_eishu_nonlin"=> Cℓ_sk))
            merge!(test_output, Dict("cl_sk_eishu_nonlin"=> Cℓ_kk))
            
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.005))
            # The ℓ=10 point is a bit inaccurate for some reason
            @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_kk/Cℓ_kk_bm-1.0) < 0.01))
        end

        @testset "emul_Cℓs" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(1, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end
            z = Vector(range(0., stop=2., length=1000))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo_emul, z, nz; b=1.0)
            ts = WeakLensingTracer(cosmo_emul, z, nz;
                                m=0.0,
                                IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo_emul)
            Cℓ_gg = angularCℓs(cosmo_emul, tg, tg, ℓs) 
            Cℓ_gs = angularCℓs(cosmo_emul, tg, ts, ℓs)
            Cℓ_ss = angularCℓs(cosmo_emul, ts, ts, ℓs) 
            Cℓ_gk = angularCℓs(cosmo_emul, tg, tk, ℓs) 
            Cℓ_sk = angularCℓs(cosmo_emul, ts, tk, ℓs) 
            Cℓ_kk = angularCℓs(cosmo_emul, tk, tk, ℓs) 
            Cℓ_gg_bm = test_results["cl_gg_camb"]
            Cℓ_gs_bm = test_results["cl_gs_camb"]
            Cℓ_ss_bm = test_results["cl_ss_camb"]
            Cℓ_gk_bm = test_results["cl_gk_camb"]
            Cℓ_sk_bm = test_results["cl_sk_camb"]
            Cℓ_kk_bm = test_results["cl_kk_camb"]
            merge!(test_output, Dict("cl_gg_camb"=> Cℓ_gg))
            merge!(test_output, Dict("cl_gs_camb"=> Cℓ_gs))
            merge!(test_output, Dict("cl_ss_camb"=> Cℓ_ss))
            merge!(test_output, Dict("cl_gk_camb"=> Cℓ_gk))
            merge!(test_output, Dict("cl_sk_camb"=> Cℓ_sk))
            merge!(test_output, Dict("cl_kk_camb"=> Cℓ_kk))
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.05))
            # The ℓ=10 point is a bit inaccurate for some reason
            @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_kk/Cℓ_kk_bm-1.0) < 0.05))
        end

        @testset "EisHu_Halo_Cℓs" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(1, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end
            z = Vector(range(0.0, stop=2., length=256))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo_EisHu_nonlin, z, nz; b=1.0)
            ts = WeakLensingTracer(cosmo_EisHu_nonlin, z, nz;
                                m=0.0,
                                IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo_EisHu_nonlin)
            Cℓ_gg = angularCℓs(cosmo_EisHu_nonlin, tg, tg, ℓs)
            Cℓ_gs = angularCℓs(cosmo_EisHu_nonlin, tg, ts, ℓs) 
            Cℓ_ss = angularCℓs(cosmo_EisHu_nonlin, ts, ts, ℓs) 
            Cℓ_gk = angularCℓs(cosmo_EisHu_nonlin, tg, tk, ℓs) 
            Cℓ_sk = angularCℓs(cosmo_EisHu_nonlin, ts, tk, ℓs)
            Cℓ_kk = angularCℓs(cosmo_EisHu_nonlin, tk, tk, ℓs)
            
            Cℓ_gg_bm = test_results["cl_gg_eishu_nonlin"]
            Cℓ_gs_bm = test_results["cl_gs_eishu_nonlin"]
            Cℓ_ss_bm = test_results["cl_ss_eishu_nonlin"]
            Cℓ_gk_bm = test_results["cl_gk_eishu_nonlin"]
            Cℓ_sk_bm = test_results["cl_sk_eishu_nonlin"]
            Cℓ_kk_bm = test_results["cl_kk_eishu_nonlin"]
            merge!(test_output, Dict("cl_gg_eishu_nonlin"=> Cℓ_gg))
            merge!(test_output, Dict("cl_gs_eishu_nonlin"=> Cℓ_gs))
            merge!(test_output, Dict("cl_ss_eishu_nonlin"=> Cℓ_ss))
            merge!(test_output, Dict("cl_gk_eishu_nonlin"=> Cℓ_gk))
            merge!(test_output, Dict("cl_sk_eishu_nonlin"=> Cℓ_sk))
            merge!(test_output, Dict("cl_kk_eishu_nonlin"=> Cℓ_kk))
            
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.005))
            @test all(@. (abs(Cℓ_kk/Cℓ_kk_bm-1.0) < 0.01))
        end

        @testset "emul_Halo_Cℓs" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(1, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end
            z = Vector(range(0., stop=2., length=1000))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo_emul_nonlin, z, nz; b=1.0)
            ts = WeakLensingTracer(cosmo_emul_nonlin, z, nz;
                                m=0.0,
                                IA_params=[0.0, 0.0])
            tk = CMBLensingTracer(cosmo_emul_nonlin)
            Cℓ_gg = angularCℓs(cosmo_emul_nonlin, tg, tg, ℓs)
            Cℓ_gs = angularCℓs(cosmo_emul_nonlin, tg, ts, ℓs) 
            Cℓ_ss = angularCℓs(cosmo_emul_nonlin, ts, ts, ℓs) 
            Cℓ_gk = angularCℓs(cosmo_emul_nonlin, tg, tk, ℓs) 
            Cℓ_sk = angularCℓs(cosmo_emul_nonlin, ts, tk, ℓs)
            Cℓ_kk = angularCℓs(cosmo_emul_nonlin, tk, tk, ℓs)
            Cℓ_gg_bm = test_results["cl_gg_camb_nonlin"]
            Cℓ_gs_bm = test_results["cl_gs_camb_nonlin"]
            Cℓ_ss_bm = test_results["cl_ss_camb_nonlin"]
            Cℓ_gk_bm = test_results["cl_gk_camb_nonlin"]
            Cℓ_sk_bm = test_results["cl_sk_camb_nonlin"]
            Cℓ_kk_bm = test_results["cl_kk_camb_nonlin"]
            merge!(test_output, Dict("cl_gg_emul_nonlin"=> Cℓ_gg))
            merge!(test_output, Dict("cl_gs_emul_nonlin"=> Cℓ_gs))
            merge!(test_output, Dict("cl_ss_emul_nonlin"=> Cℓ_ss))
            merge!(test_output, Dict("cl_gk_emul_nonlin"=> Cℓ_gk))
            merge!(test_output, Dict("cl_sk_emul_nonlin"=> Cℓ_sk))
            merge!(test_output, Dict("cl_kk_emul_nonlin"=> Cℓ_kk))
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_kk/Cℓ_kk_bm-1.0) < 0.05))
        end

        @testset "IsBaseDiff" begin
            if extensive
                zs_Dz = LinRange(0.01, 3.0, 100)
                zs_Chi = 10 .^ Vector(LinRange(-2, log10(1100), 300))
            else    
                zs_Dz = zs_Chi = [0.1, 0.5, 1.0, 3.0]
            end

            function h(p)
                cosmo = LimberJack.Cosmology(Ωm=p)
                Hz = 100*cosmo.cpar.h*LimberJack.Ez(cosmo, zs_Chi)
                return Hz
            end

            function f(p)
                cosmo = LimberJack.Cosmology(Ωm=p)
                chi = LimberJack.comoving_radial_distance(cosmo, zs_Chi)
                return chi
            end

            function g(p)
                cosmo = LimberJack.Cosmology(Ωm=p)
                chi = LimberJack.growth_factor(cosmo, zs_Dz)
                return chi
            end

            Ωm0 = 0.3
            dΩm = 0.001

            hs = ForwardDiff.derivative(h, Ωm0)
            fs = ForwardDiff.derivative(f, Ωm0)
            gs = ForwardDiff.derivative(g, Ωm0)
            hs_bm = (h(Ωm0+dΩm)-h(Ωm0-dΩm))/2dΩm
            fs_bm = (f(Ωm0+dΩm)-f(Ωm0-dΩm))/2dΩm
            gs_bm = (g(Ωm0+dΩm)-g(Ωm0-dΩm))/2dΩm

            merge!(test_output, Dict("Hz_autodiff"=> hs))
            merge!(test_output, Dict("Chi_autodiff"=> fs))
            merge!(test_output, Dict("Dz_autodiff"=> gs))
            merge!(test_output, Dict("Hz_num"=> hs_bm))
            merge!(test_output, Dict("Chi_num"=> fs_bm))
            merge!(test_output, Dict("Dz_num"=> gs_bm))

            @test all(@. (abs(hs/hs_bm-1) < 0.005))
            @test all(@. (abs(fs/fs_bm-1) < 0.005))
            @test all(@. (abs(gs/gs_bm-1) < 0.005))
        end

        @testset "IsLinPkDiff" begin
            if extensive
                logk = range(log(0.0001), stop=log(100.0), length=500)
                ks = exp.(logk)
            else
                ks = exp.(LimberJack.emulator.training_karr)
            end

            function lin_EisHu(p)
                cosmo = Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:linear)
                pk = lin_Pk(cosmo, ks, 0.)
                return pk
            end

            function lin_emul(p)
                cosmo = Cosmology(Ωm=p, tk_mode=:EmuPk, Pk_mode=:linear)
                pk = lin_Pk(cosmo, ks, 0.)
                return pk
            end

            Ωm0 = 0.25
            dΩm = 0.01

            lin_EisHu_autodiff = abs.(ForwardDiff.derivative(lin_EisHu, Ωm0))
            lin_emul_autodiff = abs.(ForwardDiff.derivative(lin_emul, Ωm0))
            lin_EisHu_num = abs.((lin_EisHu(Ωm0+dΩm)-lin_EisHu(Ωm0-dΩm))/(2dΩm))
            lin_emul_num = abs.((lin_emul(Ωm0+dΩm)-lin_emul(Ωm0-dΩm))/(2dΩm))

            merge!(test_output, Dict("lin_EisHu_autodiff"=> lin_EisHu_autodiff))
            merge!(test_output, Dict("lin_emul_autodiff"=> lin_emul_autodiff))
            merge!(test_output, Dict("lin_EisHu_num"=> lin_EisHu_num))
            merge!(test_output, Dict("lin_emul_num"=> lin_emul_num))       
            # Median needed since errors shoot up when derivatieve
            # crosses zero
            @test median(lin_EisHu_autodiff./lin_EisHu_num.-1) < 0.05
            @test median(lin_emul_autodiff./lin_emul_num.-1) < 0.05
        end    

        @testset "IsNonlinPkDiff" begin
            if extensive
                logk = range(log(0.0001), stop=log(100.0), length=500)
                ks = exp.(logk)
            else
                ks = exp.(LimberJack.emulator.training_karr)
            end
                                                    
            function nonlin_EisHu(p)
                cosmo = Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit)
                pk = nonlin_Pk(cosmo, ks, z)
                return pk
            end

            function nonlin_emul(p)
                cosmo = Cosmology(Ωm=p, tk_mode=:EmuPk, Pk_mode=:Halofit)
                pk = nonlin_Pk(cosmo, ks, z)
                return pk
            end

            Ωm0 = 0.25
            dΩm = 0.001

            z = 0.0
            nonlin_EisHu_z0_autodiff = abs.(ForwardDiff.derivative(nonlin_EisHu, Ωm0))
            nonlin_EisHu_z0_num = abs.((nonlin_EisHu(Ωm0+dΩm)-nonlin_EisHu(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_EisHu_z0_autodiff"=> nonlin_EisHu_z0_autodiff))
            merge!(test_output, Dict("nonlin_EisHu_z0_num"=> nonlin_EisHu_z0_num))

            z = 0.5
            nonlin_EisHu_z05_autodiff = abs.(ForwardDiff.derivative(nonlin_EisHu, Ωm0))
            nonlin_EisHu_z05_num = abs.((nonlin_EisHu(Ωm0+dΩm)-nonlin_EisHu(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_EisHu_z05_autodiff"=> nonlin_EisHu_z05_autodiff))
            merge!(test_output, Dict("nonlin_EisHu_z05_num"=> nonlin_EisHu_z05_num))

            z = 1.0
            nonlin_EisHu_z1_autodiff = abs.(ForwardDiff.derivative(nonlin_EisHu, Ωm0))
            nonlin_EisHu_z1_num = abs.((nonlin_EisHu(Ωm0+dΩm)-nonlin_EisHu(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_EisHu_z1_autodiff"=> nonlin_EisHu_z1_autodiff))
            merge!(test_output, Dict("nonlin_EisHu_z1_num"=> nonlin_EisHu_z1_num))

            z = 2.0
            nonlin_EisHu_z2_autodiff = abs.(ForwardDiff.derivative(nonlin_EisHu, Ωm0))
            nonlin_EisHu_z2_num = abs.((nonlin_EisHu(Ωm0+dΩm)-nonlin_EisHu(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_EisHu_z2_autodiff"=> nonlin_EisHu_z2_autodiff))
            merge!(test_output, Dict("nonlin_EisHu_z2_num"=> nonlin_EisHu_z2_num))

            z = 0.0
            nonlin_emul_z0_autodiff = abs.(ForwardDiff.derivative(nonlin_emul, Ωm0))
            nonlin_emul_z0_num = abs.((nonlin_emul(Ωm0+dΩm)-nonlin_emul(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_emul_z0_autodiff"=> nonlin_emul_z0_autodiff))
            merge!(test_output, Dict("nonlin_emul_z0_num"=> nonlin_emul_z0_num))

            z = 0.5
            nonlin_emul_z05_autodiff = abs.(ForwardDiff.derivative(nonlin_emul, Ωm0))
            nonlin_emul_z05_num = abs.((nonlin_emul(Ωm0+dΩm)-nonlin_emul(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_emul_z05_autodiff"=> nonlin_emul_z05_autodiff))
            merge!(test_output, Dict("nonlin_emul_z05_num"=> nonlin_emul_z05_num))

            z = 1.0
            nonlin_emul_z1_autodiff = abs.(ForwardDiff.derivative(nonlin_emul, Ωm0))
            nonlin_emul_z1_num = abs.((nonlin_emul(Ωm0+dΩm)-nonlin_emul(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_emul_z1_autodiff"=> nonlin_emul_z1_autodiff))
            merge!(test_output, Dict("nonlin_emul_z1_num"=> nonlin_emul_z1_num))

            z = 2.0
            nonlin_emul_z2_autodiff = abs.(ForwardDiff.derivative(nonlin_emul, Ωm0))
            nonlin_emul_z2_num = abs.((nonlin_emul(Ωm0+dΩm)-nonlin_emul(Ωm0-dΩm))/(2dΩm))
            merge!(test_output, Dict("nonlin_emul_z2_autodiff"=> nonlin_emul_z2_autodiff))
            merge!(test_output, Dict("nonlin_emul_z2_num"=> nonlin_emul_z2_num))

            # Median needed since errors shoot up when derivatieve
            # crosses zero
            @test median(nonlin_EisHu_z0_autodiff./nonlin_EisHu_z0_num.-1) < 0.1
            @test median(nonlin_EisHu_z05_autodiff./nonlin_EisHu_z05_num.-1) < 0.1
            @test median(nonlin_EisHu_z1_autodiff./nonlin_EisHu_z1_num.-1) < 0.1
            @test median(nonlin_EisHu_z2_autodiff./nonlin_EisHu_z2_num.-1) < 0.1

            @test median(nonlin_emul_z0_autodiff./nonlin_emul_z0_num.-1) < 0.1
            @test median(nonlin_emul_z05_autodiff./nonlin_emul_z05_num.-1) < 0.1
            @test median(nonlin_emul_z1_autodiff./nonlin_emul_z1_num.-1) < 0.1
            @test median(nonlin_emul_z2_autodiff./nonlin_emul_z2_num.-1) < 0.1
        end

        @testset "AreClsDiff" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(2, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end

            function Cl_gg(p::T)::Array{T,1} where T<:Real
                cosmo = LimberJack.Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit,
                    nz=700, nz_t=700, nz_pk=700)
                z = Vector(range(0., stop=2., length=1000))
                nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
                tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
                Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
                return Cℓ_gg
            end
            
            function Cl_gs(p::T)::Array{T,1} where T<:Real
                cosmo = LimberJack.Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit, 
                    nz=700, nz_t=700, nz_pk=700)
                z = Vector(range(0., stop=2., length=1000))
                nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
                tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
                ts = WeakLensingTracer(cosmo, z, nz;
                                    m=0.0,
                                    IA_params=[0.0, 0.0])
                Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs) 
                return Cℓ_gs
            end

            function Cl_ss(p::T)::Array{T,1} where T<:Real
                cosmo = LimberJack.Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit,
                    nz=700, nz_t=700, nz_pk=700)
                z = Vector(range(0., stop=2., length=1000))
                nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
                ts = WeakLensingTracer(cosmo, z, nz;
                                    m=0.0,
                                    IA_params=[0.0, 0.0])
                Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
                return Cℓ_ss
            end
            
            function Cl_sk(p::T)::Array{T,1} where T<:Real
                cosmo = LimberJack.Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit,
                    nz=500, nz_t=500, nz_pk=700)
                z = range(0., stop=2., length=256)
                nz = @. exp(-0.5*((z-0.5)/0.05)^2)
                ts = WeakLensingTracer(cosmo, z, nz;
                                    m=0.0,
                                    IA_params=[0.0, 0.0])
                tk = CMBLensingTracer(cosmo)
                Cℓ_sk = angularCℓs(cosmo, ts, tk, ℓs)
                return Cℓ_sk
            end

            function Cl_gk(p::T)::Array{T,1} where T<:Real
                cosmo = LimberJack.Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit,
                    nz=700, nz_t=700, nz_pk=700)
                z = range(0., stop=2., length=256)
                nz = @. exp(-0.5*((z-0.5)/0.05)^2)
                tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
                tk = CMBLensingTracer(cosmo)
                Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs)
                return Cℓ_gk
            end

            function Cl_kk(p::T)::Array{T,1} where T<:Real
                cosmo = LimberJack.Cosmology(Ωm=p, tk_mode=:EisHu, Pk_mode=:Halofit,
                    nz=700, nz_t=700, nz_pk=700)
                z = range(0., stop=2., length=256)
                nz = @. exp(-0.5*((z-0.5)/0.05)^2)
                tk = CMBLensingTracer(cosmo)
                Cℓ_kk = angularCℓs(cosmo, tk, tk, ℓs)
                return Cℓ_kk
            end

            Ωm0 = 0.3
            dΩm = 0.0001

            Cl_gg_autodiff = ForwardDiff.derivative(Cl_gg, Ωm0)
            Cl_gg_num = (Cl_gg(Ωm0+dΩm)-Cl_gg(Ωm0-dΩm))/2dΩm
            Cl_gs_autodiff = ForwardDiff.derivative(Cl_gs, Ωm0)
            Cl_gs_num = (Cl_gs(Ωm0+dΩm)-Cl_gs(Ωm0-dΩm))/2dΩm
            Cl_ss_autodiff = ForwardDiff.derivative(Cl_ss, Ωm0)
            Cl_ss_num = (Cl_ss(Ωm0+dΩm)-Cl_ss(Ωm0-dΩm))/2dΩm
            Cl_sk_autodiff = ForwardDiff.derivative(Cl_sk, Ωm0)
            Cl_sk_num = (Cl_sk(Ωm0+dΩm)-Cl_sk(Ωm0-dΩm))/2dΩm
            Cl_gk_autodiff = ForwardDiff.derivative(Cl_gk, Ωm0)
            Cl_gk_num = (Cl_gk(Ωm0+dΩm)-Cl_gk(Ωm0-dΩm))/2dΩm
            Cl_kk_autodiff = ForwardDiff.derivative(Cl_kk, Ωm0)
            Cl_kk_num = (Cl_kk(Ωm0+dΩm)-Cl_kk(Ωm0-dΩm))/2dΩm

            merge!(test_output, Dict("Cl_gg_autodiff"=> Cl_gg_autodiff))
            merge!(test_output, Dict("Cl_gs_autodiff"=> Cl_gs_autodiff))
            merge!(test_output, Dict("Cl_ss_autodiff"=> Cl_ss_autodiff))
            merge!(test_output, Dict("Cl_sk_autodiff"=> Cl_sk_autodiff))
            merge!(test_output, Dict("Cl_gk_autodiff"=> Cl_gk_autodiff))
            merge!(test_output, Dict("Cl_kk_autodiff"=> Cl_kk_autodiff))
            merge!(test_output, Dict("Cl_gg_num"=> Cl_gg_num))
            merge!(test_output, Dict("Cl_gs_num"=> Cl_gs_num))
            merge!(test_output, Dict("Cl_ss_num"=> Cl_ss_num))
            merge!(test_output, Dict("Cl_sk_num"=> Cl_sk_num))
            merge!(test_output, Dict("Cl_gk_num"=> Cl_gk_num))
            merge!(test_output, Dict("Cl_kk_num"=> Cl_kk_num))

            @test all(@. (abs(Cl_gg_autodiff/Cl_gg_num-1) < 0.05))
            @test all(@. (abs(Cl_gs_autodiff/Cl_gs_num-1) < 0.05))
            @test all(@. (abs(Cl_ss_autodiff/Cl_ss_num-1) < 0.05))
            @test all(@. (abs(Cl_sk_autodiff/Cl_sk_num-1) < 0.05))
            @test all(@. (abs(Cl_gk_autodiff/Cl_gk_num-1) < 0.05))
            @test all(@. (abs(Cl_kk_autodiff/Cl_kk_num-1) < 0.05))
        end

        @testset "Nuisances" begin
            if extensive
                ℓs = 10 .^ Vector(LinRange(1, 3, 100))
            else
                ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
            end
            z = Vector(range(0.01, stop=2., length=1024))
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg_b = NumberCountsTracer(cosmo_EisHu_nonlin, z, nz; b=2.0)
            ts_m = WeakLensingTracer(cosmo_EisHu_nonlin, z, nz; m=1.0, IA_params=[0.0, 0.0])
            ts_IA = WeakLensingTracer(cosmo_EisHu_nonlin, z, nz; m=0.0, IA_params=[0.1, 0.1])
            Cℓ_gg_b = angularCℓs(cosmo_EisHu_nonlin, tg_b, tg_b, ℓs)
            Cℓ_ss_m = angularCℓs(cosmo_EisHu_nonlin, ts_m, ts_m, ℓs)
            Cℓ_ss_IA = angularCℓs(cosmo_EisHu_nonlin, ts_IA, ts_IA, ℓs)
            Cℓ_gg_b_bm = test_results["cl_gg_b"]
            Cℓ_ss_m_bm = test_results["cl_ss_m"]
            Cℓ_ss_IA_bm = test_results["cl_ss_IA"]
            merge!(test_output, Dict("cl_gg_b"=> Cℓ_gg_b))
            merge!(test_output, Dict("cl_ss_m"=> Cℓ_ss_m))
            merge!(test_output, Dict("cl_ss_IA"=> Cℓ_ss_IA))
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(Cℓ_gg_b/Cℓ_gg_b_bm-1.0) < 0.05))
            # This is problematic
            @test all(@. (abs(Cℓ_ss_m/Cℓ_ss_m_bm-1.0) < 0.05))
            @test all(@. (abs(Cℓ_ss_IA/Cℓ_ss_IA_bm-1.0) < 0.05))
        end

        @testset "AreNuisancesDiff" begin
            
            function bias(p::T)::Array{T,1} where T<:Real
                cosmo = Cosmology(tk_mode=:EisHu, Pk_mode=:Halofit)
                cosmo.settings.cosmo_type = typeof(p)
                z = Vector(range(0., stop=2., length=1000))
                nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
                tg = NumberCountsTracer(cosmo, z, nz; b=p)
                ℓs = [10.0, 30.0, 100.0, 300.0]
                Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
                return Cℓ_gg
            end
            
            function dz(p::T)::Array{T,1} where T<:Real
                cosmo = Cosmology(tk_mode=:EisHu, Pk_mode=:Halofit, nz=300)
                cosmo.settings.cosmo_type = typeof(p)
                z = Vector(range(0., stop=2., length=1000)) .- p
                nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
                tg = NumberCountsTracer(cosmo, z, nz; b=1)
                ℓs = [10.0, 30.0, 100.0, 300.0]
                Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
                return Cℓ_gg
            end
            
            function mbias(p::T)::Array{T,1} where T<:Real
                cosmo = Cosmology(tk_mode=:EisHu, Pk_mode=:Halofit)
                cosmo.settings.cosmo_type = typeof(p)
                z = range(0., stop=2., length=256)
                nz = @. exp(-0.5*((z-0.5)/0.05)^2)
                ts = WeakLensingTracer(cosmo, z, nz; m=p, IA_params=[0.0, 0.0])
                ℓs = [10.0, 30.0, 100.0, 300.0]
                Cℓ_sk = angularCℓs(cosmo, ts, ts, ℓs)
                return Cℓ_sk
            end
            
            function IA_A(p::T)::Array{T,1} where T<:Real
                cosmo = Cosmology(tk_mode=:EisHu, Pk_mode=:Halofit, )
                cosmo.settings.cosmo_type = typeof(p)
                z = range(0., stop=2., length=256)
                nz = @. exp(-0.5*((z-0.5)/0.05)^2)
                ts = WeakLensingTracer(cosmo, z, nz; m=2, IA_params=[p, 0.1])
                ℓs = [10.0, 30.0, 100.0, 300.0]
                Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
                return Cℓ_ss
            end
            
            function IA_alpha(p::T)::Array{T,1} where T<:Real
                cosmo = Cosmology(tk_mode=:EisHu, Pk_mode=:Halofit)
                cosmo.settings.cosmo_type = typeof(p)
                z = range(0., stop=2., length=256)
                nz = @. exp(-0.5*((z-0.5)/0.05)^2)
                ts = WeakLensingTracer(cosmo, z, nz; m=2, IA_params=[0.3, p])
                ℓs = [10.0, 30.0, 100.0, 300.0]
                Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
                return Cℓ_ss
            end

            d = 0.00005
            b_autodiff = ForwardDiff.derivative(bias, 2.0)
            b_anal = (bias(2.0+d)-bias(2.0-d))/2d
            dz_autodiff = ForwardDiff.derivative(dz, -0.1)
            dz_anal = (dz(-0.1+d)-dz(-0.1-d))/2d
            mb_autodiff = ForwardDiff.derivative(mbias, 2.0)
            mb_anal = (mbias(2.0+d)-mbias(2.0-d))/2d
            IA_A_autodiff = ForwardDiff.derivative(IA_A, 0.3)
            IA_A_anal = (IA_A(0.3+d)-IA_A(0.3-d))/2d
            IA_alpha_autodiff = ForwardDiff.derivative(IA_alpha, 0.1)
            IA_alpha_anal = (IA_alpha(0.1+d)-IA_alpha(0.1-d))/2d

            @test all(@. (abs(b_autodiff/b_anal-1) < 0.05))
            @test all(@. (abs(dz_autodiff/dz_anal-1) < 0.05))
            @test all(@. (abs(mb_autodiff/mb_anal-1) < 0.05))
            @test all(@. (abs(IA_A_autodiff/IA_A_anal-1) < 0.05))
            @test all(@. (abs(IA_alpha_autodiff/IA_alpha_anal-1) < 0.05))
        end
        if extensive
            npzwrite("test_extensive_output.npz", test_output)
        else    
            npzwrite("test_output.npz", test_output)
        end    
    end
end  

if test_Bolt
    Bolt_test_output = Dict{String}{Vector}()

    cosmo_Bolt_As = Cosmology(Ωm=0.27, Ωb=0.046, h=0.7, ns=1.0, As=2.097e-9,
                            nk=70, nz=300, nz_pk=70, tk_mode="Bolt")
    cosmo_Bolt_nonlin = Cosmology(Ωm=0.27, Ωb=0.046, h=0.70, ns=1.0, σ8=0.81,
                                nk=70, nz=300, nz_pk=70,
                                tk_mode="Bolt", Pk_mode=:Halofit)

    @testset "Bolt tests" begin
        @testset "linear_Pk_As" begin
            ks = exp.(LimberJack.emulator.training_karr)
            pk_Bolt = nonlin_Pk(cosmo_Bolt_As, ks, 0.0)
            pk_Bolt_bm = test_results["pk_Bolt_As"]
            merge!(Bolt_test_output, Dict("pk_Bolt_As"=> pk_Bolt))
            @test all(@. (abs(pk_Bolt/pk_Bolt_bm-1.0) <  0.05))
        end              

        @testset "nonlinear_Pk" begin
            ks = exp.(LimberJack.emulator.training_karr)
            pk_Bolt = nonlin_Pk(cosmo_Bolt_nonlin, ks, 0)
            pk_Bolt_bm = test_results["pk_Bolt_nonlin"]
            merge!(Bolt_test_output, Dict("pk_Bolt_nonlin"=> pk_Bolt))
            # It'd be best if this was < 1E-4...
            @test all(@. (abs(pk_Bolt/pk_Bolt_bm-1.0) < 0.05))
            
        end

       
        @testset "IsBoltPkDiff" begin
            if extensive
                logk = range(log(0.0001), stop=log(100.0), length=100)
                ks = exp.(logk)
            else
                logk = range(log(0.0001), stop=log(100.0), length=20)
                ks = exp.(logk)
            end     

            function lin_Bolt(p)
                cosmo = Cosmology(Ωm=p, tk_mode="Bolt", Pk_mode=:linear)
                pk = lin_Pk(cosmo, ks, 0.)
                return pk
            end

            Ωm0 = 0.25
            dΩm = 0.01

            lin_Bolt_autodiff = abs.(ForwardDiff.derivative(lin_Bolt, Ωm0))
            lin_Bolt_num = abs.((lin_Bolt(Ωm0+dΩm)-lin_Bolt(Ωm0-dΩm))/(2dΩm))

            merge!(Bolt_test_output, Dict("lin_Bolt_autodiff"=> lin_Bolt_autodiff))
            merge!(Bolt_test_output, Dict("lin_Bolt_num"=> lin_Bolt_num))        
            # Median needed since errors shoot up when derivatieve
            # crosses zero
            @test median(lin_Bolt_autodiff./lin_Bolt_num.-1) < 0.05
        end
        if extensive
            npzwrite("Bolt_test_extensive_output.npz", Bolt_test_output)
        else
            npzwrite("Bolt_test_extensive_output.npz", Bolt_test_output)
        end      
    end
end    

