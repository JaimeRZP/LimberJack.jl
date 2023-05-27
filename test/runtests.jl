using Test
using LimberJack
using ForwardDiff
using NPZ
using Statistics

extensive=false
if extensive
    println("extensive")
end    

test_results = npzread("test_results.npz")
test_cls = npzread("test_cls.npz")["cls"]
test_cls_files = npzread("test_cls_files.npz")
test_output = Dict{String}{Vector}()

cosmo_EisHu = Cosmology(nk=300, nz=300, nz_pk=70, tk_mode="EisHu")
cosmo_emul = Cosmology(Ωm=(0.12+0.022)/0.75^2, Ωb=0.022/0.75^2, h=0.75, ns=1.0, σ8=0.81,
                       nk=300, nz=300, nz_pk=70, tk_mode="emupk")

cosmo_EisHu_nonlin = Cosmology(nk=300, nz=300, nz_pk=70,
                               tk_mode="EisHu", Pk_mode="Halofit")
cosmo_emul_nonlin = Cosmology(Ωm=(0.12+0.022)/0.75^2, Ωb=0.022/0.75^2, h=0.75, ns=1.0, σ8=0.81,
                              nk=300, nz=300, nz_pk=70,
                              tk_mode="emupk", Pk_mode="Halofit")
cosmo_Bolt_nonlin = Cosmology(Ωm=0.27, Ωb=0.046, h=0.70, ns=1.0, σ8=0.81,
                              nk=70, nz=300, nz_pk=70,
                              tk_mode="Bolt", Pk_mode="Halofit")

@testset "All tests" begin
    @testset "CreateCosmo" begin
        @test cosmo_EisHu.cpar.Ωm == 0.3
    end

    @testset "BMHz" begin
        c = 299792458.0
        ztest = [0.1, 0.5, 1.0, 3.0]
        H = cosmo_EisHu.cpar.h*100*Ez(cosmo_EisHu, ztest)
        H_bm = @. 67*sqrt(0.3 * (1+ztest)^3 + (1-0.3-0.69991) * (1+ztest)^4 + 0.69991)
        @test all(@. (abs(H/H_bm-1.0) < 0.0005))
    end

    @testset "BMChi" begin
        ztest = LinRange(0.01, 3.0, 100)
        chi = comoving_radial_distance(cosmo_EisHu, ztest)
        chi_bm = test_results["Chi"]
        merge!(test_output, Dict("Chi"=> chi))
        @test all(@. (abs(chi/chi_bm-1.0) < 0.0005))
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

    if extensive
        @testset "linear_Pk_As" begin
            cosmo_emul_As = Cosmology(Ωm=0.27, Ωb=0.046, h=0.7, ns=1.0, As=2.097e-9,
                                    nk=300, nz=300, nz_pk=70, tk_mode="emupk")
            cosmo_Bolt_As = Cosmology(Ωm=0.27, Ωb=0.046, h=0.7, ns=1.0, As=2.097e-9,
                                     nk=70, nz=300, nz_pk=70, tk_mode="Bolt")
            ks = exp.(LimberJack.emulator.training_karr)
            pk_emul = nonlin_Pk(cosmo_emul_As, ks, 0.0)
            pk_Bolt = nonlin_Pk(cosmo_Bolt_As, ks, 0.0)
            pk_emul_bm = test_results["pk_emul_As"]
            pk_Bolt_bm = test_results["pk_Bolt_As"]
            merge!(test_output, Dict("pk_emul_As"=> pk_emul))
            merge!(test_output, Dict("pk_Bolt_As"=> pk_Bolt))
            @test all(@. (abs(pk_emul/pk_emul_bm-1.0) <  0.05))
            @test all(@. (abs(pk_Bolt/pk_Bolt_bm-1.0) <  0.05))
        end
    end                 

    @testset "nonlinear_Pk" begin
        ks = exp.(LimberJack.emulator.training_karr)
        pk_EisHu = nonlin_Pk(cosmo_EisHu_nonlin, ks, 0)
        pk_emul = nonlin_Pk(cosmo_emul_nonlin, ks, 0)
        pk_Bolt = nonlin_Pk(cosmo_Bolt_nonlin, ks, 0)
        pk_EisHu_bm = test_results["pk_EisHu_nonlin"]
        pk_emul_bm = test_results["pk_emul_nonlin"]
        pk_Bolt_bm = test_results["pk_Bolt_nonlin"]
        merge!(test_output, Dict("pk_EisHu_nonlin"=> pk_EisHu))
        merge!(test_output, Dict("pk_emul_nonlin"=> pk_emul))
        merge!(test_output, Dict("pk_Bolt_nonlin"=> pk_Bolt))
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk_EisHu/pk_EisHu_bm-1.0) < 0.005))
        @test all(@. (abs(pk_emul/pk_emul_bm-1.0) < 0.05))
        @test all(@. (abs(pk_Bolt/pk_Bolt_bm-1.0) < 0.05))
        
    end
   
    @testset "CreateTracer" begin
        p_of_z(x) = @. exp(-0.5*((x-0.5)/0.05)^2)
        z = Vector(range(0., stop=2., length=200))
        nz = Vector(p_of_z(z))
        t = NumberCountsTracer(cosmo_EisHu, z, nz, b=1.0)
        wz1 = t.wint(cosmo_EisHu.chi(0.5))
        hz = Hmpc(cosmo_EisHu, 0.5)
        wz2 = p_of_z(0.5)*hz/(sqrt(2π)*0.05)
        @test abs(wz2/wz1 - 1) < 0.005
    end

    @testset "EisHu_Cℓs" begin
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo_EisHu, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo_EisHu, z, nz;
                               m=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo_EisHu)
        ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
        Cℓ_gg = angularCℓs(cosmo_EisHu, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo_EisHu, tg, ts, ℓs)
        Cℓ_ss = angularCℓs(cosmo_EisHu, ts, ts, ℓs)
        Cℓ_gk = angularCℓs(cosmo_EisHu, tg, tk, ℓs)
        Cℓ_sk = angularCℓs(cosmo_EisHu, ts, tk, ℓs)
        Cℓ_gg_bm = test_results["cl_gg_eishu"]
        Cℓ_gs_bm = test_results["cl_gs_eishu"]
        Cℓ_ss_bm = test_results["cl_ss_eishu"]
        Cℓ_gk_bm = test_results["cl_gk_eishu"]
        Cℓ_sk_bm = test_results["cl_sk_eishu"]
        merge!(test_output, Dict("cl_gg_eishu"=> Cℓ_gg))
        merge!(test_output, Dict("cl_gs_eishu"=> Cℓ_gs))
        merge!(test_output, Dict("cl_ss_eishu"=> Cℓ_ss))
        merge!(test_output, Dict("cl_gk_eishu"=> Cℓ_sk))
        merge!(test_output, Dict("cl_sk_eishu"=> Cℓ_ss))
        
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.005))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.005))
    end

    @testset "nonlin_EisHu_Cℓs" begin
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo_EisHu, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo_EisHu, z, nz;
                               m=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo_EisHu)
        ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
        Cℓ_gg = angularCℓs(cosmo_EisHu_nonlin, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo_EisHu_nonlin, tg, ts, ℓs)
        Cℓ_ss = angularCℓs(cosmo_EisHu_nonlin, ts, ts, ℓs)
        Cℓ_gk = angularCℓs(cosmo_EisHu_nonlin, tg, tk, ℓs)
        Cℓ_sk = angularCℓs(cosmo_EisHu_nonlin, ts, tk, ℓs)
        Cℓ_gg_bm = test_results["cl_gg_eishu_nonlin"]
        Cℓ_gs_bm = test_results["cl_gs_eishu_nonlin"]
        Cℓ_ss_bm = test_results["cl_ss_eishu_nonlin"]
        Cℓ_gk_bm = test_results["cl_gk_eishu_nonlin"]
        Cℓ_sk_bm = test_results["cl_sk_eishu_nonlin"]
        merge!(test_output, Dict("cl_gg_eishu_nonlin"=> Cℓ_gg))
        merge!(test_output, Dict("cl_gs_eishu_nonlin"=> Cℓ_gs))
        merge!(test_output, Dict("cl_ss_eishu_nonlin"=> Cℓ_ss))
        merge!(test_output, Dict("cl_gk_eishu_nonlin"=> Cℓ_sk))
        merge!(test_output, Dict("cl_sk_eishu_nonlin"=> Cℓ_ss))
        
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.005))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.005))
    end

    @testset "emul_Cℓs" begin
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo_emul, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo_emul, z, nz;
                               m=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo_emul)
        ℓs = [10.0, 30.0, 100.0, 300.0]
        Cℓ_gg = angularCℓs(cosmo_emul, tg, tg, ℓs) 
        Cℓ_gs = angularCℓs(cosmo_emul, tg, ts, ℓs)
        Cℓ_ss = angularCℓs(cosmo_emul, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo_emul, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo_emul, ts, tk, ℓs) 
        Cℓ_gg_bm = test_results["cl_gg_camb"]
        Cℓ_gs_bm = test_results["cl_gs_camb"]
        Cℓ_ss_bm = test_results["cl_ss_camb"]
        Cℓ_gk_bm = test_results["cl_gk_camb"]
        Cℓ_sk_bm = test_results["cl_sk_camb"]
        merge!(test_output, Dict("cl_gg_camb"=> Cℓ_gg))
        merge!(test_output, Dict("cl_gs_camb"=> Cℓ_gs))
        merge!(test_output, Dict("cl_ss_camb"=> Cℓ_ss))
        merge!(test_output, Dict("cl_gk_camb"=> Cℓ_gk))
        merge!(test_output, Dict("cl_sk_camb"=> Cℓ_sk))
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.05))
        # The ℓ=10 point is a bit inaccurate for some reason
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.05))
    end

    @testset "EisHu_Halo_Cℓs" begin
        z = Vector(range(0.0, stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo_EisHu_nonlin, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo_EisHu_nonlin, z, nz;
                               m=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo_EisHu_nonlin)
        ℓs = [10.0, 30.0, 100.0, 300.0, 1000.0]
        Cℓ_gg = angularCℓs(cosmo_EisHu_nonlin, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo_EisHu_nonlin, tg, ts, ℓs) 
        Cℓ_ss = angularCℓs(cosmo_EisHu_nonlin, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo_EisHu_nonlin, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo_EisHu_nonlin, ts, tk, ℓs)
        
        Cℓ_gg_bm = test_results["cl_gg_eishu_nonlin"]
        Cℓ_gs_bm = test_results["cl_gs_eishu_nonlin"]
        Cℓ_ss_bm = test_results["cl_ss_eishu_nonlin"]
        Cℓ_gk_bm = test_results["cl_gk_eishu_nonlin"]
        Cℓ_sk_bm = test_results["cl_sk_eishu_nonlin"]
        merge!(test_output, Dict("cl_gg_eishu_nonlin"=> Cℓ_gg))
        merge!(test_output, Dict("cl_gs_eishu_nonlin"=> Cℓ_gs))
        merge!(test_output, Dict("cl_ss_eishu_nonlin"=> Cℓ_ss))
        merge!(test_output, Dict("cl_gk_eishu_nonlin"=> Cℓ_sk))
        merge!(test_output, Dict("cl_sk_eishu_nonlin"=> Cℓ_ss))
        
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.005))
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.005))
    end

    @testset "emul_Halo_Cℓs" begin
        z = Vector(range(0., stop=2., length=256))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg = NumberCountsTracer(cosmo_emul_nonlin, z, nz; b=1.0)
        ts = WeakLensingTracer(cosmo_emul_nonlin, z, nz;
                               m=0.0,
                               IA_params=[0.0, 0.0])
        tk = CMBLensingTracer(cosmo_emul_nonlin)
        ℓs =  Vector(range(10., stop=1000., length=100))
        Cℓ_gg = angularCℓs(cosmo_emul_nonlin, tg, tg, ℓs)
        Cℓ_gs = angularCℓs(cosmo_emul_nonlin, tg, ts, ℓs) 
        Cℓ_ss = angularCℓs(cosmo_emul_nonlin, ts, ts, ℓs) 
        Cℓ_gk = angularCℓs(cosmo_emul_nonlin, tg, tk, ℓs) 
        Cℓ_sk = angularCℓs(cosmo_emul_nonlin, ts, tk, ℓs)
        Cℓ_gg_bm = test_results["cl_gg_camb_nonlin"]
        Cℓ_gs_bm = test_results["cl_gs_camb_nonlin"]
        Cℓ_ss_bm = test_results["cl_ss_camb_nonlin"]
        Cℓ_gk_bm = test_results["cl_gk_camb_nonlin"]
        Cℓ_sk_bm = test_results["cl_sk_camb_nonlin"]
        merge!(test_output, Dict("cl_gg_emul_nonlin"=> Cℓ_gg))
        merge!(test_output, Dict("cl_gs_emul_nonlin"=> Cℓ_gs))
        merge!(test_output, Dict("cl_ss_emul_nonlin"=> Cℓ_ss))
        merge!(test_output, Dict("cl_gk_emul_nonlin"=> Cℓ_gk))
        merge!(test_output, Dict("cl_sk_emul_nonlin"=> Cℓ_sk))
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(Cℓ_gg/Cℓ_gg_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gs/Cℓ_gs_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_ss/Cℓ_ss_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_gk/Cℓ_gk_bm-1.0) < 0.05))
        @test all(@. (abs(Cℓ_sk/Cℓ_sk_bm-1.0) < 0.05))
    end

    @testset "IsBaseDiff" begin
        if extensive
            zs = LinRange(0.01, 3.0, 100)
        else    
            zs = [0.1, 0.5, 1.0, 3.0]
        end

        function f(p)
            cosmo = LimberJack.Cosmology(Ωm=p)
            chi = LimberJack.comoving_radial_distance(cosmo, zs)
            return chi
        end

        function g(p)
            cosmo = LimberJack.Cosmology(Ωm=p)
            chi = LimberJack.growth_factor(cosmo, zs)
            return chi
        end

        Ωm0 = 0.3
        dΩm = 0.02

        fs = ForwardDiff.derivative(f, Ωm0)
        gs = ForwardDiff.derivative(g, Ωm0)
        fs_bm = (f(Ωm0+dΩm)-f(Ωm0-dΩm))/2dΩm
        gs_bm = (g(Ωm0+dΩm)-g(Ωm0-dΩm))/2dΩm

        merge!(test_output, Dict("Chi_autodiff"=> fs))
        merge!(test_output, Dict("Dz_autodiff"=> gs))
        merge!(test_output, Dict("Chi_num"=> fs_bm))
        merge!(test_output, Dict("Dz_num"=> gs_bm))

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
            cosmo = Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="linear")
            pk = lin_Pk(cosmo, ks, 0.)
            return pk
        end

        function lin_emul(p)
            cosmo = Cosmology(Ωm=p, tk_mode="emupk", Pk_mode="linear")
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

    if false #extensive
        @testset "IsBoltPkDiff" begin
            logk = range(log(0.0001), stop=log(100.0), length=20)
            ks = exp.(logk)

            function lin_Bolt(p)
                cosmo = Cosmology(Ωm=p, tk_mode="Bolt", Pk_mode="linear")
                pk = lin_Pk(cosmo, ks, 0.)
                return pk
            end

            Ωm0 = 0.25
            dΩm = 0.01

            lin_Bolt_autodiff = abs.(ForwardDiff.derivative(lin_Bolt, Ωm0))
            lin_Bolt_num = abs.((lin_Bolt(Ωm0+dΩm)-lin_Bolt(Ωm0-dΩm))/(2dΩm))

            merge!(test_output, Dict("lin_Bolt_autodiff"=> lin_Bolt_autodiff))
            merge!(test_output, Dict("lin_Bolt_num"=> lin_Bolt_num))        
            # Median needed since errors shoot up when derivatieve
            # crosses zero
            @test median(lin_Bolt_autodiff./lin_Bolt_num.-1) < 0.05
        end
    end    
    

    @testset "IsNonlinPkDiff" begin
        if extensive
            logk = range(log(0.0001), stop=log(100.0), length=500)
            ks = exp.(logk)
        else
            ks = exp.(LimberJack.emulator.training_karr)
        end

                                                
        function nonlin_EisHu(p)
            cosmo = Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="Halofit")
            pk = nonlin_Pk(cosmo, ks, 0.)
            return pk
        end

        #=
        function nonlin_emul(p)
            cosmo = Cosmology(Ωm=p, tk_mode="emupk", Pk_mode="Halofit")
            pk = nonlin_Pk(cosmo, ks, 0.)
            return pk
        end

        function nonlin_Bolt(p)
            cosmo = Cosmology(Ωm=p, tk_mode="Bolt", Pk_mode="Halofit")
            pk = nonlin_Pk(cosmo, ks, 0.)
            return pk
        end
        =#

        Ωm0 = 0.25
        dΩm = 0.01

        nonlin_EisHu_autodiff = abs.(ForwardDiff.derivative(nonlin_EisHu, Ωm0))
        #nonlin_emul_autodiff = abs.(ForwardDiff.derivative(nonlin_emul, Ωm0))
        #nonlin_Bolt_autodiff = abs.(ForwardDiff.derivative(nonlin_Bolt, Ωm0))
        nonlin_EisHu_num = abs.((nonlin_EisHu(Ωm0+dΩm)-nonlin_EisHu(Ωm0-dΩm))/(2dΩm))
        #nonlin_emul_num = abs.((nonlin_emul(Ωm0+dΩm)-nonlin_emul(Ωm0-dΩm))/(2dΩm));
        #nonlin_Bolt_num = abs.((nonlin_Bolt(Ωm0+dΩm)-nonlin_Bolt(Ωm0-dΩm))/(2dΩm));
                                                
        merge!(test_output, Dict("nonlin_EisHu_autodiff"=> nonlin_EisHu_autodiff))
        #merge!(test_output, Dict("nonlin_emul_autodiff"=> nonlin_emul_autodiff))
        #merge!(test_output, Dict("nonlin_Bolt_autodiff"=> nonlin_Bolt_autodiff))
        merge!(test_output, Dict("nonlin_EisHu_num"=> nonlin_EisHu_num))
        #merge!(test_output, Dict("nonlin_emul_num"=> nonlin_emul_num))
        #merge!(test_output, Dict("nonlin_Bolt_num"=> nonlin_Bolt_num))
        # Median needed since errors shoot up when derivatieve
        # crosses zero
        @test median(nonlin_EisHu_autodiff./nonlin_EisHu_num.-1) < 0.1
        #@test median(nonlin_emul_autodiff./nonlin_emul_num.-1) < 0.1
        #@test median(nonlin_Bolt_autodiff./nonlin_Bolt_num.-1) < 0.1
    end
    


    @testset "AreClsDiff" begin
        
        if extensive
            ℓs = 10 .^ Vector(LinRange(2, 3, 100))
        else
            ℓs = [10.0, 30.0, 100.0, 300.0]
        end

        function Cl_gg(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
            Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
            return Cℓ_gg
        end
        
        function Cl_gs(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
            ts = WeakLensingTracer(cosmo, z, nz;
                                   m=0.0,
                                   IA_params=[0.0, 0.0])
            Cℓ_gs = angularCℓs(cosmo, tg, ts, ℓs) 
            return Cℓ_gs
        end

        function Cl_ss(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="Halofit")
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            ts = WeakLensingTracer(cosmo, z, nz;
                                   m=0.0,
                                   IA_params=[0.0, 0.0])
            Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_ss
        end
        
        function Cl_sk(p::T)::Array{T,1} where T<:Real
            cosmo = LimberJack.Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="Halofit")
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
            cosmo = LimberJack.Cosmology(Ωm=p, tk_mode="EisHu", Pk_mode="Halofit")
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            tg = NumberCountsTracer(cosmo, z, nz; b=1.0)
            tk = CMBLensingTracer(cosmo)
            Cℓ_gk = angularCℓs(cosmo, tg, tk, ℓs)
            return Cℓ_gk
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

        merge!(test_output, Dict("Cl_gg_autodiff"=> Cl_gg_autodiff))
        merge!(test_output, Dict("Cl_gs_autodiff"=> Cl_gs_autodiff))
        merge!(test_output, Dict("Cl_ss_autodiff"=> Cl_ss_autodiff))
        merge!(test_output, Dict("Cl_sk_autodiff"=> Cl_sk_autodiff))
        merge!(test_output, Dict("Cl_gk_autodiff"=> Cl_gk_autodiff))
        merge!(test_output, Dict("Cl_gg_num"=> Cl_gg_num))
        merge!(test_output, Dict("Cl_gs_num"=> Cl_gs_num))
        merge!(test_output, Dict("Cl_ss_num"=> Cl_ss_num))
        merge!(test_output, Dict("Cl_sk_num"=> Cl_sk_num))
        merge!(test_output, Dict("Cl_gk_num"=> Cl_gk_num))

        @test all(@. (abs(Cl_gg_autodiff/Cl_gg_num-1) < 0.05))
        @test all(@. (abs(Cl_gs_autodiff/Cl_gs_num-1) < 0.05))
        @test all(@. (abs(Cl_ss_autodiff/Cl_ss_num-1) < 0.05))
        @test all(@. (abs(Cl_sk_autodiff/Cl_sk_num-1) < 0.05))
        @test all(@. (abs(Cl_gk_autodiff/Cl_gk_num-1) < 0.05))
    end

    @testset "Nuisances" begin
        z = Vector(range(0.01, stop=2., length=1024))
        nz = @. exp(-0.5*((z-0.5)/0.05)^2)
        tg_b = NumberCountsTracer(cosmo_EisHu_nonlin, z, nz; b=2.0)
        ts_m = WeakLensingTracer(cosmo_EisHu_nonlin, z, nz; m=1.0, IA_params=[0.0, 0.0])
        ts_IA = WeakLensingTracer(cosmo_EisHu_nonlin, z, nz; m=0.0, IA_params=[0.1, 0.1])
        ℓs = [10.0, 30.0, 100.0, 300.0]
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
            cosmo = Cosmology(tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = Vector(range(0., stop=2., length=256))
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; b=p)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
            return Cℓ_gg
        end
        
        function dz(p::T)::Array{T,1} where T<:Real
            cosmo = Cosmology(tk_mode="EisHu", Pk_mode="Halofit", nz=300)
            cosmo.settings.cosmo_type = typeof(p)
            z = Vector(range(0., stop=2., length=256)) .- p
            nz = Vector(@. exp(-0.5*((z-0.5)/0.05)^2))
            tg = NumberCountsTracer(cosmo, z, nz; b=1)
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_gg = angularCℓs(cosmo, tg, tg, ℓs) 
            return Cℓ_gg
        end
        
        function mbias(p::T)::Array{T,1} where T<:Real
            cosmo = Cosmology(tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz; m=p, IA_params=[0.0, 0.0])
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_sk = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_sk
        end
        
        function IA_A(p::T)::Array{T,1} where T<:Real
            cosmo = Cosmology(tk_mode="EisHu", Pk_mode="Halofit")
            cosmo.settings.cosmo_type = typeof(p)
            z = range(0., stop=2., length=256)
            nz = @. exp(-0.5*((z-0.5)/0.05)^2)
            ts = WeakLensingTracer(cosmo, z, nz; m=2, IA_params=[p, 0.1])
            ℓs = [10.0, 30.0, 100.0, 300.0]
            Cℓ_ss = angularCℓs(cosmo, ts, ts, ℓs)
            return Cℓ_ss
        end
        
        function IA_alpha(p::T)::Array{T,1} where T<:Real
            cosmo = Cosmology(tk_mode="EisHu", Pk_mode="Halofit")
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
    npzwrite("test_output.npz", test_output)
end
