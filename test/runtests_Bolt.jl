using Test
using LimberJack
using ForwardDiff
using NPZ
using Statistics  

test_results = npzread("test_results.npz")
test_cls = npzread("test_cls.npz")["cls"]
test_cls_files = npzread("test_cls_files.npz")
test_output = Dict{String}{Vector}()

cosmo_Bolt_As = Cosmology(Ωm=0.27, Ωb=0.046, h=0.7, ns=1.0, As=2.097e-9,
                          nk=70, nz=300, nz_pk=70, tk_mode="Bolt")
cosmo_Bolt_nonlin = Cosmology(Ωm=0.27, Ωb=0.046, h=0.70, ns=1.0, σ8=0.81,
                              nk=70, nz=300, nz_pk=70,
                              tk_mode="Bolt", Pk_mode="Halofit")

@testset "All tests" begin
    @testset "linear_Pk_As" begin
        ks = exp.(LimberJack.emulator.training_karr)
        pk_Bolt = nonlin_Pk(cosmo_Bolt_As, ks, 0.0)
        pk_Bolt_bm = test_results["pk_Bolt_As"]
        merge!(test_output, Dict("pk_Bolt_As"=> pk_Bolt))
        @test all(@. (abs(pk_Bolt/pk_Bolt_bm-1.0) <  0.05))
    end              

    @testset "nonlinear_Pk" begin
        ks = exp.(LimberJack.emulator.training_karr)
        pk_Bolt = nonlin_Pk(cosmo_Bolt_nonlin, ks, 0)
        pk_Bolt_bm = test_results["pk_Bolt_nonlin"]
        merge!(test_output, Dict("pk_Bolt_nonlin"=> pk_Bolt))
        # It'd be best if this was < 1E-4...
        @test all(@. (abs(pk_Bolt/pk_Bolt_bm-1.0) < 0.05))
        
    end

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

    npzwrite("Bolt_test_output.npz", test_output)
end
