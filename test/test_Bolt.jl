using LimberJack
using ForwardDiff
using NPZ

test_output = Dict{String}{Vector}()
logk = range(log(0.0001), stop=log(100.0), length=100)
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

npzwrite("Bolt_test_output.npz", test_output)