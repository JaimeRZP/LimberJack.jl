var documenterSearchIndex = {"docs":
[{"location":"api/#LimberJack.jl","page":"API","title":"LimberJack.jl","text":"","category":"section"},{"location":"api/#Core","page":"API","title":"Core","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Core performs the main computations of LimberJack.jl.  When using LimberJack.jl, the first step is to create an instance of the  Cosmology structure. This is as easy as calling:","category":"page"},{"location":"api/","page":"API","title":"API","text":"    using LimberJack\n    cosmology = Cosmology()","category":"page"},{"location":"api/","page":"API","title":"API","text":"This will generate the an instance of Cosmology given the vanilla LambdaCDM cosmology of CCL. Cosmology() then computes the value of the comoving distance, the growth factor, the growth rate and matter power spectrum at an array of values and generates interpolators for said quantites.  The user can acces the value of these interpolator at an arbitrary input using the public functions of the model. Moreover, the Cosmology structure has its own type called Cosmology.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LimberJack.Settings\nLimberJack.CosmoPar\nLimberJack.Cosmology\nLimberJack.Ez\nLimberJack.Hmpc\nLimberJack.comoving_radial_distance\nLimberJack.growth_factor\nLimberJack.growth_rate\nLimberJack.sigma8\nLimberJack.fs8\nLimberJack.lin_Pk\nLimberJack.nonlin_Pk","category":"page"},{"location":"api/#LimberJack.Settings","page":"API","title":"LimberJack.Settings","text":"Settings(;kwargs...)\n\nConstructor of settings structure constructor. \n\nKwargs:\n\nnz::Int=300 : number of nodes in the general redshift array.\nnz_chi::Int=1000 : number of nodes in the redshift array used to compute matter power spectrum grid.\nnz_t::Int=350 : number of nodes in the general redshift array.\nnk::Int=500: number of nodes in the k-scale array used to compute matter power spectrum grid.\nnℓ::Int=300: number of nodes in the multipoles array.\nusing_As::Bool=false: True if using the As parameter.\ncosmo_type::Type=Float64 : type of cosmological parameters. \ntk_mode::String=:EisHu : choice of transfer function.\nDz_mode::String=:RK2 : choice of method to compute the linear growth factor.\nPk_mode::String=:linear : choice of method to apply non-linear corrections to the matter power spectrum.\n\nReturns:\n\nmutable struct Settings\n    nz::Int\n    nz_chi::Int\n    nz_t::Int\n    nk::Int\n    nℓ::Int\n\n    xs\n    zs\n    zs_t\n    ks\n    ℓs\n    logk\n    dlogk\n\n    using_As::Bool\n\n    cosmo_type::DataType\n    tk_mode::Symbol\n    Dz_mode::Symbol\n    Pk_mode::Symbol\nend        \n\n\n\n\n\n","category":"type"},{"location":"api/#LimberJack.CosmoPar","page":"API","title":"LimberJack.CosmoPar","text":"CosmoPar(kwargs...)\n\nCosmology parameters structure constructor.  \n\nKwargs:\n\nΩm::Dual=0.30 : cosmological matter density. \nΩb::Dual=0.05 : cosmological baryonic density.\nh::Dual=0.70 : reduced Hubble parameter.\nns::Dual=0.96 : Harrison-Zeldovich spectral index.\nAs::Dual=2.097e-9 : Harrison-Zeldovich spectral amplitude.\nσ8::Dual=0.81: variance of the matter density field in a sphere of 8 Mpc.\nY_p::Dual=0.24: primordial helium fraction.\nN_ν::Dual=3.046: effective number of relativisic species (PDG25 value).\nΣm_ν::Dual=0.0: sum of neutrino masses (eV), Planck 15 default ΛCDM value.\nθCMB::Dual=2.725/2.7: CMB temperature over 2.7.\nΩg::Dual=2.38163816E-5*θCMB^4/h^2: cosmological density of relativistic species.\nΩr::Dual=Ωg*(1.0 + N_ν * (7.0/8.0) * (4.0/11.0)^(4.0/3.0)): cosmological radiation density.\n\nReturns:\n\nmutable struct CosmoPar{T}\n    Ωm::T\n    Ωb::T\n    h::T\n    ns::T\n    As::T\n    σ8::T\n    θCMB::T\n    Y_p::T\n    N_ν::T\n    Σm_ν::T\n    Ωg::T\n    Ωr::T\n    Ωc::T\n    ΩΛ::T\nend     \n\n\n\n\n\n","category":"type"},{"location":"api/#LimberJack.Cosmology","page":"API","title":"LimberJack.Cosmology","text":"Cosmology(cpar::CosmoPar, settings::Settings)\n\nBase cosmology structure constructor.\n\nCalculates the LCDM expansion history based on the different species densities provided in CosmoPar.\n\nThe comoving distance is then calculated integrating the expansion history. \n\nDepending on the choice of transfer function in the settings, the primordial power spectrum is calculated using: \n\ntk_mode = :EisHu : the Eisenstein & Hu formula (arXiv:astro-ph/9710252)\ntk_mode = :EmuPk : the Mootoovaloo et al 2021 emulator EmuPk (arXiv:2105.02256v2) \n\nDepending on the choice of power spectrum mode in the settings, the matter power spectrum is either: \n\nPk_mode = :linear : the linear matter power spectrum.\nPk_mode = :halofit : the Halofit non-linear matter power spectrum (arXiv:astro-ph/0207664).\n\nArguments:\n\nSettings::MutableStructure : cosmology constructure settings. \nCosmoPar::Structure : cosmological parameters.\n\nReturns:\n\nmutable struct Cosmology\n    settings::Settings\n    cpar::CosmoPar\n    chi::AbstractInterpolation\n    z_of_chi::AbstractInterpolation\n    t_of_z::AbstractInterpolation\n    chi_max\n    chi_LSS\n    Dz::AbstractInterpolation\n    fs8z::AbstractInterpolation\n    PkLz0::AbstractInterpolation\n    Pk::AbstractInterpolation\nend     \n\n\n\n\n\nCosmology(;kwargs...)\n\nShort form to call Cosmology(cpar::CosmoPar, settings::Settings). kwargs are passed to the constructors of CosmoPar and Settings. Returns:\n\nCosmology : cosmology structure.\n\n\n\n\n\n","category":"type"},{"location":"api/#LimberJack.Ez","page":"API","title":"LimberJack.Ez","text":"Ez(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns the expansion rate (H(z)/H0). \n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nz::Dual : redshift\n\nReturns:\n\nEz::Dual : expansion rate \n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.Hmpc","page":"API","title":"LimberJack.Hmpc","text":"Hmpc(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns the expansion history (H(z)) in Mpc. \n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nz::Dual : redshift\n\nReturns:\n\nHmpc::Dual : expansion rate \n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.comoving_radial_distance","page":"API","title":"LimberJack.comoving_radial_distance","text":"comoving_radial_distance(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns the comoving radial distance. \n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nz::Dual : redshift\n\nReturns:\n\nChi::Dual : comoving radial distance\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.growth_factor","page":"API","title":"LimberJack.growth_factor","text":"growth_factor(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns the growth factor (D(z) = log(δ)). \n\nArguments:\n\ncosmo::Cosmology : cosmological structure\nz::Dual : redshift\n\nReturns:\n\nDz::Dual : comoving radial distance\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.growth_rate","page":"API","title":"LimberJack.growth_rate","text":"growth_rate(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns growth rate. \n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nz::Dual : redshift\n\nReturns:\n\nf::Dual : f\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.sigma8","page":"API","title":"LimberJack.sigma8","text":"sigma8(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns s8. \n\nArguments:\n\ncosmo::Cosmology : cosmological structure\nz::Dual : redshift\n\nReturns:\n\ns8::Dual : comoving radial distance\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.fs8","page":"API","title":"LimberJack.fs8","text":"fs8(cosmo::Cosmology, z)\n\nGiven a Cosmology instance, it returns fs8. \n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nz::Dual : redshift\n\nReturns:\n\nfs8::Dual : fs8\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.lin_Pk","page":"API","title":"LimberJack.lin_Pk","text":"lin_Pk(cosmo::Cosmology, k, z)\n\nGiven a Cosmology instance, it returns the linear matter power spectrum (P(k,z))\n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nk::Dual : scale\nz::Dual : redshift\n\nReturns:\n\nPk::Dual : linear matter power spectrum\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.nonlin_Pk","page":"API","title":"LimberJack.nonlin_Pk","text":"nonlin_Pk(cosmo::Cosmology, k, z)\n\nGiven a Cosmology instance, it returns the non-linear matter power spectrum (P(k,z)) using the Halofit fitting formula (arXiv:astro-ph/0207664). \n\nArguments:\n\ncosmo::Cosmology : cosmology structure\nk::Dual : scale\nz::Dual : redshift\n\nReturns:\n\nPk::Dual : non-linear matter power spectrum\n\n\n\n\n\n","category":"function"},{"location":"api/#Tracers","page":"API","title":"Tracers","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"In LimberJack.jl each tracer is structure containing at least of three fields:","category":"page"},{"location":"api/","page":"API","title":"API","text":"wint: an interpolator between chis and warr.\nF: an interpolator between chis and warr.","category":"page"},{"location":"api/","page":"API","title":"API","text":"On top of these, tracers might contain the value of any nuisance parameter associated with them. Moreover, tracers within LimberJack.jl tracer objects have their own type called Tracer.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LimberJack.NumberCountsTracer\nLimberJack.WeakLensingTracer\nLimberJack.CMBLensingTracer\nLimberJack.get_IA","category":"page"},{"location":"api/#LimberJack.NumberCountsTracer","page":"API","title":"LimberJack.NumberCountsTracer","text":"NumberCountsTracer(warr, chis, wint, b, lpre)\n\nNumber counts tracer structure.  Arguments:\n\nwint::Interpolation : interpolation of the radial kernel over comoving distance.\nF::Function : prefactor.\n\nReturns:\n\nNumberCountsTracer::NumberCountsTracer : Number counts tracer structure.\n\n\n\n\n\n","category":"type"},{"location":"api/#LimberJack.WeakLensingTracer","page":"API","title":"LimberJack.WeakLensingTracer","text":"WeakLensingTracer(cosmo::Cosmology, z_n, nz; kwargs...)\n\nWeak lensing tracer structure constructor.  Arguments:\n\ncosmo::Cosmology : cosmology structure. \nz_n::Vector{Dual} : redshift array.\nnz::Interpolation : distribution of sources over redshift.\n\nKwargs:\n\nmb::Dual = 1 : multiplicative bias. \nIA_params::Vector{Dual} = [A_IA, alpha_IA]: instrinsic aligment parameters.\n\nReturns:\n\nWeakLensingTracer::WeakLensingTracer : Weak lensing tracer structure.\n\n\n\n\n\n","category":"type"},{"location":"api/#LimberJack.CMBLensingTracer","page":"API","title":"LimberJack.CMBLensingTracer","text":"CMBLensingTracer(warr, chis, wint, lpre)\n\nCMB lensing tracer structure.  Arguments:\n\nwint::Interpolation : interpolation of the radial kernel over comoving distance.\nF::Function : prefactor.\n\nReturns:\n\nCMBLensingTracer::CMBLensingTracer : CMB lensing tracer structure.\n\n\n\n\n\n","category":"type"},{"location":"api/#LimberJack.get_IA","page":"API","title":"LimberJack.get_IA","text":"get_IA(cosmo::Cosmology, zs, IA_params)\n\nCMB lensing tracer structure.  Arguments:\n\ncosmo::Cosmology : cosmology structure.\nzs::Vector{Dual} : redshift array.\nIA_params::Vector{Dual} : Intrinsic aligment parameters.\n\nReturns:\n\nIA_corr::Vector{Dual} : Intrinsic aligment correction to radial kernel.\n\n\n\n\n\n","category":"function"},{"location":"api/#Spectra","page":"API","title":"Spectra","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Performs the computation of the angular power spectra of any two tracers.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LimberJack.Cℓintegrand\nLimberJack.angularCℓs","category":"page"},{"location":"api/#LimberJack.Cℓintegrand","page":"API","title":"LimberJack.Cℓintegrand","text":"Cℓintegrand(cosmo::Cosmology, t1::Tracer, t2::Tracer, logk, ℓ)\n\nReturns the integrand of the angular power spectrum.  Arguments:\n\ncosmo::Cosmology : cosmology structure.\nt1::Tracer : tracer structure.\nt2::Tracer : tracer structure.\nlogk::Vector{Float} : log scale array.\nℓ::Float : multipole.\n\nReturns:\n\nintegrand::Vector{Real} : integrand of the angular power spectrum.\n\n\n\n\n\n","category":"function"},{"location":"api/#LimberJack.angularCℓs","page":"API","title":"LimberJack.angularCℓs","text":"angularCℓs(cosmo::Cosmology, t1::Tracer, t2::Tracer, ℓs)\n\nReturns the angular power spectrum.  Arguments:\n\ncosmo::Cosmology : cosmology structure.\nt1::Tracer : tracer structure.\nt2::Tracer : tracer structure.\nℓs::Vector{Float} : multipole array.\n\nReturns:\n\nCℓs::Vector{Real} : angular power spectrum.\n\n\n\n\n\n","category":"function"},{"location":"api/#Data-Utils","page":"API","title":"Data Utils","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Parses sacc  and YAML files.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LimberJack.make_data","category":"page"},{"location":"api/#Theory","page":"API","title":"Theory","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Performs the computation of complex theory vectors given sacc  and YAML files.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LimberJack.Theory","category":"page"},{"location":"api/#LimberJack.Theory","page":"API","title":"LimberJack.Theory","text":"Theory(cosmology::Cosmology,\n       instructions::Instructions, files;\n       Nuisances=Dict())\n\nComposes a theory vector given a Cosmology object, a Meta objectm, a files npz file and a dictionary of nuisance parameters.\n\nArguments:\n\ncosmology::Cosmology : Cosmology object.\nmeta::Meta : Meta object.\nfiles : files npz file.\nNuisances::Dict : dictonary of nuisace parameters. \n\nReturns:\n\nMeta: structure\n\nstruct Meta\n    names : names of tracers.\n    pairs : pairs of tracers to compute angular spectra for.\n    types : types of the tracers.\n    idx : positions of cls in theory vector.\n    data : data vector.\n    cov : covariance of the data.\n    inv_cov : inverse covariance of the data.\nend\n\nfiles: npz file\n\n\n\n\n\n","category":"function"},{"location":"#LimberJack.jl","page":"Home","title":"LimberJack.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Build Status) (Image: Dev) (Image: size)","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"<p align=\"center\"> A differentiable cosmological code in Julia. </p>","category":"page"},{"location":"#Design-Philosophy","page":"Home","title":"Design Philosophy","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modularity: each main function within LimberJack.jl has its own module. New functions can be added by including extra modules. LimberJack.jl has the following modules:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Module function\nboltzmann.jl Performs the computation of primordial power spectrum\ncore.jl Defines the structures where the theoretical predictions are stored and computes the background quantities\ndata_utils.jl Manages sacc files for large data vectors\ngrowth.jl Computes the growth factor\nhalofit.jl Computes the non-linear matter power spectrum as given by the Halofit fitting formula\nspectra.jl Computes the power spectra of any two tracers\ntheory.jl Computes large data vectors that combine many spectra\ntracers.jl Computes the kernels associated with each type of kernel","category":"page"},{"location":"","page":"Home","title":"Home","text":"Object-oriented: LimberJack.jl mimics CCL.py class structure by using Julia's structures.\nTransparency: LimberJack.jl is fully written in Julia without needing to interface to any other programming language (C, Python...) to compute thoretical predictions. This allows the user full access to the code from input to output.","category":"page"},{"location":"#Goals","page":"Home","title":"Goals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Gradients: one order of magnitude faster gradients than finite differences.\nPrecision: sub-percentage error with respect to CCL.\nSpeed: C-like performance.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In order to run LimberJack.jl you will need Julia-1.7.0 or newer installed in your system. Older versions of Julia might be compatible but haven't been tested. You can find instructions on how to install Julia here: https://julialang.org/downloads/.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Once you have installed Julia you can install LimberJack.jl following these steps:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Clone the git repository\nFrom the repository directory open Julia\nIn the Julia command line run:","category":"page"},{"location":"","page":"Home","title":"Home","text":"    using Pkg\n    Pkg.add(\"LimberJack\")","category":"page"},{"location":"#Installing-Sacc.py-in-Julia","page":"Home","title":"Installing Sacc.py in Julia","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"    using Pkg\n    Pkg.add(\"CondaPkg\")\n    CondaPkg.add(\"sacc\")","category":"page"},{"location":"#Use","page":"Home","title":"Use","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"    # Import\n    using LimberJack\n    \n    # create LimberJack.jl Cosmology instance\n    cosmology = Cosmology(Ωm=0.30, Ωb=0.05, h=0.70, ns=0.96, s8=0.81;\n                          tk_mode=:EisHu,\n                          Pk_mode=:Halofit)\n    \n    z = Vector(range(0., stop=2., length=256))\n    nz = @. exp(-0.5*((z-0.5)/0.05)^2)\n    tracer = NumberCountsTracer(cosmology, zs, nz; b=1.0)\n    ls = [10.0, 30.0, 100.0, 300.0]\n    cls = angularCℓs(cosmology, tracer, tracer, ls)","category":"page"},{"location":"#Challenges","page":"Home","title":"Challenges","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Parallelization: the current threading parallelization of LimberJack.jl is far away from the optimal one over number of threads scaling. Future works could study alternative parallalization schemes or possible ineficiencies in the code. \nGPU's: LimberJack.jl currently cannot run on GPUs which are known to significantly speed-up cosmological inference. Future works could study implementing Julia GPU libraries such as CUDA.jl.\nBackwards-AD: currently LimberJack.jl's preferred AD mode is forward-AD. However, the key computation of cosmological inference, obtaining the chi^2, is a map from N parameters to a scalar. For a large number of parameters, backwards-AD is in theory the preferred AD mode and should significantly speed up the computation of the gradient. Future works could look into making LimberJack.jl compatible with the latest Julia AD libraries such as Zygote.jl to implement efficient backwards-AD.","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Please find a tutorial on how to use LimberJack.jl here","category":"page"}]
}
