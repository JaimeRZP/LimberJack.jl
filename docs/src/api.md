# LimberJack.jl

## Core

```Core``` performs the main computations of ```LimberJack.jl```. 
When using ```LimberJack.jl```, the first step is to create an instance of the  ```Cosmology``` structure.
This is as easy as calling:

```julia
    using LimberJack
    cosmology = Cosmology()
```

This will generate the an instance of ```Cosmology``` given the vanilla $\Lambda$CDM cosmology of ```CCL ```.
```Cosmology()``` then computes the value of the comoving distance, the growth factor, the growth rate and matter power spectrum at an array of values and generates interpolators for said quantites. 
The user can acces the value of these interpolator at an arbitrary input using the public functions of the model.
Moreover, the ```Cosmology``` structure has its own type called ```Cosmology```.

```@docs
LimberJack.Settings
LimberJack.CosmoPar
LimberJack.Cosmology
LimberJack.Ez
LimberJack.Hmpc
LimberJack.comoving_radial_distance
LimberJack.growth_factor
LimberJack.growth_rate
LimberJack.sigma8
LimberJack.fs8
LimberJack.lin_Pk
LimberJack.nonlin_Pk
```

## Emulator

Implements the Mootoovaloo et al (2021) emulator (https://arxiv.org/abs/2105.02256) for the primordial matter power spectrum.
The emulator is imported as a linear model stored in ```emulator/files.npz```.

```@docs
LimberJack.Emulator
LimberJack.get_emulated_log_pk0
```

## Halofit

Implements the Halotfit fitting formula for the non-linear matter power spectrum as described as Smith et al (2003) (https://arxiv.org/abs/astro-ph/0207664)

```@docs
LimberJack.get_PKnonlin
```

## Spectra

Performs the computation of the angular power spectra of any two tracers.

```@docs
LimberJack.Cℓintegrand
LimberJack.angularCℓs
```

## Tracers

In ```LimberJack.jl``` each tracer is ```structure``` containing at least of three fields:
+ ```warr```: an array of values of the kernel of the tracer.
+ ```chis```: the array of comoving distances at which ```warr``` was associated.
+ ```wint```: an interpolator between ```chis``` and ```warr```.
On top of these, tracers might contain the value of any nuisance parameter associated with them.
Moreover, tracers within ```LimberJack.jl``` tracer objects have their own type called ```Tracer```.

```@docs
LimberJack.NumberCountsTracer
LimberJack.WeakLensingTracer
LimberJack.CMBLensingTracer
```
