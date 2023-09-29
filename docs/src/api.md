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
## Tracers

In ```LimberJack.jl``` each tracer is ```structure``` containing at least of three fields:
+ ```wint```: an interpolator between ```chis``` and ```warr```.
+ ```F```: an interpolator between ```chis``` and ```warr```.
On top of these, tracers might contain the value of any nuisance parameter associated with them.
Moreover, tracers within ```LimberJack.jl``` tracer objects have their own type called ```Tracer```.

```@docs
LimberJack.NumberCountsTracer
LimberJack.WeakLensingTracer
LimberJack.CMBLensingTracer
LimberJack.get_IA
```

## Spectra

Performs the computation of the angular power spectra of any two tracers.

```@docs
LimberJack.Cℓintegrand
LimberJack.angularCℓs
```
## Data Utils

Parses ```sacc```  and ```YAML``` files.

```@docs
LimberJack.make_data
```

## Theory

Performs the computation of complex theory vectors given ```sacc```  and ```YAML``` files.

```@docs
LimberJack.Theory
```


