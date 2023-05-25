# LimberJack.jl

[![Build Status](https://github.com/JaimeRZP/LimberJack.jl/workflows/CI/badge.svg)](https://github.com/JaimeRZP/LimberJack.jl/actions?query=workflow%3ALimberJack-CI+branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jaimerzp.github.io/LimberJack.jl/dev/)

![](https://raw.githubusercontent.com/JaimeRZP/LimberJack.jl/main/docs/src/assets/LimberJack_logo.png)

A differentiable cosmological code in Julia.

## Design Philosophy

 + **Modularity**: each main function within ```LimberJack.jl``` has its own module. New functions can be added  by including extra modules. ```LimberJack.jl``` has the following modules:
 
| Module      | function    |
| ----------- | :----------- |
| ```core.jl```    | Performs the computation of the fundamental theoretical predictions   |
| ```tracers.jl``` | Computes the kernels associated with each type of kernel      |
| ```spectra.jl```  | Computes the power spectra of any two tracers       |
| ```emulator.jl``` | Computes the primordial powerspectrum as given by the Mootoovaloo et al 2022 emulator       |
| ```Halofit.jl```  | Computes the non-linear matter power spectrum as given by the Halofit fitting formula       |
| ```theory.jl```   | Interface to ```Turing.jl```        |

+ **Object-oriented**: ```LimberJack.jl```  mimics ```CCL.py``` class structure by using ```Julia```'s ```structures```.
+ **```Julia```**: ```LimberJack.jl```  is fully written in ```Julia``` without needing to inerface to any other programming language (```C```, ```Python```...) to compute thoretical predictions. This allows the user full access to the code from beginning to end.

## Goals

+ **Gradients**: cheap gradients of its theoretical predictions.
+ **Precision**: within 0.005 relative error of```CCL```'s precision.
+ **Speed**: ```C```-like performance.


## Installation

Once you have installed ```Julia``` you can install ```LimberJack.jl``` by:
1. Clone the git repository
2. From the repository directory open ```Julia```
3. In the ```Julia``` command line run:
``` julia
    using Pkg
    Pkg.add(path=".")
```

## Use

``` julia
    # Import
    using LimberJack
    
    Ωm = 0.31
    s8 = 0.81
    Ωb = 0.05
    h = 0.67
    ns = 0.96
    
    # create LimberJack.jl Cosmology instance
    cosmology = Cosmology(Ωm, Ωb, h, ns, s8;
                          tk_mode="EisHu",
                          Pk_mode="Halofit")
    
    z = Vector(range(0., stop=2., length=256))
    nz = @. exp(-0.5*((z-0.5)/0.05)^2)
    tracer = NumberCountsTracer(cosmology, zs, nz; b=1.0)
    ls = [10.0, 30.0, 100.0, 300.0]
    cls = angularCℓs(cosmology, tracer, tracer, ls)
```

## Challenges

1. Threading parallalization.
2. GPU implementation.
3. ```Zygote``` compatibility.
