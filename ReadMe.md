# LimberJack.jl

[![Build Status](https://github.com/JaimeRZP/LimberJack.jl/workflows/CI/badge.svg)](https://github.com/JaimeRZP/LimberJack.jl/actions?query=workflow%3ALimberJack-CI+branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jaimeruizzapatero.net/LimberJack.jl/dev/)
![size](https://img.shields.io/github/repo-size/jaimerzp/LimberJack.jl)

![](https://raw.githubusercontent.com/JaimeRZP/LimberJack.jl/main/docs/src/assets/LimberJack_logo.png)
![](https://raw.githubusercontent.com/JaimeRZP/LimberJack.jl/main/docs/src/assets/Pk_diff.png)
![](https://raw.githubusercontent.com/JaimeRZP/LimberJack.jl/main/docs/src/assets/cls_diff.png)

<p align="center"> A differentiable cosmological code in Julia. </p>

## Design Philosophy

 + **Modularity**: each main function within ```LimberJack.jl``` has its own module. New functions can be added  by including extra modules. ```LimberJack.jl``` has the following modules:
 
| Module      | function    |
| ----------- | :----------- |
| ```boltzmann.jl```    | Performs the computation of primordial power spectrum   |
| ```core.jl```    | Defines the structures where the theoretical predictions are stored and computes the background quantities   |
| ```data_utils.jl```   | Manages ```sacc``` files for large data vectors       |
| ```growth.jl```   | Computes the growth factor       |
| ```halofit.jl```  | Computes the non-linear matter power spectrum as given by the Halofit fitting formula       |
| ```spectra.jl```  | Computes the power spectra of any two tracers       |
| ```theory.jl```   | Computes large data vectors that combine many spectra     |
| ```tracers.jl``` | Computes the kernels associated with each type of kernel      |

+ **Object-oriented**: ```LimberJack.jl```  mimics ```CCL.py``` class structure by using ```Julia```'s ```structures```.
+ **Transparency**: ```LimberJack.jl```  is fully written in ```Julia``` without needing to inerface to any other programming language (```C```, ```Python```...) to compute thoretical predictions. This allows the user full access to the code from input to output.

## Goals

+ **Gradients**: one order of magnitude faster gradients than finite differences.
+ **Precision**: sub-percentage error with respect to ```CCL```.
+ **Speed**: ```C```-like performance.

## Installation

In order to run ```LimberJack.jl``` you will need ```Julia-1.7.0``` or newer installed in your system.
Older versions of ```Julia``` might be compatible but haven't been tested.
You can find instructions on how to install ```Julia``` here: https://julialang.org/downloads/.

Once you have installed ```Julia``` you can install ```LimberJack.jl``` following these steps:
1. Clone the git repository
2. From the repository directory open ```Julia```
3. In the ```Julia``` command line run:
``` julia
    using Pkg
    Pkg.add(path=".")
```
### Installing Sacc.py in Julia

``` julia
    using Pkg
    Pkg.add("CondaPkg")
    CondaPkg.add("sacc")
```

## Use

``` julia
    # Import
    using LimberJack
    
    # create LimberJack.jl Cosmology instance
    cosmology = Cosmology(Ωm=0.30, Ωb=0.05, h=0.70, ns=0.96, s8=0.81;
                          tk_mode="EisHu",
                          Pk_mode="Halofit")
    
    z = Vector(range(0., stop=2., length=256))
    nz = @. exp(-0.5*((z-0.5)/0.05)^2)
    tracer = NumberCountsTracer(cosmology, zs, nz; b=1.0)
    ls = [10.0, 30.0, 100.0, 300.0]
    cls = angularCℓs(cosmology, tracer, tracer, ls)
```

## Challenges

1.  **Parallelization**: the current threading parallelization of ```LimberJack.jl``` is far away from the optimal one over number of threads scaling. Future works could study alternative parallalization schemes or  possible inneficiencies in the code. 
2. **GPU's**: ```LimberJack.jl``` currently cannot run on GPU's which are known to significantly speed-up cosmological inference. Future works could study implementing ```Julia``` GPU libraries such as ```CUDA.jl```.
3. **Backwards-AD**: currently ```LimberJack.jl```'s preferred AD mode is forward-AD. However, the key computation of cosmological inference, obtaining the $\chi^2$, is a map from N parameters to a scalar. For a large number of parameters, backwards-AD is in theory the preferred AD mode and should significantly speed up the computation of the gradient. Future works could look into making ```LimberJack.jl``` compatible with the latest ```Julia``` AD libraries such as ```Zygote.jl``` to implement efficient backwards-AD.

## Contributors
| | | | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src=https://github.com/jaimerzp.png  width="100" height="100" /> | <img src=https://github.com/anicola.png  width="100" height="100" /> | <img src=https://github.com/carlosggarcia.png  width="100" height="100" /> |<img src=https://github.com/damonge.png  width="100" height="100" />| <img src=https://github.com/harry45.png  width="100" height="100" /> | <img src=https://github.com/jmsull.png  width="100" height="100" /> |  <img src=https://github.com/marcobonici.png  width="100" height="100" /> |
| Jaime Ruiz-Zapatero | Andrina Nicola | Carlos Garcia-garcia| David Alonso | Arrykrishna Mootoovaloo | Jamie Sullivan | Marco Bonici |
| Lead | Halofit | Validation | Tracers | EmuPk | Bolt.jl | Benchmarks |
