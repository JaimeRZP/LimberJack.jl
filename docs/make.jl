# make.jl
using Documenter, LimberJack

makedocs(sitename = "LimberJack.jl",
         modules = [LimberJack],
         pages = ["Home" => "index.md",
                  "API" => "api.md",
                  "Tutorial" => "tutorial.md"])
         
deploydocs(repo = "github.com/JaimeRZP/LimberJack.jl")