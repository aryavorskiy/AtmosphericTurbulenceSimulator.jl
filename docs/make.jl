using Documenter
using AtmosphericTurbulenceSimulator

DocMeta.setdocmeta!(AtmosphericTurbulenceSimulator, :DocTestSetup, :(using AtmosphericTurbulenceSimulator); recursive=true)

makedocs(
    sitename = "AtmosphericTurbulenceSimulator.jl",
    modules = [AtmosphericTurbulenceSimulator],
    format = Documenter.HTML(),
    pages = [
        "Tutorial" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/aryavorskiy/AtmosphericTurbulenceSimulator",
    devbranch = "master",
)