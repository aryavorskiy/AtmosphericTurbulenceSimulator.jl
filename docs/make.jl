using Documenter
using AtmosphericTurbulenceSimulator

DocMeta.setdocmeta!(AtmosphericTurbulenceSimulator, :DocTestSetup, :(using AtmosphericTurbulenceSimulator); recursive=true)

makedocs(
    sitename = "AtmosphericTurbulenceSimulator.jl",
    modules = [AtmosphericTurbulenceSimulator],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true"
    ),
    pages = [
        "Overview" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = Remotes.GitHub("aryavorskiy", "AtmosphericTurbulenceSimulator.jl"),
    devbranch = "master",
)
