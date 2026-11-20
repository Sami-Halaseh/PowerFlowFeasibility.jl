using PowerFlowFeasibility
using Documenter

DocMeta.setdocmeta!(PowerFlowFeasibility, :DocTestSetup, :(using PowerFlowFeasibility); recursive=true)

makedocs(;
    modules=[PowerFlowFeasibility],
    authors="Pierre Arvy, Sami Halaseh, Yassine Koubaa, Luca Wellmeier and contributors",
    sitename="PowerFlowFeasibility.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://Sami-Halaseh.github.io/PowerFlowFeasibility.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/Sami-Halaseh/PowerFlowFeasibility.jl",
    devbranch="main",
)
