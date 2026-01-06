module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, SingleLayer
include("imaging.jl")
export FilterSpec, ImagingSpec, PointSource, DoubleSystem, TrueSkyImage, CircularAperture,
    simulate_images, simulate_phases

end # module AtmosphericTurbulenceSimulator
