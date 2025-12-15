module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, KolmogorovUncorrelated
include("imaging.jl")
export FilterSpec, ImagingSpec, PointSource, DoubleSystem, TrueSkyImage, CircularAperture, simulate_images

end # module AtmosphericTurbulenceSimulator
