module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, KolmogorovUncorrelated
include("imaging.jl")
export FilterSpec, ImagingSpec, imgsize, psf, psf!, CircularAperture, simulate_images

FFTW.set_num_threads(8)

end # module AtmosphericTurbulenceSimulator
