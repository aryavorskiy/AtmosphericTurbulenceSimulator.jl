module AtmosphericTurbulenceSimulator

include("atmosphere.jl")
export kolmogorov_covmat, KolmogorovUncorrelated
include("imaging.jl")
export BandSpec, ImagingPipeline, imgsize, psf, psf!, CircularAperture, simulate_images

FFTW.set_num_threads(8)

end # module AtmosphericTurbulenceSimulator
