using AtmosphericTurbulenceSimulator

print("Benchmarking atmospheric turbulence simulation...\n")
turb = KolmogorovUncorrelated((64, 64), 0.2 / (2/64))
aperture = CircularAperture((64, 64))
pipeline = ImagingSpec(aperture, FilterSpec(1, bandpass=0.1))

@assert all(isfinite, turb.sampler.noise_transform)
@time simulate_images(Int16, pipeline, turb, n=10000)
