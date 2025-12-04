using AtmosphericTurbulenceSimulator

print("Benchmarking atmospheric turbulence simulation...\n")
turb = KolmogorovUncorrelated(Float32, (64, 64), 0.2 / (2/64))
aperture = CircularAperture(Float32, (64, 64))
pipeline = ImagingSpec(aperture)
@time simulate_images(Int16, pipeline, turb, n=10000)
