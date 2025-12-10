using AtmosphericTurbulenceSimulator

print("Benchmarking atmospheric turbulence simulation...\n")
turb = KolmogorovUncorrelated(Float32, (85, 85), 0.2 / (2/85))
aperture = CircularAperture(Float32, (85, 85))
pipeline = ImagingSpec(aperture, (256, 256), FilterSpec(1, bandpass=0.1))
@time simulate_images(Int16, pipeline, turb, n=10000)
