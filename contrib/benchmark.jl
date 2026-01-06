using AtmosphericTurbulenceSimulator, FFTW

print("Benchmarking atmospheric turbulence simulation...\n")
turb = SingleLayer(Float32, (99, 99), 0.2 / (2/100), interpolate=:auto)
aperture = CircularAperture(Float32, (99, 99))
pipeline = ImagingSpec(aperture, (256, 256), FilterSpec(1, bandpass=0.1))
@time simulate_images(Int32, pipeline, turb, PointSource(1e7, 1.0), n=30000, savephases=false);
