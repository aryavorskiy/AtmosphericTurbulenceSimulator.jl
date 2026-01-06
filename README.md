# AtmosphericTurbulenceSimulator

A simple (yet) Julia toolchain to simulate atmospheric turbulence effects on imaging systems. It provides
utilities to define different telescope apertures and true sky models; the phase screens are generated
using common statistical models of atmospheric turbulence. The output is written into a HDF5 file
containing the simulated images and optionally the phase screens used to generate them.

## Installation

This package is not registered yet. You can install it with the following command in Julia's REPL:

```julia
using Pkg
Pkg.add(path="https://github.com/aryavorskiy/AtmosphericTurbulenceSimulator")
```

## Quick example

### Turbulent phase generation

The core functionality is generating turbulent phase screens using the `SingleLayer` atmosphere specification.
You can generate phase screens with or without Harding interpolation:

```julia
using AtmosphericTurbulenceSimulator

# Without Harding interpolation: generates 64×64 phase screens directly
# r0 = 0.2 m, assuming a 2 m telescope aperture diameter
atm_basic = SingleLayer((64, 64), 0.2 / 2 * 64)

# With Harding interpolation: samples at low resolution, then upsamples
# Using :auto to determine optimal number of interpolation passes
atm_harding = SingleLayer((256, 256), 0.2 / 2 * 256; interpolate=:auto)

# Alternative: specify number of interpolation passes explicitly
atm_2pass = SingleLayer((256, 256), 0.2 / 2 * 256; interpolate=2)

# Or specify the low-resolution grid size directly
atm_from = SingleLayer((256, 256), 0.2 / 2 * 256; interpolate_from=(32, 32))

# Generate phase screens and save to HDF5
simulate_phases(atm_harding; n=3000, filename="phases.h5")
```

The Harding interpolation (from [Harding et al. 1999](https://doi.org/10.1364/AO.38.002161))
allows efficient generation of high-resolution phase screens by sampling the turbulence at a coarser
resolution and upsampling in a way that preserves Kolmogorov statistics.

### PSF simulation with imaging pipeline

To simulate actual images through turbulence, combine the atmosphere specification with an imaging
specification and a true-sky model:

```julia
# Define circular aperture and imaging parameters
ap = CircularAperture((64, 64), 25)
img_spec = ImagingSpec(ap, nyquist_oversample=1)

# Atmosphere specification
atm = SingleLayer((64, 64), 0.2 / 2 * 64, interpolate=:auto)

# True sky models:
# Point source with 1e7 total photons and 200 photons/pixel background
ts_point = PointSource(1e7, 200)

# Binary system: secondary offset by (5, 3) pixels with 0.5× intensity
ts_double = DoubleSystem((5, 3), 0.5; nphotons=1e7, background=200)

# Custom image from array
# ts_image = TrueSkyImage(my_image_array; nphotons=Inf)

# Simulate images and save to HDF5 (includes phase screens by default)
simulate_images(img_spec, atm, ts_point; n=3000, filename="simulation.h5")
```

This will create a HDF5 file `simulation.h5` containing 3000 simulated images of the point source
through the turbulent atmosphere, along with the phase screens used. The result can be visualized as follows:
<details>
<summary>Show code</summary>

```julia
using HDF5, CairoMakie, Statistics

img_dataset = h5read("simulation.h5", "images")
first_image = img_dataset[:, :, 1]
first_phase = h5read("simulation.h5", "phases", (:, :, 1))
mean_image = dropdims(mean(img_dataset, dims=3), dims=3)

fig = Figure(size=(900, 300))
ax1, hm  = heatmap(fig[1, 1], first_phase, colormap=:viridis, axis=(aspect=DataAspect(), title="Phase screen"))
hidedecorations!(ax1)
ax2, hm2 = heatmap(fig[1, 2], first_image, colormap=:jet, axis=(aspect=DataAspect(), title="Simulated image"))
hidedecorations!(ax2)
ax3, hm3 = heatmap(fig[1, 3], mean_image, colormap=:jet, axis=(aspect=DataAspect(), title="Accumulated (3000 frames)"))
hidedecorations!(ax3)
Colorbar(fig[1, 0], hm; label="[rad]", ticks = MultiplesTicks(5, π, "π"), flipaxis=false)
fig
```
</details>

![Demo image](demo.svg)

## Notes

This toolchain utilizes Julia's multi-threading capabilities; add more threads by launching Julia with `julia --threads=N`, where `N` is the number of threads you want. You can set `N=auto` to use all available CPU threads.

To enable CUDA (or other GPU backends), import the respective packages (e.g., `CUDA.jl`) and add `deviceadapter=CuArray` (or other device array type) to the `simulate_phases`/`simulate_images` function call. As of version 0.1, only [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is tested for compatibility, feel free to open an issue if you encounter problems with other backends.

Please note that this package is in early development. There are multiple features planned for the future:
- More advanced atmosphere models
    - [x] Harding interpolation
    - [ ] Frozen flow
    - [ ] Multi-layer atmospheres
- More advanced imaging models
    - [ ] Multi-wavelength imaging
    - [ ] Non-circular apertures
    - [ ] Off-axis propagation
- [ ] FITS support (?)