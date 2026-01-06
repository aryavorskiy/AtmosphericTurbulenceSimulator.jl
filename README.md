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

This minimal example creates a simple circular aperture, an independent-frame atmosphere
sampler, and simulates a small stack of images of a point source. The generated images are
written to `simulation.h5` in the current directory.

```julia
using AtmosphericTurbulenceSimulator

# Aperture and imaging spec
ap = CircularAperture((64, 64), 25)
img_spec = ImagingSpec(ap, nyquist_oversample=1)

# Atmosphere: independent phase patterns with r0 = 0.2 m
# Assuming Kolmogorov turbulence, a 2 m telescope aperture and 64-pixel images
atm = SingleLayer((64, 64), 0.2 / 2 * 50)

# True sky: point source (1e7 photons per PSF, 200 photon per pixel background)
ts = PointSource(1e7, 200)

# Simulate 3000 images, write to simulation.h5, save phases as well
simulate_images(img_spec, atm, ts; n=3000, filename="simulation.h5")
```


!!! Note
    If yoyu need to generate only turbulent phase screens without simulating images, you can use
    `simulate_phases(atm; n=3000)` instead.

Let's take a look at what we just created:

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
![Demo image](demo.svg)

## Notes

This toolchain utilizes Julia's multi-threading capabilities; add more threads by launching Julia with `julia --threads=N`, where `N` is the number of threads you want. You can set `N=auto` to use all available CPU threads.

To enable CUDA (or other GPU backends), import the respective packages (e.g., `CUDA.jl`) and add `deviceadapter=CuArray` (or other device array type) to the `simulate_images` function call. As of version 0.1, only [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is tested for compatibility, feel free to open an issue if you encounter problems with other backends.

Please note that this package is in early development. There are multiple features planned for the future:
- [ ] More advanced atmosphere models (frozen flow, multi-layer, etc.)
- [ ] Off-axis propagation
- [ ] Multi-wavelength imaging
- [ ] FITS support (?)