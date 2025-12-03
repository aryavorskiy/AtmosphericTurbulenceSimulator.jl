using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter

struct BandSpec
    base_wavelength::Float64
    wavelengths::Vector{Float64}
    intensities::Vector{Float64}
    function BandSpec(base_wavelength, wavelengths, intensities=ones(length(wavelengths)))
        @assert length(wavelengths) == length(intensities)
        new(base_wavelength, wavelengths, intensities)
    end
end
function BandSpec(base_wavelength::Real; bandpass, tmax=1, tmin=1, npts=7)
    wavelengths = range(base_wavelength - bandpass / 2, base_wavelength + bandpass / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tmax - tmin) + tmin
    return BandSpec(base_wavelength, wavelengths, intensities)
end
function radial_blur!(out, src, bspec::BandSpec)
    # TODO refactor vibe-coded function
    @assert size(out) == size(src)
    n1, n2 = size(src)
    ctr1, ctr2 = (n1 ÷ 2 + 1, n2 ÷ 2 + 1)

    @inbounds for j in 1:n2
        dy = j - ctr2
        for i in 1:n1
            dx = i - ctr1
            acc = zero(eltype(out))
            for k in eachindex(bspec.wavelengths)
                r = bspec.wavelengths[k] / bspec.base_wavelength
                sx = ctr1 + dx * r
                sy = ctr2 + dy * r
                ix = floor(Int, sx)
                iy = floor(Int, sy)
                if 1 <= ix < n1 && 1 <= iy < n2
                    tx = sx - ix
                    ty = sy - iy
                    v00 = src[ix,     iy    ]
                    v10 = src[ix + 1, iy    ]
                    v01 = src[ix,     iy + 1]
                    v11 = src[ix + 1, iy + 1]
                    val = (1 - tx) * (1 - ty) * v00 +
                          tx       * (1 - ty) * v10 +
                          (1 - tx) * ty       * v01 +
                          tx       * ty       * v11
                    acc += bspec.intensities[k] * val
                end
            end
            out[i, j] = acc
        end
    end
    return out
end

"""
    ImagingPipeline

A struct that represents an imaging pipeline of a telescope.
"""
struct ImagingPipeline{AT,MT,PT,ST<:Union{Nothing,BandSpec}}
    aperture::AT
    band_spec::ST
    aperture_buffer::MT
    focal_buffer::MT
    fftplan!::PT
end

"""
    ImagingPipeline(aperture; nyqist_oversample=1)

Creates an imaging pipeline for a telescope with a given aperture function.

# Arguments
- `aperture`: the aperture function of the telescope (a 2D array).
- `nyqist_oversample`: the oversampling factor of the Nyquist frequency.
    Basically, if the aperture is sampled at a frequency `f`, the Nyquist frequency is
    `f / 2`. The aperture is then padded with zeros to reach a frequency of
    `f / 2 * nyqist_oversample`.
"""
function ImagingPipeline(aperture, bspec=nothing; nyqist_oversample=1, batch=64)
    buf1 = similar(aperture, ComplexF64, round.(Int, 2 .* nyqist_oversample .* size(aperture))..., batch)
    buf2 = similar(buf1)
    fill!(buf1, 0); fill!(buf2, 0)
    return ImagingPipeline(aperture, bspec, buf1, buf2, plan_fft!(buf1, (1, 2)))
end
Base.size(pipeline::ImagingPipeline) = size(pipeline.aperture)
imgsize(pipeline::ImagingPipeline) = size(pipeline.focal_buffer)[1:2]
batch_length(pipeline::ImagingPipeline) = size(pipeline.aperture_buffer, 3)

function write_phases!(aperture_buffer, phases, aperture)
    M, N = size(phases)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :] .=
        aperture .* exp.(im .* phases)
    return pipeline
end

"""
    psf!(pipeline, phases)

Create an image of a point source with a given phase pattern.
"""
function psf!(pipeline::ImagingPipeline, phases)
    write_phases!(pipeline.aperture_buffer, phases, pipeline.aperture)
    pipeline.fftplan! * pipeline.aperture_buffer
    ifftshift!(pipeline.focal_buffer, pipeline.aperture_buffer, (1, 2))
    if pipeline.band_spec !== nothing
        pipeline.aperture_buffer .= abs2.(pipeline.focal_buffer)
        radial_blur!(pipeline.focal_buffer, pipeline.aperture_buffer, pipeline.band_spec)
    else
        pipeline.focal_buffer .= abs2.(pipeline.focal_buffer)
    end
end

"""
    psf(pipeline, phases)

Compute the point spread function (PSF) with a given phase pattern.
"""
psf(pipeline::ImagingPipeline, phases) = real(psf!(pipeline, phases))

function apply_image_fft!(pipeline::ImagingPipeline, true_img_fft; kw...)
    pipeline.fftplan! \ pipeline.focal_buffer
    pipeline.focal_buffer .*= true_img_fft
    pipeline.fftplan! * pipeline.focal_buffer
    return pipeline.focal_buffer
end

function simulate_readout!(dst, img, readout_buf; photons=Inf, background=1)
    if isfinite(photons)
        @assert maximum(abs ∘ imag, img) / maximum(abs ∘ real, img) < 1e-5
        @assert all(x -> real(x) ≥ 0, img)
        psf_norm = sum(real, img, dims=(1,2))
        @. dst = rand(Poisson(real(img) / psf_norm * photons + background))
    else
        dst .= img
    end
end

function CircularAperture(sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); aa_dist=1)
    aperture = zeros(sz)
    X, Y = sz .÷ 2 .+ 1
    for I in eachindex(IndexCartesian(), aperture)
        x, y = I[1] - X, I[2] - Y
        r = sqrt(x^2 + y^2)
        if r < radius - aa_dist / 2
            aperture[I] = 1
        elseif r > radius + aa_dist / 2
            aperture[I] = 0
        else
            aperture[I] = 0.5 - (r - radius) / aa_dist
        end
    end
    return aperture
end

_isfinite_photons(readout::NamedTuple) = get(readout, :photons, 0.0) isa Integer
function simulate_images(imag_pipe, phase_sampler, true_sky=nothing; n, filename="images.h5",
        verbose=true, readout=(photons=10_000, background=1), true_sky_fft=isnothing(true_sky) ? nothing : ifft(ifftshift(true_sky)))
    h5open(filename, "w") do fid
        batch = min(batch_length(imag_pipe), n)
        dataset = create_dataset(fid, "images", _isfinite_photons(readout) ? Int : Float64, (imgsize(imag_pipe)..., n), chunk=(imgsize(imag_pipe)..., batch))
        p = Progress(n, "Simulating images", enabled=verbose, dt=1)
        real_img = zeros(_isfinite_photons(readout) ? Int : Float64, imgsize(imag_pipe)..., batch)
        noise_buf = noise_buffer(phase_sampler, batch)
        phase_buf = samplephases(phase_sampler, batch)
        readout_buf = similar(real_img, 1, 1, batch)
        for j in 1:cld(n, batch)
            samplephases!(phase_buf, phase_sampler, noise_buf)
            img = psf!(imag_pipe, phase_buf)
            if true_sky_fft !== nothing
                img = apply_image_fft!(imag_pipe, true_sky_fft)
            end
            simulate_readout!(real_img, img, readout_buf; readout...)
            HDF5.write_chunk(dataset, j-1, real_img)
            next!(p, step=min(batch, n - (j - 1) * batch))
        end
        finish!(p)
    end
end
