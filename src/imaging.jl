using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter

struct FilterSpec{T}
    base_wavelength::T
    wavelengths::Vector{T}
    intensities::Vector{T}
end
function FilterSpec(::Type{T}, base_wavelength::Real; bandpass, tmax=1, tmin=1, npts=7) where T<:Real
    wavelengths = range(base_wavelength - bandpass / 2, base_wavelength + bandpass / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tmax - tmin) + tmin
    return FilterSpec{T}(base_wavelength, wavelengths, intensities)
end
FilterSpec(base_wavelength::Real; kw...) = FilterSpec(Float64, base_wavelength; kw...)
Base.convert(::Type{FilterSpec{T}}, bspec::FilterSpec) where T<:Real =
    FilterSpec{T}(bspec.base_wavelength,
        bspec.wavelengths, bspec.intensities)

function radial_blur!(out, src, bspec::FilterSpec)
    @assert size(out) == size(src)
    n1, n2 = size(src)
    ctr1, ctr2 = (n1 ÷ 2 + 1, n2 ÷ 2 + 1)

    Threads.@threads for p in axes(src, 3)
        for j in 1:n2
            dy = j - ctr2
            for i in 1:n1
                dx = i - ctr1
                ze = zero(eltype(src))
                for k in eachindex(bspec.wavelengths)
                    r = bspec.wavelengths[k] / bspec.base_wavelength
                    sx = ctr1 + dx * r
                    sy = ctr2 + dy * r
                    ix = floor(Int, sx)
                    iy = floor(Int, sy)
                    if 1 <= ix < n1 && 1 <= iy < n2
                        tx = sx - ix
                        ty = sy - iy
                        v00 = src[ix,     iy    , p]
                        v10 = src[ix + 1, iy    , p]
                        v01 = src[ix,     iy + 1, p]
                        v11 = src[ix + 1, iy + 1, p]
                        ze += ((1 - tx) * (1 - ty) * v00 +
                            tx       * (1 - ty) * v10 +
                            (1 - tx) * ty       * v01 +
                            tx       * ty       * v11) * bspec.intensities[k]
                    end
                end
                out[i, j, p] = ze
            end
        end
    end
    return out
end

struct ImagingSpec{T, AT<:AbstractMatrix{T}}
    aperture::AT
    img_size::NTuple{2,Int}
    band_spec::FilterSpec{T}
end

"""
    ImagingSpec(aperture[, imsize, bspec]; nyqist_oversample=1)

Creates an imaging pipeline spec object for a telescope with a given aperture function.

# Arguments
- `aperture`: the aperture function of the telescope (a 2D array).
- `imsize`: the size of the output images (a tuple of two integers). If not provided,
    it is computed as double the size of the aperture times the `nyqist_oversample` factor.
"""
ImagingSpec(aperture, bspec=FilterSpec(1, [1], [1]); nyqist_oversample=1) =
    ImagingSpec(aperture, round.(Int, size(aperture) .* 2 .* nyqist_oversample),
        convert(FilterSpec{eltype(aperture)}, bspec))
ImagingSpec(aperture, imsize::NTuple{2,Int}) =
    ImagingSpec(aperture, imsize, FilterSpec{eltype(aperture)}(1, [1], [1]))

struct ImagingBuffers{T, AT, MT, PT}
    aperture::AT
    band_spec::FilterSpec{T}
    aperture_buffer::MT
    focal_buffer::MT
    fftplan::PT
end

function ImagingBuffers(imgspec::ImagingSpec, batch)
    complex_type = complex(eltype(imgspec.aperture))
    buf1 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    buf2 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    return ImagingBuffers(imgspec.aperture, imgspec.band_spec, buf1, buf2, plan_fft(buf1, (1, 2)))
end

function write_phases!(aperture_buffer, phases, aperture)
    M, N = size(phases)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :] .=
        aperture .* cis.(phases)
    return pipeline
end

function psf!(bufs::ImagingBuffers, phases)
    write_phases!(bufs.focal_buffer, phases, bufs.aperture)
    mul!(bufs.aperture_buffer, bufs.fftplan, bufs.focal_buffer)
    ifftshift!(bufs.focal_buffer, bufs.aperture_buffer, (1, 2))
    if length(bufs.band_spec.wavelengths) > 1
        bufs.aperture_buffer .= abs2.(bufs.focal_buffer)
        radial_blur!(bufs.focal_buffer, bufs.aperture_buffer, bufs.band_spec)
    else
        bufs.focal_buffer .= abs2.(bufs.focal_buffer)
    end
end

function apply_image_fft!(pipeline::ImagingBuffers, true_img_fft; kw...)
    mul!(pipeline.aperture_buffer, pipeline.fftplan, pipeline.focal_buffer)
    pipeline.aperture_buffer .*= true_img_fft
    ldiv!(pipeline.focal_buffer, pipeline.fftplan, pipeline.aperture_buffer)
    return pipeline.focal_buffer
end

function simulate_readout!(dst, img; photons=Inf, background=1)
    if isfinite(photons)
        @assert maximum(abs ∘ imag, img) / maximum(abs ∘ real, img) < 1e-5
        @assert all(x -> real(x) ≥ 0, img)
        Threads.@threads for j in axes(img, 3)
            psf_norm = sum(real, @view img[:, :, j])
            @. dst[:, :, j] = rand(Poisson(real(@view img[:, :, j]) / psf_norm * photons + background))
        end
    else
        copyto!(dst, img)
    end
end

function CircularAperture(::Type{T}, sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); aa_dist=1) where T<:Real
    aperture = zeros(T, sz)
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
CircularAperture(sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); kw...) =
    CircularAperture(Float64, sz, radius; kw...)

function simulate_images(::Type{T}, img_spec::ImagingSpec, phase_sampler::PhaseSampler, true_sky=nothing; n::Int,
        batch::Int=64, filename="images.h5", verbose=true, readout=(photons=10_000, background=1),
        true_sky_fft=isnothing(true_sky) ? nothing : ifft(ifftshift(true_sky))) where T
    img_buffers = ImagingBuffers(img_spec, batch)
    img_size = img_spec.img_size
    batch = min(batch, n)
    h5open(filename, "w") do fid
        dataset = create_dataset(fid, "images", T, (img_size..., n), chunk=(img_size..., batch))
        p = Progress(n, "Simulating images", enabled=verbose, dt=1)
        real_img = zeros(T, img_size..., batch)
        noise_buf = noise_buffer(phase_sampler, batch)
        phase_buf = samplephases(phase_sampler, batch)
        for j in 1:cld(n, batch)
            samplephases!(phase_buf, phase_sampler, noise_buf)
            img = psf!(img_buffers, phase_buf)
            if true_sky_fft !== nothing
                img = apply_image_fft!(img_buffers, true_sky_fft)
            end
            simulate_readout!(real_img, img; readout...)
            HDF5.write_chunk(dataset, j - 1, real_img)
            next!(p, step=min(batch, n - (j - 1) * batch))
        end
        finish!(p)
    end
end
simulate_images(img_spec::ImagingSpec, phase_sampler::PhaseSampler; readout=(photons=10_000, background=1), kwargs...) =
    simulate_images(isfinite(readout.photons) ? Int : Float64, img_spec, phase_sampler; readout=readout, kwargs...)
