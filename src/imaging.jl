using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, SparseArrays

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

function prepare_spmat(::Type{T}, img_size, bspec::FilterSpec) where T<:Real
    ctr1, ctr2 = img_size .÷ 2 .+ 1
    is = Int[]
    js = Int[]
    vs = T[]
    linds = LinearIndices(img_size)
    nx, ny = img_size
    for k in eachindex(bspec.wavelengths)
        r = bspec.wavelengths[k] / bspec.base_wavelength
        inten = bspec.intensities[k]
        for j in 1:ny
            dy = j - ctr2
            sy = ctr2 + dy * r
            iy = floor(Int, sy)
            (1 <= iy < ny) || continue
            for i in 1:nx
                dx = i - ctr1
                sx = ctr1 + dx * r
                ix = floor(Int, sx)
                if 1 <= ix < nx
                    tx = sx - ix
                    ty = sy - iy
                    push!(is, linds[i, j], linds[i, j], linds[i, j], linds[i, j])
                    push!(js, linds[ix, iy], linds[ix + 1, iy], linds[ix, iy + 1], linds[ix + 1, iy + 1])
                    push!(vs,   (1 - tx) * (1 - ty) * inten,
                                tx       * (1 - ty) * inten,
                                (1 - tx) * ty       * inten,
                                tx       * ty       * inten)
                end
            end
        end
    end
    return sparse(is, js, vs, nx * ny, nx * ny)
end

abstract type TrueSky{T} end

@kwdef struct PointSource{T} <: TrueSky{T}
    nphotons::T=Inf
    background::T=1.0
end
Base.convert(::Type{TrueSky{T}}, b::PointSource) where {T<:Real} =
    PointSource{T}(b.nphotons, b.background)
isfinite_photons(ts::PointSource) = isfinite(ts.nphotons)

struct DoubleSystem{T} <: TrueSky{T}
    rel_position::NTuple{2,Int}
    intensity::T
    brightness::PointSource{T}
    DoubleSystem(position, intensity::Real, brightness::PointSource=PointSource()) =
        new{typeof(intensity)}(Tuple(position), intensity, convert(TrueSky{typeof(intensity)}, brightness))
end

Base.convert(::Type{TrueSky{T}}, b::DoubleSystem) where {T<:Real} =
    DoubleSystem(b.rel_position, T(b.intensity), convert(TrueSky{T}, b.brightness))
isfinite_photons(ds::DoubleSystem) = isfinite_photons(ds.brightness)

struct TrueSkyImage{T, MT<:AbstractMatrix{Complex{T}}} <: TrueSky{T}
    true_sky_fft::MT
    brightness::PointSource{T}
end
function TrueSkyImage(true_sky::AbstractMatrix{T}, brightness=PointSource()) where {T<:Real}
    true_sky_fft = ifft(ifftshift(true_sky))
    return new{T, typeof(true_sky_fft)}(true_sky_fft, convert(PointSource{T}, brightness))
end
Base.convert(::Type{TrueSky{T}}, b::TrueSkyImage) where {T<:Real} =
    TrueSkyImage{T, typeof(b.true_sky_fft)}(convert.(Complex{T}, b.true_sky_fft), convert(PointSource{T}, b.brightness))
isfinite_photons(ts::TrueSkyImage) = isfinite_photons(ts.brightness)

struct ImagingSpec{T, AT<:AbstractMatrix{T}}
    aperture::AT
    img_size::NTuple{2,Int}
    filter_spec::FilterSpec{T}
end
ImagingSpec(aperture::AbstractMatrix{T}, imsize::NTuple{2,Int}, bspec::FilterSpec) where T<:Real =
    ImagingSpec{T, typeof(aperture)}(aperture, imsize, convert(FilterSpec{T}, bspec))

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

struct ImagingBuffers{AT, BT, MT, PT}
    aperture::AT
    radial_blur::BT
    aperture_buffer::MT
    focal_buffer::MT
    fftplan::PT
end

function ImagingBuffers(imgspec::ImagingSpec, batch)
    complex_type = complex(eltype(imgspec.aperture))
    buf1 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    buf2 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    if length(imgspec.filter_spec.wavelengths) > 1
        smat = prepare_spmat(eltype(imgspec.aperture), imgspec.img_size, imgspec.filter_spec)
    else
        smat = nothing
    end
    return ImagingBuffers(imgspec.aperture, smat, buf1, buf2, plan_fft(buf1, (1, 2)))
end

function write_phases!(aperture_buffer, phases, aperture)
    M, N = size(phases)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :] .=
        aperture .* cis.(phases)
    return pipeline
end

function radial_blur!(out, src, smat::AbstractMatrix)
    mul!(reshape(out, :, size(src, 3)), smat, reshape(src, :, size(src, 3)))
    return out
end
radial_blur!(out, src, ::Nothing) = copyto!(out, src)

function psf!(bufs::ImagingBuffers, phases)
    write_phases!(bufs.focal_buffer, phases, bufs.aperture)
    mul!(bufs.aperture_buffer, bufs.fftplan, bufs.focal_buffer)
    fftshift!(bufs.focal_buffer, bufs.aperture_buffer, (1, 2))
    bufs.aperture_buffer .= abs2.(bufs.focal_buffer)
    radial_blur!(bufs.focal_buffer, bufs.aperture_buffer, bufs.radial_blur)
end

function apply_image!(dst, ibufs::ImagingBuffers, ts::TrueSkyImage, psf_norm)
    mul!(ibufs.aperture_buffer, ibufs.fftplan, ibufs.focal_buffer)
    ibufs.aperture_buffer .*= ts.true_sky_fft
    ldiv!(ibufs.focal_buffer, ibufs.fftplan, ibufs.aperture_buffer)
    apply_image!(dst, ibufs, ts.brightness, psf_norm)
end
function apply_image!(dst, ibufs::ImagingBuffers, ds::DoubleSystem, psf_norm)
    img = ibufs.focal_buffer
    @assert all(abs.(ds.rel_position) .< size(img)[1:2] .÷ 2)
    @assert size(dst) == size(img)
    o1, o2 = ds.rel_position
    s1_dest, s1_src = o1 > 0 ? (o1 + 1:size(img, 1), 1:size(img, 1) - o1) : (1:size(img, 1) + o1, -o1 + 1:size(img, 1))
    s2_dest, s2_src = o2 > 0 ? (o2 + 1:size(img, 2), 1:size(img, 2) - o2) : (1:size(img, 2) + o2, -o2 + 1:size(img, 2))
    @. img[s1_dest, s2_dest, :] += img[s1_src, s2_src, :] * ds.intensity
    apply_image!(dst, ibufs, ds.brightness, psf_norm)
end
function apply_image!(dst, ibufs::ImagingBuffers, pt::PointSource, psf_norm)
    img = ibufs.focal_buffer
    if isfinite(pt.nphotons)
        @assert maximum(abs ∘ imag, img) / maximum(abs ∘ real, img) < 1e-5
        @assert all(x -> real(x) ≥ 0, img)
        @. dst = rand(Poisson(real(img) / psf_norm * pt.nphotons + pt.background))
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

function _prepare_imgbuffers(img_spec::ImagingSpec, batch::Int, ::Val{:serial})
    return ImagingBuffers(img_spec, batch)
end
@inline function _kernel!(real_img, img_buf::ImagingBuffers, phase_buf, true_sky, psf_norm)
    psf!(img_buf, phase_buf)
    apply_image!(real_img, img_buf, true_sky, psf_norm)
end
function _prepare_imgbuffers(img_spec::ImagingSpec, ::Int, ::Val{:threaded})
    imgbuf1 = ImagingBuffers(img_spec, 1)
    img_buf_channel = Channel{typeof(imgbuf1)}(Threads.nthreads())
    put!(img_buf_channel, imgbuf1)
    Threads.@threads for _ in 1:Threads.nthreads()-1
        img_buffs = ImagingBuffers(img_spec, 1)
        put!(img_buf_channel, img_buffs)
    end
    return img_buf_channel
end
@inline function _kernel!(real_img, img_buf_channel::Channel, phase_buf, true_sky, psf_norm)
    Threads.@threads for j in 1:size(real_img, 3)
        img_buffs = take!(img_buf_channel)
        psf!(img_buffs, view(phase_buf, :, :, j))
        apply_image!(view(real_img, :, :, j), img_buffs, true_sky, psf_norm)
        put!(img_buf_channel, img_buffs)
    end
end
function simulate_images(::Type{T}, img_spec::ImagingSpec{T2}, phase_sampler::PhaseSampler,
        true_sky::TrueSky=PointSource(); n::Int, batch::Int=64, filename="images.h5", verbose=true,
        save_phases::Bool=true, serial::Bool=false) where {T,T2}
    if !isfinite_photons(true_sky) && T <: Integer
        throw(ArgumentError("Integer image eltype not compatible with infinite-photon true sky model."))
    end
    true_sky_conv = convert(TrueSky{T2}, true_sky)
    batch = min(batch, n)
    img_size = img_spec.img_size
    h5open(filename, "w") do fid
        img_dataset = create_dataset(fid, "images", T, (img_size..., n), chunk=(img_size..., batch))
        p = Progress(n, "Simulating images", enabled=verbose, dt=1)
        real_img = zeros(T, img_size..., batch)
        noise_buf = noise_buffer(phase_sampler, batch)
        phase_buf = samplephases(phase_sampler, batch)
        if save_phases
            phs_size = plate_size(phase_sampler)
            fid["aperture"] = img_spec.aperture
            phs_dataset = create_dataset(fid, "phases", eltype(phase_buf), (phs_size..., n), chunk=(phs_size..., batch))
        end
        psf_norm = sum(img_spec.aperture)^2

        img_buf_channel = _prepare_imgbuffers(img_spec, batch, serial ? Val(:serial) : Val(:threaded))
        for j in 1:cld(n, batch)
            samplephases!(phase_buf, phase_sampler, noise_buf)
            _kernel!(real_img, img_buf_channel, phase_buf, true_sky_conv, psf_norm)
            HDF5.write_chunk(img_dataset, j - 1, real_img)
            save_phases && HDF5.write_chunk(phs_dataset, j - 1, phase_buf)
            next!(p, step=min(batch, n - (j - 1) * batch))
        end
        finish!(p)
    end
end
simulate_images(img_spec::ImagingSpec, phase_sampler::PhaseSampler, true_sky::TrueSky=PointSource(); kwargs...) =
    simulate_images(isfinite_photons(true_sky) ? Int : Float64, img_spec, phase_sampler, true_sky; kwargs...)
