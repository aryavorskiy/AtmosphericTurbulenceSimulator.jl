using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, SparseArrays, Adapt

"""
    FilterSpec

Representation of a spectral filter used by the imaging pipeline.

---
    FilterSpec(base_wavelength, wavelengths[, intensities])

# Arguments
- `base_wavelength`: central wavelength for the filter.
- `wavelengths`: vector of sampled wavelengths within the filter bandpass.
- `intensities`: vector of relative intensities at each sampled wavelength. If not provided,
  equal weights are assumed.
"""
struct FilterSpec{T}
    base_wavelength::T
    wavelengths::Vector{T}
    intensities::Vector{T}
end
FilterSpec(base_wavelength::T1, wavelengths::AbstractVector{T2},
    intensities::AbstractVector{T3}=ones(Int, length(wavelengths))) where {T1,T2,T3} =
    FilterSpec{promote_type(T1, T2, T3)}(base_wavelength, wavelengths, intensities)

"""
    FilterSpec([T, ]base_wavelength; bandpass, tcenter=1, tedge=1, npts=7)

# Arguments
- `base_wavelength`: central wavelength for the filter (same units as `wavelengths`).

# Keyword Arguments
- `bandpass`: total width of the filter bandpass in wavelength units.
- `tcenter`: relative intensity at the center wavelength (default 1).
- `tedge`: relative intensity at the edges of the bandpass (default 1).
"""
function FilterSpec(::Type{T}, base_wavelength::Real; bandpass, tcenter=1, tedge=1, npts=7) where T<:Real
    wavelengths = range(base_wavelength - bandpass / 2, base_wavelength + bandpass / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tcenter - tedge) + tedge
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
    vs = complex(T)[]
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

"""
    PointSource(nphotons[, background])
    PointSource([;nphotons, background])

Simple true-sky brightness model. When `nphotons` is finite the simulator will
Poisson-sample pixel values according to the PSF-normalized flux with added background; if `nphotons` is
`Inf` the continuous flux is used (no shot noise), and background is ignored.

# Arguments
- `nphotons`: total photons for the source (or `Inf` for continuous mode).
- `background`: constant background added to the flux in photons per pixel.
"""
@kwdef struct PointSource{T} <: TrueSky{T}
    nphotons::T=Inf
    background::T=1.0
end
PointSource(nphotons::T1, background::T2) where {T1<:Real,T2<:Real} =
    return PointSource{promote_type(T1, T2)}(nphotons, background)
Base.convert(::Type{TrueSky{T}}, b::PointSource) where {T<:Real} =
    PointSource{T}(b.nphotons, b.background)
isfinite_photons(ts::PointSource) = isfinite(ts.nphotons)

"""
    DoubleSystem(rel_position, intensity[; nphotons, background])

Model for a two-component source (binary): primary plus a secondary offset by `rel_position`.
`nphotons` and `background` describe the brightness of the primary component; the default is
continuous flux, see [`PointSource`](@ref) for more info.

# Arguments
- `rel_position`: `(dx, dy)` integer tuple specifying the secondary's pixel offset.
- `intensity`: multiplicative intensity of the secondary relative to the primary.
"""
struct DoubleSystem{T} <: TrueSky{T}
    rel_position::NTuple{2,Int}
    intensity::T
    brightness::PointSource{T}
    DoubleSystem(position, intensity::Real, brightness::PointSource) =
        new{typeof(intensity)}(Tuple(position), intensity, convert(TrueSky{typeof(intensity)}, brightness))
end
DoubleSystem(position, intensity::Real; kw...) =
    DoubleSystem(position, intensity, PointSource(; kw...))

Base.convert(::Type{TrueSky{T}}, b::DoubleSystem) where {T<:Real} =
    DoubleSystem(b.rel_position, T(b.intensity), convert(TrueSky{T}, b.brightness))
isfinite_photons(ds::DoubleSystem) = isfinite_photons(ds.brightness)

"""
    TrueSkyImage(true_sky::AbstractMatrix{T}[; nphotons, background])

Wrap a real-valued true-sky image for use with the imaging pipeline. `nphotons` and
`background` describe the brightness of the source; the default is continuous flux, see
[`PointSource`](@ref) for more info.

# Arguments
- `true_sky`: real image array representing spatial sky brightness.
"""
struct TrueSkyImage{T, MT<:AbstractMatrix{Complex{T}}} <: TrueSky{T}
    true_sky_fft::MT
    brightness::PointSource{T}
end
function TrueSkyImage(true_sky::AbstractMatrix{T}, brightness=PointSource()) where {T<:Real}
    true_sky_fft = ifft(ifftshift(true_sky))
    return new{T, typeof(true_sky_fft)}(true_sky_fft, convert(PointSource{T}, brightness))
end
TrueSkyImage(mat::AbstractMatrix; kw...) =
    TrueSkyImage(mat, PointSource(; kw...))
Base.convert(::Type{TrueSky{T}}, b::TrueSkyImage) where {T<:Real} =
    TrueSkyImage{T, typeof(b.true_sky_fft)}(convert.(Complex{T}, b.true_sky_fft), convert(PointSource{T}, b.brightness))
Adapt.adapt_structure(to, ts::TrueSkyImage) =
    TrueSkyImage(Adapt.adapt_storage(to, ts.true_sky_fft), ts.brightness)
isfinite_photons(ts::TrueSkyImage) = isfinite_photons(ts.brightness)

"""
    ImagingSpec

Container describing the imaging system configuration. It is defined by the aperture function,
the specification of the filter and the output image size. If the image size does not match the
aperture size, the aperture is zero-padded accordingly.

---
    ImagingSpec(aperture, [img_size, filter_spec; nyquist_oversample=1])

# Arguments
- `aperture`: 2D aperture (pupil) array describing the telescope pupil.
- `img_size`: output image size `(nx, ny)`. If not provided, it is computed as double the
  size of the aperture times the `nyquist_oversample` factor.
- `filter_spec`: `FilterSpec` instance describing spectral sampling and relative intensities.
"""
struct ImagingSpec{T, AT<:AbstractMatrix{T}}
    aperture::AT
    img_size::NTuple{2,Int}
    filter_spec::FilterSpec{T}
end
ImagingSpec(aperture::AbstractMatrix{T}, imsize::NTuple{2,Int}, bspec::FilterSpec) where T<:Real =
    ImagingSpec{T, typeof(aperture)}(aperture, imsize, convert(FilterSpec{T}, bspec))
ImagingSpec(aperture, imsize::NTuple{2,Int}) =
    ImagingSpec(aperture, imsize, FilterSpec{eltype(aperture)}(1, [1], [1]))
ImagingSpec(aperture, filter_spec=FilterSpec(1, [1], [1]); nyquist_oversample=1) =
    ImagingSpec(aperture, round.(Int, size(aperture) .* 2 .* nyquist_oversample),
        convert(FilterSpec{eltype(aperture)}, filter_spec))
Adapt.adapt_structure(to, imgspec::ImagingSpec) =
    ImagingSpec(Adapt.adapt_storage(to, imgspec.aperture), imgspec.img_size, imgspec.filter_spec)

struct ImagingBuffers{AT, BT, MT, PT}
    aperture::AT
    radial_blur::BT
    aperture_buffer::MT
    focal_buffer::MT
    fftplan::PT
end

function ImagingBuffers(imgspec::ImagingSpec, blur, batch::Int)
    complex_type = complex(eltype(imgspec.aperture))
    buf1 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    buf2 = similar(imgspec.aperture, complex_type, imgspec.img_size..., batch)
    return ImagingBuffers(imgspec.aperture, blur, buf1, buf2, plan_fft(buf1, (1, 2)))
end
function prepare_blur(imgspec::ImagingSpec)
    if length(imgspec.filter_spec.wavelengths) > 1
        return prepare_spmat(eltype(imgspec.aperture), imgspec.img_size, imgspec.filter_spec)
    else
        return nothing
    end
end
function ImagingBuffers(imgspec::ImagingSpec, batch::Int)
    blur = prepare_blur(imgspec)
    return ImagingBuffers(imgspec, blur, batch)
end

function write_phases!(aperture_buffer, phases, aperture)
    M, N = size(phases)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :] .=
        aperture .* cis.(phases)
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
    @views @. img[s1_dest, s2_dest, :] += img[s1_src, s2_src, :] * ds.intensity
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

"""
    CircularAperture([T, ]sz, radius[; aa_dist=1])

Create a circular (optionally anti-aliased) aperture array of shape `sz`. Returns a 2D
numeric array suitable for use as an aperture in `ImagingSpec`.

# Arguments
- `T`: desired numeric element type, `Float64` by default.
- `sz`: aperture size `(nx, ny)`.
- `radius`: radius of the circular aperture in pixels. Defaults to the largest that fits.
- `aa_dist`: anti-aliasing transition width in pixels at the aperture edge.
"""
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

function prepare_imgbuffers(::Type{T}, img_spec::ImagingSpec, batch::Int, deviceadapter) where T
    return ImagingBuffers(
        adapt(deviceadapter, img_spec),
        adapt(deviceadapter, prepare_blur(img_spec)),
        batch),
    adapt(deviceadapter, zeros(T, img_spec.img_size..., batch))
end
@inline function imagephases!(real_img, img_buf_tuple::Tuple, phase_buf, true_sky, psf_norm)
    img_buf, real_img_buf = img_buf_tuple
    psf!(img_buf, phase_buf)
    apply_image!(real_img_buf, img_buf, true_sky, psf_norm)
    copyto!(real_img, real_img_buf)
end
function prepare_imgbuffers(::Type, img_spec::ImagingSpec, ::Int, ::Type{<:Array})
    imgbuf1 = ImagingBuffers(img_spec, 1)
    img_buf_vector = Array{typeof(imgbuf1)}(undef, Threads.nthreads())
    img_buf_vector[1] = imgbuf1
    Threads.@threads for i in 2:Threads.nthreads()
        img_buf_vector[i] = ImagingBuffers(img_spec, 1)
    end
    return img_buf_vector
end
@inline function imagephases!(real_img, img_buf_vector::Vector, phase_buf, true_sky, psf_norm)
    Threads.@threads for i in eachindex(img_buf_vector)
        img_buffs = img_buf_vector[i]
        for j in i:length(img_buf_vector):size(phase_buf, 3)
            psf!(img_buffs, view(phase_buf, :, :, j))
            apply_image!(view(real_img, :, :, j), img_buffs, true_sky, psf_norm)
        end
    end
end

"""
    simulate_images([T, ]img_spec::ImagingSpec, atm_spec::AtmosphereSpec[, truesky::TrueSky];
        n, [batch, filename, verbose, savephases, deviceadapter])

Simulate `n` images using the provided imaging and atmosphere specifications and write
the results to an HDF5 file.

# Arguments
- `T`: output image element type; if not provided, defaults to `Int` for
  finite-photon true sky models and `Float64` for infinite-photon models.
- `img_spec`: an `ImagingSpec` describing the aperture, image size and filter.
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `truesky`: a `TrueSky` model (e.g. `PointSource`, `DoubleSystem`, `TrueSkyImage`).

# Keyword Arguments
- `n`: number of images to simulate.
- `batch`: batch size for buffered HDF5 writes (default 512).
- `filename`: output HDF5 filename (default "images.h5").
- `verbose`: show progress meter (true by default).
- `savephases`: when true, the sampled phase screens are saved in the HDF5 in dataset with
  key `"phases"`, and the pupil function is saved under key `"aperture"` (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_images(::Type{T}, img_spec::ImagingSpec{FT}, atm_spec::AtmosphereSpec{FT2},
    truesky::TrueSky=PointSource(); n::Int, batch::Int=512, filename="images.h5", verbose=true,
    savephases::Bool=true, deviceadapter=Array) where {T,FT,FT2}
    if !isfinite_photons(truesky) && T <: Integer
        throw(ArgumentError("Integer image eltype not compatible with infinite-photon true sky model."))
    end
    batch = min(batch, n)
    img_size = img_spec.img_size
    real_img = zeros(T, img_size..., batch)
    psf_norm = sum(abs2, img_spec.aperture) * prod(img_size) *
        sum(img_spec.filter_spec.intensities .* img_spec.filter_spec.wavelengths .^ 2) /
        img_spec.filter_spec.base_wavelength ^ 2
    truesky_adapt = adapt(deviceadapter, convert(TrueSky{FT}, truesky))
    phasebuffers = prepare_phasebuffers(atm_spec, batch, deviceadapter)
    imgbuffers = prepare_imgbuffers(T, img_spec, batch, deviceadapter)

    h5open(filename, "w") do fid
        img_dataset = create_dataset(fid, "images", T, (img_size..., n), chunk=(img_size..., batch))
        p = Progress(n, "Simulating images", enabled=verbose, dt=1)
        if savephases
            phs_size = plate_size(phasebuffers)
            fid["aperture"] = img_spec.aperture
            phs_dataset = create_dataset(fid, "phases", FT2, (phs_size..., n), chunk=(phs_size..., batch))
            phase_buf_h5 = zeros(FT2, phs_size..., batch)
        end
        for j in 1:cld(n, batch)
            phases = samplephases!(phasebuffers)
            imagephases!(real_img, imgbuffers, phases, truesky_adapt, psf_norm)
            HDF5.write_chunk(img_dataset, j - 1, real_img)
            if savephases
                if phases isa Array
                    HDF5.write_chunk(phs_dataset, j - 1, phases)
                else
                    copyto!(phase_buf_h5, phases)
                    HDF5.write_chunk(phs_dataset, j - 1, phase_buf_h5)
                end
            end
            next!(p, step=min(batch, n - (j - 1) * batch))
        end
        finish!(p)
    end
end
simulate_images(img_spec::ImagingSpec, phase_sampler::AtmosphereSpec, true_sky::TrueSky=PointSource(); kwargs...) =
    simulate_images(isfinite_photons(true_sky) ? Int : Float64, img_spec, phase_sampler, true_sky; kwargs...)
