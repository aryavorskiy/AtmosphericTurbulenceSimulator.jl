using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, ChunkSplitters
import Adapt: adapt, adapt_storage, adapt_structure

"""
    FilterSpec

Representation of a spectral filter used by the imaging pipeline.

---
    FilterSpec(wavelengths[, intensities])

# Arguments
- `wavelengths`: vector of sampled wavelengths within the filter bandpass. Should be a Unitful
  length; plain numbers are assumed to be nanometers.
- `intensities`: vector of relative intensities at each sampled wavelength. If not provided,
  equal weights are assumed.
"""
struct FilterSpec{T1,T2}
    wavelengths::Vector{T1}
    intensities::Vector{T2}
    FilterSpec{T1,T2}(wavelengths, intensities) where {T1,T2} = new{T1,T2}(wavelengths, intensities)
end
function FilterSpec(wavelengths::AbstractVector, intensities::AbstractVector=ones(Int, length(wavelengths)))
    wl = _as_wavelength.(wavelengths)
    return FilterSpec{eltype(wl),eltype(intensities)}(wl, intensities)
end
nwavel(fs::FilterSpec) = length(fs.wavelengths)

"""
    FilterSpec(base_wavelength; [bandwidth=0, tcenter=1, tedge=1, npts=7])

# Arguments
- `base_wavelength`: central wavelength for the filter. Should be a Unitful length; a plain
  number is assumed to be nanometers.

# Keyword Arguments
- `bandwidth`: total width of the filter bandpass in wavelength units. If zero, the filter is monochromatic.
- `tcenter`: relative intensity at the center wavelength (default 1).
- `tedge`: relative intensity at the edges of the bandpass (default 1).
- `npts`: number of sample points across the bandpass (default 7).
"""
function FilterSpec(base_wavelength::Number=DEFAULT_WAVELEN; bandwidth=0, tcenter=1, tedge=1, npts=7)
    base = _as_wavelength(base_wavelength)
    iszero(bandwidth) && return FilterSpec([base], [tcenter])
    bw_units = _as_wavelength(bandwidth)
    wavelengths = range(base - bw_units / 2, base + bw_units / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tcenter - tedge) + tedge
    return FilterSpec(wavelengths, intensities)
end

struct BilinearInterpolator{IT,VT}
    ix::IT
    iy::IT
    ixp1::IT
    iyp1::IT
    tx::VT
    ty::VT
    can_ff::Bool
end
function BilinearScale(to::AbstractArray, scale::Real)
    nx, ny = size(to)
    sx = range((1 - nx ÷ 2) / scale + nx ÷ 2 + 1, step=1/scale, length=nx)
    sy = range((1 - ny ÷ 2) / scale + ny ÷ 2 + 1, step=1/scale, length=ny)
    ix = copy!(similar(to, Int, nx), clamp.(floor.(Int, sx), 1, nx))
    iy = copy!(similar(to, Int, ny), clamp.(floor.(Int, sy), 1, ny))
    ixp1 = copy!(similar(to, Int, nx), clamp.(ceil.(Int, sx), 1, nx))
    iyp1 = copy!(similar(to, Int, ny), clamp.(ceil.(Int, sy), 1, ny))
    tx = oftype(vec(to), sx) .- ix
    ty = oftype(vec(to), sy) .- iy
    return BilinearInterpolator(ix, iy, ixp1, iyp1, tx, ty, scale≈1)
end
function BilinearShift(to::AbstractArray, offset::NTuple{2})
    nx, ny = size(to)
    ix = range(floor(Int, 1 + offset[1]), length=nx)
    iy = range(floor(Int, 1 + offset[2]), length=ny)
    ixp1 = range(ceil(Int, 1 + offset[1]), length=nx)
    iyp1 = range(ceil(Int, 1 + offset[2]), length=ny)
    tx = convert(eltype(to), offset[1] - floor(offset[1]))
    ty = convert(eltype(to), offset[2] - floor(offset[2]))
    return BilinearInterpolator(ix, iy, ixp1, iyp1, tx, ty, all(isinteger, offset))
end
function interpolate_mapmuladd!(to::AbstractArray, from::AbstractArray, interp::BilinearInterpolator,
        factor, g=identity, h=identity)
    tx = interp.tx
    ty = transpose(interp.ty)
    if interp.can_ff
        @views @inbounds @. to += factor * g(h(from[interp.ix, interp.iy, :]))
    else
        @views @inbounds @. to += factor * g(
            (1 - tx) * (1 - ty) * h(from[interp.ix, interp.iy, :]) +
            tx * (1 - ty) * h(from[interp.ixp1, interp.iy, :]) +
            (1 - tx) * ty * h(from[interp.ix, interp.iyp1, :]) +
            tx * ty * h(from[interp.ixp1, interp.iyp1, :]))
    end
    return to
end

"""
    PhotonCount(nphotons[, background]; gaussian_approx=false)

Specifies the photon budget for imaging simulations. Set `nphotons` to `Inf` for
continuous flux (`background` can be omitted in this case).

# Keyword Arguments
- `gaussian_approx`: when true, Poisson noise is approximated by either a rounded Gaussian
  or an exact sampler (for λ < 10, Knuth counting). Default `false` (exact everywhere via `Distributions`).
"""
struct PhotonCount{GA,T<:Real}
    nphotons::T
    background::T
    PhotonCount{GA,T}(nphotons, background) where {GA,T<:Real} =
        new{GA,T}(convert(T, nphotons), convert(T, background))
    function PhotonCount{GA}(nphotons::T1, background::T2) where {T1<:Real,T2<:Real, GA}
        return PhotonCount{GA,promote_type(T1, T2)}(nphotons, background)
    end
end
function PhotonCount(nph, bg=nothing; gaussian_approx=false)
    if isinf(nph)
        PhotonCount{gaussian_approx}(nph, zero(nph))
    else
        isnothing(bg) && throw(ArgumentError("Must specify background when `nphotons` is finite"))
        PhotonCount{gaussian_approx}(nph, bg)
    end
end
convert_numtype(::Type{T}, pc::PhotonCount{GA}) where {T, GA} =
    PhotonCount{GA}(convert(T, pc.nphotons), convert(T, pc.background))
isfinite_photons(pc::PhotonCount) = isfinite(pc.nphotons)
@inline gaussian_approx(::PhotonCount{GA}) where {GA} = GA

struct Exposure{ET<:Number}
    exptime::ET
    nsteps::Int
    round_offsets::Bool
end
"""
    Exposure(exptime[, nsteps][; round_offsets])

A struct that encapsulates the exposure time and related parameters for long-exposure simulations.
The exposure is simulated by averaging `nsteps` short exposures with appropriate offsets of the wavefront.

# Arguments
- `exptime`: total exposure time. Shares the same time units as the wind velocity in the atmosphere specification.
- `nsteps`: number of steps to simulate for long exposures (default 5).
- `round_offsets`: whether to round the exposure offsets to integers (default false). When true,
  the phase screens are sampled at integer pixel offsets, which can reduce interpolation artifacts.
"""
function Exposure(exptime, nsteps=5; round_offsets=false)
    nsteps == 1 && !iszero(exptime) &&
        @warn "Ignoring non-zero exposure time for single-step exposure."
    if iszero(exptime)
        nsteps = 1
    end
    Exposure(exptime, nsteps, round_offsets)
end

abstract type TrueSky end

"""
    PointSource()

Simple true-sky brightness model. Represents a non-resolved point source.
"""
struct PointSource <: TrueSky end

"""
    DoubleSystem(rel_position, intensity)

Model for a two-component source (binary): primary plus a secondary offset by `rel_position`.

# Arguments
- `rel_position`: `(dx, dy)` integer tuple specifying the secondary's pixel offset.
- `intensity`: multiplicative intensity of the secondary relative to the primary.
"""
struct DoubleSystem{T} <: TrueSky
    rel_position::NTuple{2,Int}
    intensity::T
    DoubleSystem(position, intensity::Real) =
        new{typeof(intensity)}(Tuple(position), intensity)
end

"""
    TrueSkyImage(true_sky::AbstractMatrix{T})

Wrap a real-valued true-sky image for use with the imaging pipeline. The pixel scale matches
that of the `ImagingSpec`.

# Arguments
- `true_sky`: real image array representing spatial sky brightness.
"""
struct TrueSkyImage{MT<:AbstractMatrix{<:Complex}} <: TrueSky
    true_sky_fft::MT
end
function TrueSkyImage(true_sky::AbstractMatrix{T}) where {T<:Real}
    true_sky_fft = fft(ifftshift(true_sky))
    true_sky_fft ./= true_sky_fft[1, 1]  # normalize DC component to 1
    return TrueSkyImage{typeof(true_sky_fft)}(true_sky_fft)
end
adapt_structure(to, ts::TrueSkyImage) = TrueSkyImage(adapt_storage(to, ts.true_sky_fft))

const FilterSpecType{T} = FilterSpec{<:Union{T, Quantity{T}}, <:Union{T, Quantity{T}}}
"""
    ImagingSpec

Container for the imaging system configuration. It is defined by the telescope `aperture`, the
source brightness via `photon_count`, an optional spectral `filter_spec`, the `exposure`,
and the output `img_size`.
If `img_size` does not match the aperture’s Nyquist grid, the aperture is zero-padded accordingly.
"""
struct ImagingSpec{T, T2, GA, AT<:AbstractMatrix{T}, FST<:FilterSpec}
    aperture::AT
    grid_step::T2
    photon_count::PhotonCount{GA,T}
    filter_spec::FST
    exposure::Exposure
    img_size::NTuple{2,Int}
    function ImagingSpec(
        aperture::AbstractMatrix{T},
        grid_step::Number,
        photon_count::PhotonCount{GA,T},
        filter_spec::FilterSpecType{T},
        exposure::Exposure,
        img_size::NTuple{2,Int}) where {T<:Real, GA}
        new{T, typeof(grid_step), GA, typeof(aperture), typeof(filter_spec)}(
            aperture, grid_step, photon_count, filter_spec, exposure, img_size)
    end
end

"""
    ImagingSpec([T, ]aperture, d, photon_count[; grid_step, filter, nyquist_oversample, img_size])

Create an imaging system specification.

# Arguments
- `T`: desired number type, inferred from `aperture` if not provided.
- `aperture`: 2D aperture (pupil) array describing the telescope pupil.
- `d`: aperture diameter in the same units as ``r_0`` (e.g. `2m`). Used to compute `grid_step` as
  `d / maximum(size(aperture))`.
- `photon_count`: `PhotonCount` instance describing the photon budget and background.

# Keyword Arguments
- `filter`: `FilterSpec` describing sampled wavelengths and their relative intensities.
  Defaults to a monochromatic filter at 550 nm.
- `exposure`: a number or an [`Exposure`](@ref) instance describing the exposure time and number of
  steps for long exposures. Defaults to zero exposure time (i.e. short exposure).
- `nyquist_oversample`: multiplicative factor applied to the default Nyquist image size (`2 * size(aperture)`).
  Defaults to 1. Ignored if `img_size` is provided.
- `img_size`: explicit output image size `(nx, ny)`. If not provided, computed from aperture size
  and `nyquist_oversample`.
"""
function ImagingSpec(aperture::AbstractMatrix{T}, d::Number, photon_count::PhotonCount;
    filter::FilterSpec=FilterSpec(DEFAULT_WAVELEN),
    nyquist_oversample::Real=1, exposure::Union{Exposure,Number}=0,
    img_size::NTuple{2,Int}=round.(Int, size(aperture) .* 2 .* nyquist_oversample)) where T<:Real
    pc = convert_numtype(T, photon_count)
    ex = exposure isa Number ? Exposure(exposure) : exposure
    ft = FilterSpec(
        convert_numtype(T, filter.wavelengths), convert_numtype(T, filter.intensities))
    return ImagingSpec(aperture, d/maximum(size(aperture)), pc, ft, ex, img_size)
end
ImagingSpec(::Type{T}, aperture::AbstractMatrix, args...; kw...) where T<:Real =
    ImagingSpec(convert.(T, aperture), args...; kw...)

adapt_structure(to, img_spec::ImagingSpec) =
    ImagingSpec(adapt_storage(to, img_spec.aperture), img_spec.grid_step,
    img_spec.photon_count, img_spec.filter_spec, img_spec.exposure, img_spec.img_size)
plate_size(img_spec::ImagingSpec) = size(img_spec.aperture)
image_size(img_spec::ImagingSpec) = img_spec.img_size
psf_norm(img_spec::ImagingSpec) = sum(abs2, img_spec.aperture) * prod(img_spec.img_size) *
        sum(img_spec.filter_spec.intensities)

struct OpticalBuffers{MT, MT2, MT3, PT, IT}
    aperture_buffer::MT
    focal_buffer::MT
    psf_buffer::MT2
    read_buffer::MT3
    fftplan::PT
    interpolators::Vector{IT}
end
image_size(bufs::OpticalBuffers) = size(bufs.aperture_buffer)[1:2]
image_type(bufs::OpticalBuffers) = eltype(bufs.read_buffer)
batch_length(bufs::OpticalBuffers) = size(bufs.aperture_buffer, 3)

function OpticalBuffers(::Type{T}, img_spec::ImagingSpec{NT}, scales, batch::Int) where {T, NT}
    buf1 = similar(img_spec.aperture, complex(NT), img_spec.img_size..., batch)
    buf2 = similar(buf1)
    psf_buffer = similar(buf1, NT, img_spec.img_size..., batch)
    read_buffer = similar(buf1, T, img_spec.img_size..., batch)
    interpolators = [BilinearScale(psf_buffer, scale) for scale in scales]
    return OpticalBuffers(buf1, buf2, psf_buffer, read_buffer, plan_fft(buf1, (1, 2)), interpolators)
end
function write_phases!(aperture_buffer, phases, aperture, offset, phs_factor)
    M, N = size(aperture)
    Cx, Cy = size(aperture_buffer) .÷ 2
    fill!(aperture_buffer, 0)
    ap_slice = @view aperture_buffer[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :]
    interpolate_mapmuladd!(ap_slice, phases, offset, aperture, cis, Base.Fix2(/, phs_factor))
end
write_phases!(bufs::OpticalBuffers, phases, img_spec::ImagingSpec, offset, phs_factor) =
    write_phases!(bufs.focal_buffer, phases, img_spec.aperture, offset, phs_factor)

function phases_to_psf!(bufs::OpticalBuffers, interpolator::BilinearInterpolator, psf_factor)
    mul!(bufs.aperture_buffer, bufs.fftplan, bufs.focal_buffer)
    fftshift!(bufs.focal_buffer, bufs.aperture_buffer, (1, 2))
    interpolate_mapmuladd!(bufs.psf_buffer, bufs.focal_buffer, interpolator, psf_factor, identity, abs2)
end

function apply_truesky!(opt_buffer::OpticalBuffers, ts::TrueSkyImage)
    copyto!(opt_buffer.focal_buffer, opt_buffer.psf_buffer)
    mul!(opt_buffer.aperture_buffer, opt_buffer.fftplan, opt_buffer.focal_buffer)
    opt_buffer.aperture_buffer .*= ts.true_sky_fft
    ldiv!(opt_buffer.focal_buffer, opt_buffer.fftplan, opt_buffer.aperture_buffer)
    opt_buffer.psf_buffer .= real.(opt_buffer.focal_buffer)
end
function apply_truesky!(opt_buffer::OpticalBuffers, ds::DoubleSystem)
    img = opt_buffer.psf_buffer
    @assert all(abs.(ds.rel_position) .< image_size(opt_buffer) .÷ 2)
    o1, o2 = ds.rel_position
    s1_dest, s1_src = o1 > 0 ? (o1 + 1:size(img, 1), 1:size(img, 1) - o1) : (1:size(img, 1) + o1, -o1 + 1:size(img, 1))
    s2_dest, s2_src = o2 > 0 ? (o2 + 1:size(img, 2), 1:size(img, 2) - o2) : (1:size(img, 2) + o2, -o2 + 1:size(img, 2))
    @views @. img[s1_dest, s2_dest, :] += img[s1_src, s2_src, :] * ds.intensity
    img ./= (1 + ds.intensity)
end
apply_truesky!(::OpticalBuffers, ::PointSource) = nothing

const GAUSSIAN_POISSON_CUTOFF = 10
const KNUTH_MAXITER = round(Int, GAUSSIAN_POISSON_CUTOFF + 5 * sqrt(GAUSSIAN_POISSON_CUTOFF))
function _sample_poisson(::Val{true}, ::Type{T}, λ) where {T}
    if λ < GAUSSIAN_POISSON_CUTOFF
        # Knuth counting algorithm
        L = exp(-λ)
        k = 0
        p = one(λ)
        for _ in 1:KNUTH_MAXITER
            p *= rand(typeof(λ))
            p <= L && break
            k += 1
        end
        return convert(T, k)
    end
    return round(T, max(randn() * sqrt(λ) + λ, zero(λ)))
end
_sample_poisson(::Val{false}, ::Type{T}, λ) where {T} = convert(T, rand(Poisson(λ)))
function readout!(dst::AbstractArray{T}, img::AbstractArray, pc::PhotonCount{GA}, psf_norm) where {T, GA}
    if isfinite_photons(pc)
        @. dst = _sample_poisson($(Val(GA)), T, real(img) / psf_norm * pc.nphotons + pc.background)
    else
        @. dst = img / psf_norm
    end
end
readout!(opt_buffer::OpticalBuffers, pc::PhotonCount, psf_norm) =
    readout!(opt_buffer.read_buffer, opt_buffer.psf_buffer, pc, psf_norm)

function compute_images!(readout_to, opt_buffer::OpticalBuffers, spec::ImagingSpec, phs_factors, phases, true_sky, offsets, psf_norm)
    fill!(opt_buffer.psf_buffer, 0)
    filter = spec.filter_spec
    for offset in offsets, w in 1:nwavel(filter)
        write_phases!(opt_buffer, phases, spec, offset, phs_factors[w])
        phases_to_psf!(opt_buffer, opt_buffer.interpolators[w], filter.intensities[w] / phs_factors[w]^2)
    end
    apply_truesky!(opt_buffer, true_sky)
    readout!(readout_to, opt_buffer.psf_buffer, spec.photon_count, psf_norm * length(offsets))
end

"""
    CircularAperture([T, ]sz, [radius; aa_dist=1])

Create a circular (optionally anti-aliased) aperture array of shape `sz`. Returns a 2D
numeric array suitable for use as an aperture in `ImagingSpec`.

# Arguments
- `T`: desired number type, `Float64` by default.
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

function padded_plate_size(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    if iszero(img_spec.exposure.exptime) || all(iszero, atm_spec.wind_velocity)
        return plate_size(img_spec)
    end
    max_offset = atm_spec.wind_velocity .* img_spec.exposure.exptime ./ img_spec.grid_step
    return plate_size(img_spec) .+ ceil.(Int, abs.(NoUnits.(max_offset)))
end
function long_exp_offsets(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    n = img_spec.exposure.nsteps
    if n == 1 || iszero(img_spec.exposure.exptime) || all(iszero, atm_spec.wind_velocity)
        offset_list = [ustrip.(atm_spec.wind_velocity .* img_spec.exposure.exptime .* 0)]
    else
        offset_list = [NoUnits.(atm_spec.wind_velocity .* (img_spec.exposure.exptime * j / (n - 1) / img_spec.grid_step)) for j in 0:n-1]
    end
    if img_spec.exposure.round_offsets
        offset_list = [round.(offset) for offset in offset_list]
    end
    mins = minimum(first, offset_list), minimum(last, offset_list)
    return [BilinearShift(img_spec.aperture, offset .- mins) for offset in offset_list]
end

struct SimulationBuffers{BT<:OpticalBuffers,ST<:ImagingSpec,FT<:Real,OT,AT,CT,VT}
    opt_bufs::Vector{BT}
    chunk_ranges::Vector{CT}
    spec::ST
    psf_norm::FT
    offsets::Vector{OT}
    img_array::AT
    phs_factors::VT
end
image_size(img_buf::SimulationBuffers) = image_size(img_buf.opt_bufs[1])
image_type(img_buf::SimulationBuffers) = eltype(img_buf.img_array)
function prepare_buffers(::Type{T}, atm_spec, img_spec::ImagingSpec, batch::Int, adapter::ComputeBackend) where T
    nbufs = min(adapter.nthreads, batch)
    chunk_ranges = collect(chunks(1:batch; n=nbufs))
    img_spec_adapt = adapt(adapter, img_spec)
    phs_factors = NoUnits.(img_spec.filter_spec.wavelengths ./ atm_spec.base_wavelength)
    opt_buffer1 = OpticalBuffers(T, img_spec_adapt, phs_factors, length(chunk_ranges[1]))
    img_array = similar(opt_buffer1.read_buffer, image_size(img_spec)..., batch)
    opt_bufs = Array{typeof(opt_buffer1)}(undef, nbufs)
    opt_bufs[1] = opt_buffer1
    Threads.@threads for i in 2:nbufs
        opt_bufs[i] = OpticalBuffers(T, img_spec_adapt, phs_factors, length(chunk_ranges[i]))
    end
    return prepare_phasebuffers(atm_spec, padded_plate_size(atm_spec, img_spec),
            img_spec.grid_step, batch, adapter),
        SimulationBuffers(opt_bufs, chunk_ranges, img_spec_adapt, psf_norm(img_spec),
            long_exp_offsets(atm_spec, img_spec), img_array, phs_factors)
end
prepare_buffers(type, atm_spec, img_spec, batch, A) =
    prepare_buffers(type, atm_spec, img_spec, batch, ComputeBackend(A))
function compute_images!(img_buf::SimulationBuffers, phases, true_sky)
    if length(img_buf.chunk_ranges) == 1
        compute_images!(img_buf.img_array, only(img_buf.opt_bufs), img_buf.spec,
            img_buf.phs_factors, phases, true_sky, img_buf.offsets, img_buf.psf_norm)
    else
        Threads.@threads for i in eachindex(img_buf.opt_bufs)
            chunk_range = img_buf.chunk_ranges[i]
            compute_images!(view(img_buf.img_array, :, :, chunk_range), img_buf.opt_bufs[i], img_buf.spec,
                img_buf.phs_factors, view(phases, :, :, chunk_range),
                true_sky, img_buf.offsets, img_buf.psf_norm)
        end
    end
    return img_buf.img_array
end
