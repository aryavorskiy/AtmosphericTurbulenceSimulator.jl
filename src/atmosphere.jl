using LinearAlgebra, HDF5, Random, ChunkSplitters
import Adapt: adapt_storage

const DEFAULT_WAVELEN = 550nm
abstract type AtmosphereSpec{T} end

_as_wavelength(x::Unitful.Length) = x
_as_wavelength(x::Unitful.Quantity) =
    throw(ArgumentError("Wavelengths must be length quantities, got $(typeof(x))."))
_as_wavelength(x::Number) = x * nm
numtype(x) = typeof(ustrip(x))
convert_numtype(::Type{T}, x) where T = convert.(T, ustrip.(x)) .* unit(eltype(x))

"""
    kolmogorov_covmat(W[, r₀])
    kolmogorov_covmat([T, ]size[, r₀])

Compute the phase covariance matrix of a turbulent layer in the atmosphere, following the Kolmogorov model. The piston
term is excluded in this model.

# Arguments
- `T`: number type for the covariance matrix (default matches `r₀` if provided, otherwise `eltype(W)` or `Float64`).
- `W`: the aperture function as a 2D array of weights. Normalized to `sum(W) == 1` internally.
- `size`: a tuple `(nx, ny)` specifying the size of the aperture function.
- `r₀`: Fried parameter (``r_0``) in pixel units (default `1.0`).
"""
function kolmogorov_covmat(W::AbstractMatrix, r₀::Number=one(eltype(W)))
    I = eachindex(IndexCartesian(), W)
    C = similar(W, float(typeof(r₀)), length(I), length(I))
    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        C[i, j] = -0.5 * 6.88 * ((x^2 + y^2) / r₀^2)^(5/6)
    end
    Wp = oftype.(float(r₀), W ./ sum(W))
    Cp = vec(sum(C .* vec(Wp)', dims=2))
    Cc = sum(Cp .* vec(Wp))
    return C .- (Cp .+ Cp') .+ Cc
end
kolmogorov_covmat(::Type{T}, sz::NTuple{2,Int}, r₀::Number=1.0) where T = kolmogorov_covmat(ones(sz), convert(T, r₀))
kolmogorov_covmat(sz::NTuple{2,Int}, r₀::Number=1.0) = kolmogorov_covmat(typeof(r₀), sz, r₀)

const EigenType = Union{Tuple{<:Any,<:Any}, Eigen}
struct KarhunenLoeveBuffers{MT}
    shape::NTuple{2,Int}
    noise_buffer::MT
    noise_transform::MT
    out_array::MT
end
function KarhunenLoeveBuffers(sz::NTuple{2,Int}, (E, U)::EigenType, batch::Int)
    @assert length(E) == prod(sz)
    @assert size(U) == (length(E), length(E))
    E .= max.(E, zero(eltype(E)))
    noise_transform = U .* sqrt.(E')
    noise_buffer = similar(U, size(U, 2), batch)
    out_array = similar(U, prod(sz), batch)
    KarhunenLoeveBuffers(sz, noise_buffer, noise_transform, out_array)
end
plate_size(sampler::KarhunenLoeveBuffers) = sampler.shape
batch_length(sampler::KarhunenLoeveBuffers) = size(sampler.noise_buffer, 2)
phase_type(sampler::KarhunenLoeveBuffers) = eltype(sampler.out_array)
function samplephases!(sampler::KarhunenLoeveBuffers)
    randn!(sampler.noise_buffer)
    mul!(sampler.out_array, sampler.noise_transform, sampler.noise_buffer)
    return reshape(sampler.out_array, (sampler.shape..., size(sampler.out_array, 2)))
end

struct HardingSpec
    size_to::NTuple{2,Int}
    size_from::NTuple{2,Int}
    nsteps::Int
end
function HardingSpec(final_size::NTuple{2,Int}; interpolate=0, interpolate_from=nothing, size_heuristics=1024)
    if interpolate_from !== nothing
        any(interpolate_from .≤ 11) &&
            throw(ArgumentError("`interpolate_from` dimensions must be greater than 11."))
        interpolated_size = interpolate_from
        n = 0
        while any(final_size .> interpolated_size)
            interpolated_size = 2 .* interpolated_size .- 11
            n += 1
        end
        return HardingSpec(final_size, interpolate_from, n)
    elseif interpolate isa Number
        interpolate_from = cld.(final_size .- 11, 2^interpolate) .+ 11
        return HardingSpec(final_size, interpolate_from=interpolate_from)
    elseif interpolate === :auto
        n = 0
        interpolate_from = final_size
        while prod(interpolate_from) .> size_heuristics
            n += 1
            interpolate_from = cld.(final_size .- 11, 2^n) .+ 11
        end
        return HardingSpec(final_size, interpolate_from, n)
    else
        throw(ArgumentError("`interpolate` must be a Number or :auto"))
    end
end

"""
    SingleLayer([T, ]r0[; base_wavelength, wind_velocity, interpolate, interpolate_from, size_heuristics=1024])

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `T`: the number type for phase screens (default `Float64`).
- `r0`: Fried parameter (``r_0``) in length units (e.g. `0.2m`). A plain number is interpreted
    in whatever units `d`/`grid_step` uses.

# Keyword Arguments
- `base_wavelength`: reference wavelength used to scale phase screens when a multi-wavelength
  `FilterSpec` is used in length units (default 550 nm). A plain number is assumed to be
  nanometers.
- `wind_velocity`: two-component `(vx, vy)` wind velocity used for long-exposure offsets in
  imaging simulations in velocity units (default `(0, 0)`). A plain number is assumed to be
  in whatever units `d`/`grid_step` and `exposure` use.
- `interpolate`: when specified, the phase screen is sampled at a lower resolution and
    then upsampled using specified number of Harding interpolation passes. If set to `:auto`,
    the number of passes is chosen such that the low-res grid has at most `size_heuristics` total pixels.
- `interpolate_from`: alternatively, specify the low-res grid size directly. This must be
    greater than `(11, 11)` in each dimension.
- `size_heuristics`: when `interpolate=:auto`, the maximum allowed number of pixels
    in the low-res grid. Tweak this based on the capability of your hardware to compute `eigen`
    of a `N×N` matrix, where `N` is the number of pixels in the low-res grid.

# Notes
The Harding interpolation follows "Fast simulation of a Kolmogorov phase screen"
Cressida M. Harding, Rachel A. Johnston, and Richard G. Lane, APPLIED OPTICS Vol. 38, No. 11, April 1999
"""
struct SingleLayer{T<:Number,T2<:Number,T3<:Number,KT} <: AtmosphereSpec{T}
    r₀::T
    base_wavelength::T2
    wind_velocity::NTuple{2,T3}
    harding_kw::KT
end
function SingleLayer(r0::Number; base_wavelength=DEFAULT_WAVELEN, wind_velocity=(0, 0), kw...)
    wl = _as_wavelength(base_wavelength)
    SingleLayer(float(r0), convert_numtype(numtype(float(r0)), wl), wind_velocity, kw)
end
SingleLayer(::Type{T}, r0::Number; kw...) where T = SingleLayer(convert_numtype(T, r0); kw...)
function prepare_phasebuffers(spec::SingleLayer, plate_size::NTuple{2,Int}, plate_step::Number, batch::Int, deviceadapter)
    harding = HardingSpec(plate_size; spec.harding_kw...)
    low_size = harding.size_from
    low_r₀_px = oftype(ustrip(spec.r₀), NoUnits(spec.r₀ / plate_step) / 2^harding.nsteps)
    covar_host = kolmogorov_covmat(low_size, low_r₀_px)
    if device_eigen(deviceadapter)
        covar = adapt_storage(deviceadapter, covar_host)
        E, U = eigen(Symmetric(covar))
    else
        E_host, U_host = eigen(Symmetric(covar_host))
        E = adapt_storage(deviceadapter, E_host)
        U = adapt_storage(deviceadapter, U_host)
    end
    kl = KarhunenLoeveBuffers(low_size, (E, U), batch)
    return HardingInterpolator(kl, low_r₀_px, harding, deviceadapter)
end

struct HardingBuffers{NAT}
    out_bufs::NAT
    noise_std::Float64
end
function harding_interpolate!(to, hbuf::HardingBuffers, from)
    N = length(hbuf.out_bufs)
    harding_upsample!(hbuf.out_bufs[1], from, hbuf.noise_std)
    for i in 2:N
        prev_buffer = hbuf.out_bufs[i - 1]
        harding_upsample!(hbuf.out_bufs[i], @view(prev_buffer[6:end-5, 6:end-5, :]),
            hbuf.noise_std / 2^(5/6 * (i-1)))
    end
    out_buffer = hbuf.out_bufs[N]
    crop_offset = (size(out_buffer)[1:2] .- size(to)[1:2]) .÷ 2
    copyto!(to, view(out_buffer,
        crop_offset[1] + 1:crop_offset[1] + size(to, 1),
        crop_offset[2] + 1:crop_offset[2] + size(to, 2), :))
end
function harding_upsample!(to, from, noise_std_e)
    T = eltype(to)
    c_d = convert(T, 0.3198)
    c_m = convert(T, -0.0341)
    c_f = convert(T, -0.0017)
    noise_std = convert(T, noise_std_e)

    # Padding offset
    n, m = size(from)
    inds_odd_x = range(5, length=n-4, step=2)
    inds_odd_y = range(5, length=m-4, step=2)
    inds_even_x = range(4, length=n-3, step=2)
    inds_even_y = range(4, length=m-3, step=2)

    # Copy low-res
    @views copy!(to[1:2:end, 1:2:end, :], from)

    # Interpolate checker pattern sites
    randn!(@view to[inds_even_x, inds_even_y, :])
    @views @. to[inds_even_x, inds_even_y, :] =
        noise_std * to[inds_even_x, inds_even_y, :] +
        c_d * (to[inds_even_x .+ 1, inds_even_y .+ 1, :] + to[inds_even_x .+ 1, inds_even_y .- 1, :] +
               to[inds_even_x .- 1, inds_even_y .+ 1, :] + to[inds_even_x .- 1, inds_even_y .- 1, :]) +
        c_m * ((to[inds_even_x .+ 3, inds_even_y .+ 1, :] + to[inds_even_x .+ 3, inds_even_y .- 1, :] +
               to[inds_even_x .- 3, inds_even_y .+ 1, :] + to[inds_even_x .- 3, inds_even_y .- 1, :]) +
               (to[inds_even_x .+ 1, inds_even_y .+ 3, :] + to[inds_even_x .+ 1, inds_even_y .- 3, :] +
               to[inds_even_x .- 1, inds_even_y .+ 3, :] + to[inds_even_x .- 1, inds_even_y .- 3, :])) +
        c_f * (to[inds_even_x .+ 3, inds_even_y .+ 3, :] + to[inds_even_x .+ 3, inds_even_y .- 3, :] +
               to[inds_even_x .- 3, inds_even_y .+ 3, :] + to[inds_even_x .- 3, inds_even_y .- 3, :])

    # Fill remaining sites
    noise_std_2 = convert(T, noise_std * 2^(-5/12))
    randn!(@view to[inds_odd_x, inds_even_y, :])
    @views @. to[inds_odd_x, inds_even_y, :] =
        noise_std_2 * to[inds_odd_x, inds_even_y, :] +
        c_d * (to[inds_odd_x, inds_even_y .+ 1, :] + to[inds_odd_x, inds_even_y .- 1, :] +
                to[inds_odd_x .+ 1, inds_even_y, :] + to[inds_odd_x .- 1, inds_even_y, :]) +
        c_m * (to[inds_odd_x .+ 1, inds_even_y .+ 2, :] + to[inds_odd_x .+ 1, inds_even_y .- 2, :] +
                to[inds_odd_x .- 1, inds_even_y .+ 2, :] + to[inds_odd_x .- 1, inds_even_y .- 2, :] +
                to[inds_odd_x .+ 2, inds_even_y .+ 1, :] + to[inds_odd_x .+ 2, inds_even_y .- 1, :] +
                to[inds_odd_x .- 2, inds_even_y .+ 1, :] + to[inds_odd_x .- 2, inds_even_y .- 1, :]) +
        c_f * (to[inds_odd_x .+ 3, inds_even_y, :] + to[inds_odd_x .- 3, inds_even_y, :] +
                to[inds_odd_x, inds_even_y .+ 3, :] + to[inds_odd_x, inds_even_y .- 3, :])

    randn!(@view to[inds_even_x, inds_odd_y, :])
    @views @. to[inds_even_x, inds_odd_y, :] =
        noise_std_2 * to[inds_even_x, inds_odd_y, :] +
        c_d * (to[inds_even_x .+ 1, inds_odd_y, :] + to[inds_even_x .- 1, inds_odd_y, :] +
                to[inds_even_x, inds_odd_y .+ 1, :] + to[inds_even_x, inds_odd_y .- 1, :]) +
        c_m * (to[inds_even_x .+ 1, inds_odd_y .+ 2, :] + to[inds_even_x .+ 1, inds_odd_y .- 2, :] +
                to[inds_even_x .- 1, inds_odd_y .+ 2, :] + to[inds_even_x .- 1, inds_odd_y .- 2, :] +
                to[inds_even_x .+ 2, inds_odd_y .+ 1, :] + to[inds_even_x .+ 2, inds_odd_y .- 1, :] +
                to[inds_even_x .- 2, inds_odd_y .+ 1, :] + to[inds_even_x .- 2, inds_odd_y .- 1, :]) +
        c_f * (to[inds_even_x .+ 3, inds_odd_y, :] + to[inds_even_x .- 3, inds_odd_y, :] +
                to[inds_even_x, inds_odd_y .+ 3, :] + to[inds_even_x, inds_odd_y .- 3, :])
end

"""
    ComputeBackend([adapter; nthreads, device_eigen=true])

Device adapter that enables multi-threaded phase generation and imaging on CPU.

# Arguments
- `adapter`: optional storage adapter for internal buffers.

# Keyword Arguments
- `nthreads`: number of threads to use (default: `Threads.nthreads()` if `adapter` is not
    provided, otherwise 1).
- `device_eigen`: whether the Karhunen-Loeve eigendecomposition is computed on device
    (default `true`).
"""
struct ComputeBackend{AT}
    adapter::AT
    nthreads::Int
    device_eigen::Bool
    ComputeBackend(adapter, nthreads::Int, device_eigen::Bool) =
        new{typeof(adapter)}(adapter, nthreads, device_eigen)
    ComputeBackend(::Type{T}, nthreads::Int, device_eigen::Bool) where {T} =
        new{Val{T}}(Val(T), nthreads, device_eigen)
end
ComputeBackend(adapter=identity;
    nthreads::Int=adapter === identity ? Threads.nthreads() : 1,
    device_eigen::Bool=true) =
    ComputeBackend(adapter, nthreads, device_eigen)
adapt_storage(am::ComputeBackend{AT}, x) where {AT} = adapt_storage(am.adapter, x)
adapt_storage(::ComputeBackend{Val{AT}}, x) where {AT} = adapt_storage(AT, x)
device_eigen(adapter::ComputeBackend) = adapter.device_eigen
device_eigen(::Any) = false

MultiThreaded(adapter, nthreads::Int=1) = ComputeBackend(adapter, nthreads, false)
MultiThreaded(nthreads::Int=Threads.nthreads()) = ComputeBackend(identity, nthreads, false)

struct HardingInterpolator{BT,HBT,AT,CT}
    phs_buf::BT
    harding_bufs::Vector{HBT}
    chunk_ranges::Vector{CT}
    out_array::AT
end
function HardingInterpolator(phs_buf, r0::Number, hspec::HardingSpec, adapter::ComputeBackend)
    low_size = hspec.size_from
    any(low_size .≤ 11) && throw(ArgumentError("Dimensions must be greater than 11"))
    T = phase_type(phs_buf)
    batch = batch_length(phs_buf)
    nbufs = min(adapter.nthreads, batch)
    chunk_ranges = collect(chunks(1:batch; n=nbufs))
    noise_std = sqrt(0.5265 / r0^(5/3))
    harding_bufs = map(chunk_ranges) do chunk_range
        bufs = map(1:hspec.nsteps) do i
            lsz = low_size
            for _ in 1:i
                lsz = 2 .* lsz .- 11
            end
            adapt_storage(adapter, zeros(T, (lsz .+ 10)..., length(chunk_range)))
        end
        HardingBuffers(bufs, noise_std)
    end
    out_array = adapt_storage(adapter, zeros(T, hspec.size_to..., batch))
    return HardingInterpolator(phs_buf, harding_bufs, chunk_ranges, out_array)
end
HardingInterpolator(phs_buf, r0::Number, hspec::HardingSpec, adapter) =
    HardingInterpolator(phs_buf, r0, hspec, ComputeBackend(adapter))
plate_size(sampler::HardingInterpolator) = size(sampler.out_array)[1:2]
batch_length(sampler::HardingInterpolator) = batch_length(sampler.phs_buf)
phase_type(sampler::HardingInterpolator) = eltype(sampler.out_array)

function samplephases!(harding::HardingInterpolator)
    low = samplephases!(harding.phs_buf)
    N = length(first(harding.harding_bufs).out_bufs)
    N == 0 && return low
    if length(harding.harding_bufs) == 1
        harding_interpolate!(harding.out_array, only(harding.harding_bufs), low)
    else
        Threads.@threads for i in eachindex(harding.harding_bufs)
            chunk = harding.chunk_ranges[i]
            harding_interpolate!(view(harding.out_array, :, :, chunk),
                harding.harding_bufs[i], view(low, :, :, chunk))
        end
    end
    return harding.out_array
end

"""
    SavedPhases(dataset[, d]; wind_velocity, base_wavelength, grid_step)

An `AtmosphereSpec` that reuses phase screens saved in a dataset.

The dataset must match the phase-screen layout written by [`simulate_phases`](@ref) and
[`simulate_images`](@ref). Frames are read in order; if more frames are requested than
available, an error is thrown, there is no guarantee what is in the tail of the last batch.

# Arguments
- `dataset`: a 3D array-like containing phase screens with dimensions `(nx, ny, nframes)`.
  This can be an in-memory array or an HDF5 dataset (e.g. `HDF5.Dataset`).
- `d`: aperture diameter of the saved phase screens (in the same units as ``r_0``). Optional;
  when omitted grid step defaults to `1` (one pixel = one unit). Overriden by `grid_step`.

# Keyword Arguments
- `base_wavelength`: reference wavelength in nm used to scale phase screens when a multi-wavelength
  `FilterSpec` is used (default 550 nm).
- `wind_velocity`: two-component `(vx, vy)` wind velocity used for long-exposure offsets in
  imaging simulations (default `(0, 0)`).
- `grid_step`: physical size of one aperture pixel in the same units as ``r_0``. Overrides the
  value derived from `d` when provided.
"""
struct SavedPhases{T<:Real,D,WT,WL,GT} <: AtmosphereSpec{T}
    dataset::D
    wind_velocity::NTuple{2,WT}
    base_wavelength::WL
    grid_step::GT
end
function SavedPhases(dataset, d::Union{Number,Nothing}=nothing;
        wind_velocity::NTuple{2,<:Number}=(0, 0), base_wavelength=DEFAULT_WAVELEN,
        grid_step=nothing)
    T = eltype(dataset)
    WT = typeof(wind_velocity[1])
    base_wl = _as_wavelength(base_wavelength)
    grid_step_final = grid_step !== nothing ? grid_step :
        d !== nothing ? d / maximum(size(dataset)[1:2]) : nothing
    return SavedPhases{T,typeof(dataset),WT,typeof(base_wl),typeof(grid_step_final)}(
        dataset, wind_velocity, base_wl, grid_step_final)
end

mutable struct SavedPhaseBuffers{BDT, BT, CT}
    bd::BDT
    crop_indices::CT
    batch_idx::Int
    out_array::BT
end
plate_size(sampler::SavedPhaseBuffers) = size(sampler.out_array)[1:2]
batch_length(sampler::SavedPhaseBuffers) = size(sampler.out_array, 3)
phase_type(sampler::SavedPhaseBuffers) = eltype(sampler.out_array)
function prepare_phasebuffers(spec::SavedPhases{T}, plate_size::NTuple{2,Int},
        grid_step::Number, batch::Int, deviceadapter) where T
    spec.grid_step !== nothing && (spec.grid_step != grid_step) && throw(ArgumentError(
        "Saved phase dataset has grid step $(spec.grid_step), but $grid_step was requested."
    ))
    saved_size = size(spec.dataset)::NTuple{3,Int}
    saved_plate_size = saved_size[1:2]
    all(saved_plate_size .>= plate_size) || throw(ArgumentError(
        "Saved phase dataset has plate size $saved_plate_size, but at least $plate_size was requested."
    ))
    crop_indices = (1:plate_size[1], 1:plate_size[2])
    bd = BufferedDataset(spec.dataset, batch)
    out_array = adapt_storage(deviceadapter, zeros(T, (plate_size..., batch)))
    return SavedPhaseBuffers(bd, crop_indices, 1, out_array)
end
function samplephases!(sampler::SavedPhaseBuffers)
    read_batch!(sampler.out_array, sampler.bd, sampler.batch_idx,
        sampler.crop_indices[1], sampler.crop_indices[2])
    sampler.batch_idx += 1
    return sampler.out_array
end
