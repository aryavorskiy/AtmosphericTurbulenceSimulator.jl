using LinearAlgebra, HDF5, Random, Adapt

abstract type AtmosphereSpec{T} end

"""
    kolmogorov_covmat(W)

Compute the phase covariance matrix of a turbulent layer in the atmosphere, following the Kolmogorov model. The piston
term is excluded in this model. This function assumes unit Fried parameter r0=1.

# Arguments
- `W`: the aperture function. Either a 2D array, or a `(x, y)` tuple representing the size
    of the aperture (in this case the aperture function is assumed to be a square of this
    size).
"""
function kolmogorov_covmat(W::AbstractMatrix)
    I = eachindex(IndexCartesian(), W)
    C = similar(W, length(I), length(I))
    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        C[i, j] = -0.5 * 6.88 * (x^2 + y^2)^(5/6)
    end
    @assert sum(W) ≈ 1
    Cp = vec(sum(C .* vec(W)', dims=2))
    Cc = sum(Cp .* vec(W))
    return Symmetric(C .- Cp .- Cp' .+ Cc)
end
kolmogorov_covmat(::Type{T}, sz::NTuple{2,Int}) where T =
    kolmogorov_covmat(fill(convert(T, 1/prod(sz)), sz...))
kolmogorov_covmat(sz::NTuple{2,Int}) = kolmogorov_covmat(Float64, sz)

const EigenType = Union{Tuple{<:Any,<:Any}, Eigen}
struct KarhunenLoeveBuffers{MT}
    shape::NTuple{2,Int}
    noise_buffer::MT
    noise_transform::MT
    out_buffer::MT
end
function KarhunenLoeveBuffers(sz::NTuple{2,Int}, (E, U)::EigenType, batch::Int)
    @assert length(E) == prod(sz)
    @assert size(U) == (length(E), length(E))
    E .= clamp.(E, 0, Inf)
    noise_transform = U .* sqrt.(E')
    noise_buffer = similar(U, size(U, 2), batch)
    out_buffer = similar(U, prod(sz), batch)
    KarhunenLoeveBuffers(sz, noise_buffer, noise_transform, out_buffer)
end
plate_size(sampler::KarhunenLoeveBuffers) = sampler.shape
out_buffer(sampler::KarhunenLoeveBuffers) = reshape(sampler.out_buffer, (sampler.shape..., size(sampler.out_buffer, 2)))
batch_length(sampler::KarhunenLoeveBuffers) = size(sampler.noise_buffer, 2)
function samplephases!(sampler::KarhunenLoeveBuffers)
    randn!(sampler.noise_buffer)
    mul!(sampler.out_buffer, sampler.noise_transform, sampler.noise_buffer)
    return out_buffer(sampler)
end

struct HardingSpec{N}
    interpolate_from::NTuple{2,Int}
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
        return HardingSpec{n}(interpolate_from)
    elseif interpolate isa Number
        interpolate_from = cld.(final_size .- 11, 2^interpolate) .+ 11
        return HardingSpec{interpolate}(interpolate_from)
    elseif interpolate === :auto
        n = 0
        interpolate_from = final_size
        while prod(interpolate_from) .> size_heuristics
            n += 1
            interpolate_from = cld.(final_size .- 11, 2^n) .+ 11
        end
    else
        throw(ArgumentError("`interpolate` must be a Number or :auto"))
    end
end

"""
    IndependentFrames(size, r0[; interpolate, interpolate_from, size_heuristics=1024])

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `size`: a tuple `(nx, ny)` specifying the phase screen shape in pixels (coarse sampler grid).
- `r0`: Fried parameter (r₀) in pixels.
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
struct IndependentFrames{T,N} <: AtmosphereSpec{T}
    size::NTuple{2, Int}
    r₀::T
    harding::HardingSpec{N}
end
IndependentFrames(sz::NTuple{2,Int}, r0::T; kw...) where T =
    IndependentFrames(sz, r0, HardingSpec(sz; kw...))
IndependentFrames(::Type{T}, sz::NTuple{2,Int}, r0; kw...) where T =
    IndependentFrames(sz, convert(T, r0), HardingSpec(sz; kw...))
function prepare_phasebuffers(spec::IndependentFrames{T,N}, batch::Int, deviceadapter) where {T,N}
    low_size = spec.harding.interpolate_from
    covar = Adapt.adapt_storage(deviceadapter, kolmogorov_covmat(T, low_size))
    low_r₀ = spec.r₀ / 2^N
    covar .*= low_r₀^(-5/3)
    E, U = eigen(Symmetric(covar))
    kl = KarhunenLoeveBuffers(low_size, (E, U), batch)
    if N == 0
        return kl
    else
        return HardingInterpolator(kl, low_r₀, spec.size, Val(N))
    end
end

"""
Harding interpolator wrapper around another sampler.
Implements two-pass Harding interpolation: N×M → (2N-1)×(2M-1).
First pass fills checker pattern with base stencil, second pass fills
remaining sites with rotated stencil.
"""
struct HardingInterpolator{N,BT,AT}
    base::BT
    out_bufs::NTuple{N,AT}
    noise_std::Float64
    crop_size::NTuple{2,Int}
end

function HardingInterpolator(base, r0::Number, final_size, ::Val{N}) where N
    low_size = plate_size(base)
    any(low_size .≤ 11) && throw(ArgumentError("Dimensions must be greater than 11"))
    buffers = ntuple(Val(N)) do i
        lsz = low_size
        for _ in 1:i
            lsz = 2 .* lsz .- 11
        end
        similar(out_buffer(base), (lsz .+ 10)..., batch_length(base))
    end
    return HardingInterpolator(base, buffers, sqrt(0.5265 / r0^(5/3)), final_size)
end
plate_size(sampler::HardingInterpolator) = sampler.crop_size
out_arrtype(sampler::HardingInterpolator) = typeof(sampler.out_buf)
batch_length(sampler::HardingInterpolator) = size(sampler.out_buf, 3)

function samplephases!(harding::HardingInterpolator{N}) where N
    low = samplephases!(harding.base)
    upsample!(harding.out_bufs[1], low, harding.noise_std)
    for i in 2:N
        old_buf = harding.out_bufs[i - 1]
        upsample!(harding.out_bufs[i], @view(old_buf[6:end-5, 6:end-5, :]), harding.noise_std / 2^(5/6 * (i-1)))
    end
    out_buf = harding.out_bufs[N]
    crop_offset = (size(out_buf)[1:2] .- harding.crop_size) .÷ 2
    return @view out_buf[
        crop_offset[1] + 1:crop_offset[1] + harding.crop_size[1],
        crop_offset[2] + 1:crop_offset[2] + harding.crop_size[2],
        :]
end
function upsample!(out_buf, low, noise_std)
    c_d = 0.3198
    c_m = -0.0341
    c_f = -0.0017
    std_scale = 2^(-5/12)

    # Padding offset: actual data starts at index 4
    n, m = size(low)
    inds_odd_x = (1:n-4) .* 2 .+ 3
    inds_odd_y = (1:m-4) .* 2 .+ 3
    inds_even_x = (1:n-3) .* 2 .+ 2
    inds_even_y = (1:m-3) .* 2 .+ 2

    # Copy low-res
    out_buf[1:2:end, 1:2:end, :] .= low

    # Interpolate checker pattern sites
    @views @. out_buf[inds_even_x, inds_even_y, :] =
        c_d * (out_buf[inds_even_x .+ 1, inds_even_y .+ 1, :] + out_buf[inds_even_x .+ 1, inds_even_y .- 1, :] +
               out_buf[inds_even_x .- 1, inds_even_y .+ 1, :] + out_buf[inds_even_x .- 1, inds_even_y .- 1, :]) +
        c_m * (out_buf[inds_even_x .+ 3, inds_even_y .+ 1, :] + out_buf[inds_even_x .+ 3, inds_even_y .- 1, :] +
               out_buf[inds_even_x .- 3, inds_even_y .+ 1, :] + out_buf[inds_even_x .- 3, inds_even_y .- 1, :] +
               out_buf[inds_even_x .+ 1, inds_even_y .+ 3, :] + out_buf[inds_even_x .+ 1, inds_even_y .- 3, :] +
               out_buf[inds_even_x .- 1, inds_even_y .+ 3, :] + out_buf[inds_even_x .- 1, inds_even_y .- 3, :]) +
        c_f * (out_buf[inds_even_x .+ 3, inds_even_y .+ 3, :] + out_buf[inds_even_x .+ 3, inds_even_y .- 3, :] +
               out_buf[inds_even_x .- 3, inds_even_y .+ 3, :] + out_buf[inds_even_x .- 3, inds_even_y .- 3, :]) +
        noise_std * randn()

    # Fill remaining sites
    @views @. out_buf[inds_odd_x, inds_even_y, :] =
        c_d * (out_buf[inds_odd_x, inds_even_y .+ 1, :] + out_buf[inds_odd_x, inds_even_y .- 1, :] +
                out_buf[inds_odd_x .+ 1, inds_even_y, :] + out_buf[inds_odd_x .- 1, inds_even_y, :]) +
        c_m * (out_buf[inds_odd_x .+ 1, inds_even_y .+ 2, :] + out_buf[inds_odd_x .+ 1, inds_even_y .- 2, :] +
                out_buf[inds_odd_x .- 1, inds_even_y .+ 2, :] + out_buf[inds_odd_x .- 1, inds_even_y .- 2, :] +
                out_buf[inds_odd_x .+ 2, inds_even_y .+ 1, :] + out_buf[inds_odd_x .+ 2, inds_even_y .- 1, :] +
                out_buf[inds_odd_x .- 2, inds_even_y .+ 1, :] + out_buf[inds_odd_x .- 2, inds_even_y .- 1, :]) +
        c_f * (out_buf[inds_odd_x .+ 3, inds_even_y, :] + out_buf[inds_odd_x .- 3, inds_even_y, :] +
                out_buf[inds_odd_x, inds_even_y .+ 3, :] + out_buf[inds_odd_x, inds_even_y .- 3, :]) +
        noise_std * std_scale * randn()

    @views @. out_buf[inds_even_x, inds_odd_y, :] =
        c_d * (out_buf[inds_even_x .+ 1, inds_odd_y, :] + out_buf[inds_even_x .- 1, inds_odd_y, :] +
                out_buf[inds_even_x, inds_odd_y .+ 1, :] + out_buf[inds_even_x, inds_odd_y .- 1, :]) +
        c_m * (out_buf[inds_even_x .+ 1, inds_odd_y .+ 2, :] + out_buf[inds_even_x .+ 1, inds_odd_y .- 2, :] +
                out_buf[inds_even_x .- 1, inds_odd_y .+ 2, :] + out_buf[inds_even_x .- 1, inds_odd_y .- 2, :] +
                out_buf[inds_even_x .+ 2, inds_odd_y .+ 1, :] + out_buf[inds_even_x .+ 2, inds_odd_y .- 1, :] +
                out_buf[inds_even_x .- 2, inds_odd_y .+ 1, :] + out_buf[inds_even_x .- 2, inds_odd_y .- 1, :]) +
        c_f * (out_buf[inds_even_x .+ 3, inds_odd_y, :] + out_buf[inds_even_x .- 3, inds_odd_y, :] +
                out_buf[inds_even_x, inds_odd_y .+ 3, :] + out_buf[inds_even_x, inds_odd_y .- 3, :]) +
        noise_std * std_scale * randn()
end
