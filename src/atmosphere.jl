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
noise_eltype(sampler::KarhunenLoeveBuffers) = eltype(sampler.noise_buffer)
batch_length(sampler::KarhunenLoeveBuffers) = size(sampler.noise_buffer, 2)
function samplephases!(sampler::KarhunenLoeveBuffers)
    randn!(sampler.noise_buffer)
    mul!(sampler.out_buffer, sampler.noise_transform, sampler.noise_buffer)
    return reshape(sampler.out_buffer, (sampler.shape..., size(sampler.out_buffer, 2)))
end

"""
    IndependentFrames(size, r0[; interpolate_steps])

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `size`: a tuple `(nx, ny)` specifying the phase screen shape in pixels (coarse sampler grid).
- `r0`: Fried parameter (r₀) in pixels.
- `interpolate_steps`: when specified, the phase screen is sampled at a lower resolution and
    then upsampled using specified number of Harding interpolation passes.

# Notes
This sampler is intentionally simple: frames are independent between timesteps and are
generated using a Karhunen–Loève transform built from the Kolmogorov covariance.

The Harding interpolation follows "Fast simulation of a Kolmogorov phase screen"
Cressida M. Harding, Rachel A. Johnston, and Richard G. Lane, APPLIED OPTICS Vol. 38, No. 11, April 1999
"""
struct IndependentFrames{T} <: AtmosphereSpec{T}
    size::NTuple{2, Int}
    r₀::T
    interpolate_steps::Int
end
IndependentFrames(sz::NTuple{2,Int}, r0::T; interpolate_steps=0) where T =
    IndependentFrames{T}(sz, r0, interpolate_steps)
IndependentFrames(::Type{T}, sz::NTuple{2,Int}, r0; interpolate_steps=0) where T =
    IndependentFrames{T}(sz, r0, interpolate_steps)

@inline function interpolate_sampler(sampler, r_0, target_size::NTuple{2,Int})
    low_size = plate_size(sampler)
    hi_size = 2 .* low_size .- 1
    if all(target_size .≤ low_size)
        return sampler
    elseif all(target_size .≤ hi_size)
        return HardingInterpolator(sampler, r_0, target_size)
    else
        return interpolate_sampler(
            HardingInterpolator(sampler, r_0), r_0 * 2, target_size, # HACK why r_0 / 2 ???
        )
    end
end
function prepare_phasebuffers(spec::IndependentFrames{T}, batch::Int, deviceadapter) where T
    low_size = spec.size
    for _ in 1:spec.interpolate_steps
        low_size = low_size .÷ 2 .+ 1
    end
    covar = Adapt.adapt_storage(deviceadapter, kolmogorov_covmat(T, low_size))
    low_r₀ = spec.r₀ / 2^(spec.interpolate_steps)
    covar .*= low_r₀^(-5/3)
    E, U = eigen(covar)
    kl = KarhunenLoeveBuffers(low_size, (E, U), batch)
    return interpolate_sampler(kl, low_r₀, spec.size)
end

"""
Harding interpolator wrapper around another sampler.
Implements two-pass Harding interpolation: N×M → (2N-1)×(2M-1).
First pass fills checker pattern with base stencil, second pass fills
remaining sites with rotated stencil.
"""
struct HardingInterpolator{BT,AT}
    base::BT
    out_buf::AT
    noise_std::Float64
    crop_size::NTuple{2,Int}
end

function HardingInterpolator(base, r0::Number, crop_size::NTuple{2,Int}=plate_size(base) .* 2 .- 1)
    low_size = plate_size(base)
    @assert low_size[1] == low_size[2] "HardingInterpolator currently only supports square grids."
    @assert all(crop_size .≤ (2 .* low_size .- 1)) "crop_size must be less than or equal to (2*low_size - 1)."
    padded_size = 2 .* low_size .+ 5
    out = zeros(noise_eltype(base), padded_size..., batch_length(base))
    return HardingInterpolator(base, out, sqrt(0.0844 / r0^(5/3)), crop_size)
end
plate_size(sampler::HardingInterpolator) = sampler.crop_size
noise_eltype(sampler::HardingInterpolator) = eltype(sampler.out_buf)
batch_length(sampler::HardingInterpolator) = size(sampler.out_buf, 3)

function samplephases!(interp::HardingInterpolator)
    low = samplephases!(interp.base)
    n, m = plate_size(interp.base)

    # Harding stencil coefficients (Eq. 27)
    c_d = 0.3198
    c_m = -0.0341
    c_f = -0.0017

    out_buf = interp.out_buf

    # Padding offset: actual data starts at index 4
    offset = 3
    inds_odd = (1:n) .* 2 .- 1 .+ offset
    inds_even = (1:n-1) .* 2 .+ offset

    # Step 1: Copy low-res samples to even sites (1:2:end, 1:2:end)
    out_buf[inds_odd, inds_odd, :] .= low
    out_buf[[offset - 1, end - offset + 2], inds_odd, :] .= @view low[[1, end], :, :]
    out_buf[inds_odd, [offset - 1, end - offset + 2], :] .= @view low[:, [1, end], :]

    # Step 2: Interpolate diagonal sites (2:2:end, 2:2:end) with base stencil
    @views @. out_buf[inds_even, inds_even, :] =
        c_d * (out_buf[inds_even .+ 1, inds_even .+ 1, :] + out_buf[inds_even .+ 1, inds_even .- 1, :] +
               out_buf[inds_even .- 1, inds_even .+ 1, :] + out_buf[inds_even .- 1, inds_even .- 1, :]) +
        c_m * (out_buf[inds_even .+ 3, inds_even .+ 1, :] + out_buf[inds_even .+ 3, inds_even .- 1, :] +
               out_buf[inds_even .- 3, inds_even .+ 1, :] + out_buf[inds_even .- 3, inds_even .- 1, :] +
               out_buf[inds_even .+ 1, inds_even .+ 3, :] + out_buf[inds_even .+ 1, inds_even .- 3, :] +
               out_buf[inds_even .- 1, inds_even .+ 3, :] + out_buf[inds_even .- 1, inds_even .- 3, :]) +
        c_f * (out_buf[inds_even .+ 3, inds_even .+ 3, :] + out_buf[inds_even .+ 3, inds_even .- 3, :] +
               out_buf[inds_even .- 3, inds_even .+ 3, :] + out_buf[inds_even .- 3, inds_even .- 3, :])
    @views out_buf[[offset, end - offset + 1], inds_even, :] .= out_buf[[offset + 2, end - offset - 1], inds_even, :] .+ interp.noise_std .* randn.()
    @views out_buf[inds_even, [offset, end - offset + 1], :] .= out_buf[inds_even, [offset + 2, end - offset - 1], :] .+ interp.noise_std .* randn.()
    out_buf[inds_even, inds_even, :] .+= interp.noise_std .* randn.()

    # Step 4: Interpolate remaining sites with rotated & scaled stencil

    # (1:2:end, 2:2:end) sites
    @views @. out_buf[inds_odd, inds_even, :] =
        c_d * (out_buf[inds_odd, inds_even .+ 1, :] + out_buf[inds_odd, inds_even .- 1, :] +
                out_buf[inds_odd .+ 1, inds_even, :] + out_buf[inds_odd .- 1, inds_even, :]) +
        c_m * (out_buf[inds_odd .+ 1, inds_even .+ 2, :] + out_buf[inds_odd .+ 1, inds_even .- 2, :] +
                out_buf[inds_odd .- 1, inds_even .+ 2, :] + out_buf[inds_odd .- 1, inds_even .- 2, :] +
                out_buf[inds_odd .+ 2, inds_even .+ 1, :] + out_buf[inds_odd .+ 2, inds_even .- 1, :] +
                out_buf[inds_odd .- 2, inds_even .+ 1, :] + out_buf[inds_odd .- 2, inds_even .- 1, :]) +
        c_f * (out_buf[inds_odd .+ 3, inds_even, :] + out_buf[inds_odd .- 3, inds_even, :] +
                out_buf[inds_odd, inds_even .+ 3, :] + out_buf[inds_odd, inds_even .- 3, :])
    out_buf[inds_even, inds_odd, :] .+= (interp.noise_std / 2^(5/12)) .* randn.()

    # (2:2:end, 1:2:end) sites
    @views @. out_buf[inds_even, inds_odd, :] =
        c_d * (out_buf[inds_even .+ 1, inds_odd, :] + out_buf[inds_even .- 1, inds_odd, :] +
                out_buf[inds_even, inds_odd .+ 1, :] + out_buf[inds_even, inds_odd .- 1, :]) +
        c_m * (out_buf[inds_even .+ 1, inds_odd .+ 2, :] + out_buf[inds_even .+ 1, inds_odd .- 2, :] +
                out_buf[inds_even .- 1, inds_odd .+ 2, :] + out_buf[inds_even .- 1, inds_odd .- 2, :] +
                out_buf[inds_even .+ 2, inds_odd .+ 1, :] + out_buf[inds_even .+ 2, inds_odd .- 1, :] +
                out_buf[inds_even .- 2, inds_odd .+ 1, :] + out_buf[inds_even .- 2, inds_odd .- 1, :]) +
        c_f * (out_buf[inds_even .+ 3, inds_odd, :] + out_buf[inds_even .- 3, inds_odd, :] +
                out_buf[inds_even, inds_odd .+ 3, :] + out_buf[inds_even, inds_odd .- 3, :])
    out_buf[inds_odd, inds_even, :] .+= (interp.noise_std / 2^(5/12)) .* randn.()

    # Return unpadded view
    crop_offset = (size(out_buf)[1:2] .- interp.crop_size) .÷ 2
    return @view out_buf[
        crop_offset[1] + 1:crop_offset[1] + interp.crop_size[1],
        crop_offset[2] + 1:crop_offset[2] + interp.crop_size[2],
        :]
end
