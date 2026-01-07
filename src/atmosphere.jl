using LinearAlgebra, HDF5, Random, Adapt

abstract type AtmosphereSpec{T} end

"""
    kolmogorov_covmat(W)
    kolmogorov_covmat([T, ]size)

Compute the phase covariance matrix of a turbulent layer in the atmosphere, following the Kolmogorov model. The piston
term is excluded in this model. This function assumes unit Fried parameter ``r_0 = 1 px``.

# Arguments
- `W`: the aperture function as a 2D array of weights. Normalized to `sum(W) == 1` internally.
- `size`: a tuple `(nx, ny)` specifying the size of the aperture function.
- `T`: element type for the covariance matrix (default `Float64`). If the aperture function `W` is provided, its element type is used.
"""
function kolmogorov_covmat(W::AbstractMatrix)
    I = eachindex(IndexCartesian(), W)
    C = similar(W, length(I), length(I))
    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        C[i, j] = -0.5 * 6.88 * (x^2 + y^2)^(5/6)
    end
    Wp = W ./ sum(W)
    Cp = vec(sum(C .* vec(Wp)', dims=2))
    Cc = sum(Cp .* vec(Wp))
    return Symmetric(C .- Cp .- Cp' .+ Cc)
end
kolmogorov_covmat(::Type{T}, sz::NTuple{2,Int}) where T = kolmogorov_covmat(ones(T, sz))
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
batch_length(sampler::KarhunenLoeveBuffers) = size(sampler.noise_buffer, 2)
out_buffer(sampler::KarhunenLoeveBuffers) = reshape(sampler.out_buffer, (sampler.shape..., size(sampler.out_buffer, 2)))
function samplephases!(sampler::KarhunenLoeveBuffers)
    randn!(sampler.noise_buffer)
    mul!(sampler.out_buffer, sampler.noise_transform, sampler.noise_buffer)
    return out_buffer(sampler)
end

struct HardingSpec{N}
    interpolate_to::NTuple{2,Int}
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
        return HardingSpec{n}(final_size, interpolate_from)
    elseif interpolate isa Number
        interpolate_from = cld.(final_size .- 11, 2^interpolate) .+ 11
        return HardingSpec{interpolate}(final_size, interpolate_from)
    elseif interpolate === :auto
        n = 0
        interpolate_from = final_size
        while prod(interpolate_from) .> size_heuristics
            n += 1
            interpolate_from = cld.(final_size .- 11, 2^n) .+ 11
        end
        return HardingSpec{n}(final_size, interpolate_from)
    else
        throw(ArgumentError("`interpolate` must be a Number or :auto"))
    end
end

"""
    SingleLayer(size, r0[; interpolate, interpolate_from, size_heuristics=1024])

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `size`: a tuple `(nx, ny)` specifying the phase screen shape in pixels (coarse sampler grid).
- `r0`: Fried parameter (r₀) in pixels.

# Keyword Arguments
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
struct SingleLayer{T,N} <: AtmosphereSpec{T}
    harding::HardingSpec{N}
    r₀::T
end
SingleLayer(sz::NTuple{2,Int}, r0::T; kw...) where T =
    SingleLayer(HardingSpec(sz; kw...), r0)
SingleLayer(::Type{T}, sz::NTuple{2,Int}, r0; kw...) where T =
    SingleLayer(HardingSpec(sz; kw...), convert(T, float(r0)))
plate_size(spec::SingleLayer) = spec.harding.interpolate_to
function prepare_phasebuffers(spec::SingleLayer{T,N}, batch::Int, deviceadapter) where {T,N}
    low_size = spec.harding.interpolate_from
    covar = Adapt.adapt_storage(deviceadapter, kolmogorov_covmat(T, low_size))
    low_r₀ = spec.r₀ / 2^N
    covar .*= low_r₀^(-5/3)
    E, U = eigen(Symmetric(covar))
    kl = KarhunenLoeveBuffers(low_size, (E, U), batch)
    N == 0 && return kl
    return HardingInterpolator(kl, low_r₀, spec.harding)
end

struct HardingInterpolator{N,BT,AT}
    base::BT
    out_buffers::NTuple{N,AT}
    noise_std::Float64
    crop_size::NTuple{2,Int}
end
function HardingInterpolator(base, r0::Number, hspec::HardingSpec{N}) where N
    low_size = hspec.interpolate_from
    any(low_size .≤ 11) && throw(ArgumentError("Dimensions must be greater than 11"))
    buffers = ntuple(Val(N)) do i
        lsz = low_size
        for _ in 1:i
            lsz = 2 .* lsz .- 11
        end
        similar(out_buffer(base), (lsz .+ 10)..., batch_length(base))
    end
    return HardingInterpolator(base, buffers, sqrt(0.5265 / r0^(5/3)), hspec.interpolate_to)
end
plate_size(sampler::HardingInterpolator) = sampler.crop_size
batch_length(sampler::HardingInterpolator) = size(sampler.out_buffers[end], 3)
out_buffer(sampler::HardingInterpolator) = sampler.out_buffers[end]

function samplephases!(harding::HardingInterpolator{N}) where N
    low = samplephases!(harding.base)
    harding_upsample!(harding.out_buffers[1], low, harding.noise_std)
    for i in 2:N
        prev_buf = harding.out_buffers[i - 1]
        harding_upsample!(harding.out_buffers[i], @view(prev_buf[6:end-5, 6:end-5, :]),
            harding.noise_std / 2^(5/6 * (i-1)))
    end
    out_buf = harding.out_buffers[N]
    crop_offset = (size(out_buf)[1:2] .- harding.crop_size) .÷ 2
    return @view out_buf[
        crop_offset[1] + 1:crop_offset[1] + harding.crop_size[1],
        crop_offset[2] + 1:crop_offset[2] + harding.crop_size[2],
        :]
end
function harding_upsample!(out_buf, low, noise_std)
    c_d = 0.3198
    c_m = -0.0341
    c_f = -0.0017

    # Padding offset
    n, m = size(low)
    inds_odd_x = range(5, length=n-4, step=2)
    inds_odd_y = range(5, length=m-4, step=2)
    inds_even_x = range(4, length=n-3, step=2)
    inds_even_y = range(4, length=m-3, step=2)

    # Copy low-res
    @views copy!(out_buf[1:2:end, 1:2:end, :], low)

    # Interpolate checker pattern sites
    randn!(@view out_buf[inds_even_x, inds_even_y, :])
    @views @. out_buf[inds_even_x, inds_even_y, :] =
        noise_std * out_buf[inds_even_x, inds_even_y, :] +
        c_d * (out_buf[inds_even_x .+ 1, inds_even_y .+ 1, :] + out_buf[inds_even_x .+ 1, inds_even_y .- 1, :] +
               out_buf[inds_even_x .- 1, inds_even_y .+ 1, :] + out_buf[inds_even_x .- 1, inds_even_y .- 1, :]) +
        c_m * ((out_buf[inds_even_x .+ 3, inds_even_y .+ 1, :] + out_buf[inds_even_x .+ 3, inds_even_y .- 1, :] +
               out_buf[inds_even_x .- 3, inds_even_y .+ 1, :] + out_buf[inds_even_x .- 3, inds_even_y .- 1, :]) +
               (out_buf[inds_even_x .+ 1, inds_even_y .+ 3, :] + out_buf[inds_even_x .+ 1, inds_even_y .- 3, :] +
               out_buf[inds_even_x .- 1, inds_even_y .+ 3, :] + out_buf[inds_even_x .- 1, inds_even_y .- 3, :])) +
        c_f * (out_buf[inds_even_x .+ 3, inds_even_y .+ 3, :] + out_buf[inds_even_x .+ 3, inds_even_y .- 3, :] +
               out_buf[inds_even_x .- 3, inds_even_y .+ 3, :] + out_buf[inds_even_x .- 3, inds_even_y .- 3, :])

    # Fill remaining sites
    randn!(@view out_buf[inds_odd_x, inds_even_y, :])
    @views @. out_buf[inds_odd_x, inds_even_y, :] =
        $(noise_std * 2^(-5/12)) * out_buf[inds_odd_x, inds_even_y, :] +
        c_d * (out_buf[inds_odd_x, inds_even_y .+ 1, :] + out_buf[inds_odd_x, inds_even_y .- 1, :] +
                out_buf[inds_odd_x .+ 1, inds_even_y, :] + out_buf[inds_odd_x .- 1, inds_even_y, :]) +
        c_m * (out_buf[inds_odd_x .+ 1, inds_even_y .+ 2, :] + out_buf[inds_odd_x .+ 1, inds_even_y .- 2, :] +
                out_buf[inds_odd_x .- 1, inds_even_y .+ 2, :] + out_buf[inds_odd_x .- 1, inds_even_y .- 2, :] +
                out_buf[inds_odd_x .+ 2, inds_even_y .+ 1, :] + out_buf[inds_odd_x .+ 2, inds_even_y .- 1, :] +
                out_buf[inds_odd_x .- 2, inds_even_y .+ 1, :] + out_buf[inds_odd_x .- 2, inds_even_y .- 1, :]) +
        c_f * (out_buf[inds_odd_x .+ 3, inds_even_y, :] + out_buf[inds_odd_x .- 3, inds_even_y, :] +
                out_buf[inds_odd_x, inds_even_y .+ 3, :] + out_buf[inds_odd_x, inds_even_y .- 3, :])

    randn!(@view out_buf[inds_even_x, inds_odd_y, :])
    @views @. out_buf[inds_even_x, inds_odd_y, :] =
        $(noise_std * 2^(-5/12)) * out_buf[inds_even_x, inds_odd_y, :] +
        c_d * (out_buf[inds_even_x .+ 1, inds_odd_y, :] + out_buf[inds_even_x .- 1, inds_odd_y, :] +
                out_buf[inds_even_x, inds_odd_y .+ 1, :] + out_buf[inds_even_x, inds_odd_y .- 1, :]) +
        c_m * (out_buf[inds_even_x .+ 1, inds_odd_y .+ 2, :] + out_buf[inds_even_x .+ 1, inds_odd_y .- 2, :] +
                out_buf[inds_even_x .- 1, inds_odd_y .+ 2, :] + out_buf[inds_even_x .- 1, inds_odd_y .- 2, :] +
                out_buf[inds_even_x .+ 2, inds_odd_y .+ 1, :] + out_buf[inds_even_x .+ 2, inds_odd_y .- 1, :] +
                out_buf[inds_even_x .- 2, inds_odd_y .+ 1, :] + out_buf[inds_even_x .- 2, inds_odd_y .- 1, :]) +
        c_f * (out_buf[inds_even_x .+ 3, inds_odd_y, :] + out_buf[inds_even_x .- 3, inds_odd_y, :] +
                out_buf[inds_even_x, inds_odd_y .+ 3, :] + out_buf[inds_even_x, inds_odd_y .- 3, :])
end

"""
    simulate_phases(atm_spec::AtmosphereSpec; n, [batch, filename, verbose, deviceadapter])

Simulate `n` phase screens using the provided atmosphere specification and write
the results to an HDF5 file.

# Arguments
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.

# Keyword Arguments
- `n`: number of phase screens to simulate.
- `batch`: batch size for buffered computations and HDF5 writes (default 512).
- `filename`: output HDF5 filename (default "simulation.h5").
- `verbose`: show progress meter (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_phases(atm_spec::AtmosphereSpec{FT}; n::Int, batch::Int=DEFAULT_BATCH, filename="simulation.h5",
        verbose=true, deviceadapter=Array) where {FT}
    batch = min(batch, n)
    phasebuffers = prepare_phasebuffers(atm_spec, batch, deviceadapter)

    h5open(filename, "w") do fid
        phs_size = plate_size(atm_spec)
        phs_dataset = create_dataset(fid, "phases", FT, (phs_size..., n), chunk=(phs_size..., batch))
        p = Progress(n, "Simulating phases", enabled=verbose, dt=1)
        phase_buf_h5 = zeros(FT, phs_size..., batch)
        for j in 1:cld(n, batch)
            phases = samplephases!(phasebuffers)
            if phases isa Array
                HDF5.write_chunk(phs_dataset, j - 1, phases)
            else
                copy!(phase_buf_h5, phases)
                HDF5.write_chunk(phs_dataset, j - 1, phase_buf_h5)
            end
            next!(p, step=min(batch, n - (j - 1) * batch))
        end
        finish!(p)
    end
end
