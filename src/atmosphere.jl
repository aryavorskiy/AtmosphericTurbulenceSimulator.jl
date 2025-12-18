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
function samplephases!(sampler::KarhunenLoeveBuffers)
    randn!(sampler.noise_buffer)
    mul!(sampler.out_buffer, sampler.noise_transform, sampler.noise_buffer)
    return reshape(sampler.out_buffer, (sampler.shape..., size(sampler.out_buffer, 2)))
end

"""
    IndependentFrames(size, r0)

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `size`: a tuple `(nx, ny)` specifying the phase screen shape in pixels.
- `r0`: Fried parameter (r₀) in pixels.

# Notes
This sampler is intentionally simple: frames are independent between timesteps and are
generated using a Karhunen–Loève transform built from the Kolmogorov covariance.
"""
struct IndependentFrames{T} <: AtmosphereSpec{T}
    size::NTuple{2, Int}
    r₀::T
end
IndependentFrames(::Type{T}, sz::NTuple{2,Int}, r0) where T = IndependentFrames{T}(sz, r0)

function prepare_phasebuffers(spec::IndependentFrames{T}, batch::Int, deviceadapter) where T
    covar = Adapt.adapt_storage(deviceadapter, kolmogorov_covmat(T, spec.size))
    covar .*= spec.r₀^(-5/3)
    E, U = eigen(covar)
    return KarhunenLoeveBuffers(spec.size, (E, U), batch)
end
