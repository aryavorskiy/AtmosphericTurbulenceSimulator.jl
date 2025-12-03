using LinearAlgebra, HDF5, Random

"""
    kolmogorov_covmat(W)

Compute the phase covariance matrix of a turbulent layer in the atmosphere, following the Kolmogorov model. The piston
term is excluded in this model. This function assumes unit Fried parameter r0=1.

# Arguments
- `W`: the aperture function. Either a 2D array, or a `(x, y)` tuple representing the size
    of the aperture (in this case the aperture function is assumed to be a square of this
    size).
"""
function kolmogorov_covmat(sz::NTuple{2}, constraint=nothing)
    I = CartesianIndices(sz)
    C = Array{Float64}(undef, length(I), length(I))

    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        C[i, j] = -0.5 * 6.88 * (x^2 + y^2)^(5/6)
    end
    if constraint === nothing
        Cp = vec(sum(C, dims=2)) / prod(sz)
        Cc = sum(Cp) / prod(sz)
    elseif constraint isa AbstractMatrix
        @assert size(constraint) == sz
        @assert sum(constraint) ≈ 1
        Cp = vec(sum(C .* vec(constraint)', dims=2))
        Cc = sum(Cp .* vec(constraint))
    else
        error("Invalid constraint of type $(typeof(constraint))")
    end
    return Symmetric(C .- Cp .- Cp' .+ Cc)
end
kolmogorov_covmat(W::AbstractMatrix) =
    kolmogorov_covmat(size(W), W)

const EigenType = Union{Tuple{<:Any,<:Any}, Eigen}
struct CovariantNoise{MT}
    shape::NTuple{2,Int}
    noise_transform::MT
end
noise_buffer(sampler::CovariantNoise, batch::Int...) =
    similar(sampler.noise_transform, size(sampler.noise_transform, 2), batch...)
plate_size(sampler::CovariantNoise) = sampler.shape
function CovariantNoise(sz::NTuple{2,Int}, (E, U)::EigenType)
    @assert length(E) == prod(sz)
    @assert size(U) == (length(E), length(E))
    E .= clamp.(E, 0, Inf)
    CovariantNoise(sz, U .* sqrt.(E'))
end
screensize(sampler::CovariantNoise) = sampler.shape
function samplephases!(phases, sampler::CovariantNoise, noise_buffer)
    @assert size(noise_buffer, 1) == size(sampler.noise_transform, 2)
    @assert size(noise_buffer, 2) == size(phases, 3)
    randn!(noise_buffer)
    phases_rs = reshape(phases, size(sampler.noise_transform, 1), size(phases, 3))
    mul!(phases_rs, sampler.noise_transform, noise_buffer)
    return phases
end
samplephases(sampler, batch::Int...) =
    samplephases!(zeros(plate_size(sampler)..., batch...), sampler, noise_buffer(sampler, batch...))

function project_sampler(sampler::CovariantNoise, basis)
    @assert size(basis, 1) == size(sampler.noise_transform, 1)
    basisq = Matrix(qr(basis).Q)
    return CovariantNoise(sampler.shape, basisq * basisq' * sampler.noise_transform)
end

struct KolmogorovUncorrelated{MT<:AbstractMatrix}
    weights::MT
    sampler::CovariantNoise{MT}
    factor::Float64
end
KolmogorovUncorrelated(weights::AbstractMatrix, r₀::Real, pixel_scale::Real) =
    KolmogorovUncorrelated(weights,
        CovariantNoise(size(weights), eigen(kolmogorov_covmat(weights))),
        (pixel_scale / r₀)^(5/6))
KolmogorovUncorrelated(sz::NTuple{2,Int}, r₀::Real, pixel_scale::Real) =
    KolmogorovUncorrelated(fill(1/prod(sz), sz...), r₀, pixel_scale)
noise_buffer(turb::KolmogorovUncorrelated, batch...) = noise_buffer(turb.sampler, batch...)
plate_size(turb::KolmogorovUncorrelated) = plate_size(turb.sampler)
function samplephases!(phases, turb::KolmogorovUncorrelated, noise_buffer)
    samplephases!(phases, turb.sampler, noise_buffer)
    @. phases *= turb.factor
end
