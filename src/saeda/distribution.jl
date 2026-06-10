# SA-EDA distributions for categorical genomes (vendored from sa-eda package).
# The CGP genome is a flat Vector{Int} where each position has its own domain
# (function index range, slot index range, output index range). We model the
# distribution as one categorical per position (independent) or one joint
# distribution per CGP-node block (function + slot indices).

abstract type AbstractSAEDADistribution end

# ---------- Independent categorical per position ----------
mutable struct CategoricalVector <: AbstractSAEDADistribution
    domains::Vector{Int}
    probs::Vector{Vector{Float64}}
    pmin::Float64
end

function CategoricalVector(domains::AbstractVector{<:Integer}; pmin::Real=1e-3)
    all(d -> d >= 1, domains) || throw(ArgumentError("all domain sizes must be >= 1"))
    probs = [fill(1.0 / d, d) for d in domains]
    return CategoricalVector(collect(Int, domains), probs, Float64(pmin))
end

dim(d::CategoricalVector) = length(d.domains)

# Helper: sample a 1-based state from a discrete probability vector.
function _sample_state(rng::AbstractRNG, probs::Vector{Float64})
    r = rand(rng)
    acc = 0.0
    @inbounds for (i, p) in enumerate(probs)
        acc += p
        if r < acc
            return i
        end
    end
    return length(probs)
end

function sample(rng::AbstractRNG, d::CategoricalVector)
    x = Vector{Int}(undef, length(d.domains))
    @inbounds for i in eachindex(d.domains)
        x[i] = _sample_state(rng, d.probs[i])
    end
    return x
end

function uniform_sample(rng::AbstractRNG, d::CategoricalVector)
    x = Vector{Int}(undef, length(d.domains))
    @inbounds for i in eachindex(d.domains)
        x[i] = rand(rng, 1:d.domains[i])
    end
    return x
end

# Update from elite genomes. weights is an optional per-elite weight vector.
function update!(d::CategoricalVector, elites::AbstractMatrix{Int};
                 lr::Real, weights=nothing)
    n_elite = size(elites, 2)
    if weights === nothing
        w = ones(Float64, n_elite); total_w = Float64(n_elite)
    else
        length(weights) == n_elite || throw(DimensionMismatch("weights"))
        w = Float64.(weights); total_w = sum(w)
        total_w > 0 || return d
    end
    @inbounds for i in eachindex(d.domains)
        K = d.domains[i]
        freq = zeros(K)
        for j in 1:n_elite
            v = elites[i, j]
            (1 <= v <= K) || throw(ArgumentError("elite value $v out of domain at $i"))
            freq[v] += w[j]
        end
        freq ./= total_w
        for k in 1:K
            d.probs[i][k] = max(d.pmin, (1 - lr) * d.probs[i][k] + lr * freq[k])
        end
        d.probs[i] ./= sum(d.probs[i])
    end
    return d
end

snapshot(d::CategoricalVector) = deepcopy(d)
restore!(d::CategoricalVector, snap::CategoricalVector) = (d.probs = [copy(p) for p in snap.probs]; d)

# ---------- Block categorical (joint per CGP node) ----------
mutable struct BlockCategorical <: AbstractSAEDADistribution
    blocks::Vector{UnitRange{Int}}
    block_domains::Vector{Vector{Int}}
    probs::Vector{Array{Float64}}
    pmin::Float64
end

function BlockCategorical(blocks::Vector{<:AbstractRange{Int}},
                          block_domains::Vector{<:AbstractVector{<:Integer}};
                          pmin::Real=1e-3)
    bblocks = UnitRange{Int}[r for r in blocks]
    bdoms = Vector{Int}[Int[d for d in dom] for dom in block_domains]
    probs = Array{Float64}[]
    for (i, dom) in enumerate(bdoms)
        length(dom) == length(bblocks[i]) || throw(ArgumentError("block $i: domain length mismatch"))
        sz = tuple(dom...)
        push!(probs, fill(1.0 / prod(dom), sz))
    end
    return BlockCategorical(bblocks, bdoms, probs, Float64(pmin))
end

dim(d::BlockCategorical) = sum(length(b) for b in d.blocks; init=0)

@inline function _multi_to_flat(idx::AbstractVector{<:Integer}, dims::Vector{Int})
    flat = 1; stride = 1
    @inbounds for k in eachindex(idx)
        flat += (idx[k] - 1) * stride
        stride *= dims[k]
    end
    return flat
end

@inline function _flat_to_multi!(out::AbstractVector{Int}, flat::Integer, dims::Vector{Int})
    rem = flat - 1
    @inbounds for k in eachindex(dims)
        out[k] = (rem % dims[k]) + 1
        rem = div(rem, dims[k])
    end
    return out
end

function sample(rng::AbstractRNG, d::BlockCategorical)
    n = dim(d); x = Vector{Int}(undef, n)
    tmp = Vector{Int}(undef, 0)
    @inbounds for b in eachindex(d.blocks)
        flat = _sample_state(rng, vec(d.probs[b]))
        resize!(tmp, length(d.block_domains[b]))
        _flat_to_multi!(tmp, flat, d.block_domains[b])
        for (k, pos) in enumerate(d.blocks[b])
            x[pos] = tmp[k]
        end
    end
    return x
end

function uniform_sample(rng::AbstractRNG, d::BlockCategorical)
    n = dim(d); x = Vector{Int}(undef, n)
    @inbounds for b in eachindex(d.blocks)
        for (k, pos) in enumerate(d.blocks[b])
            x[pos] = rand(rng, 1:d.block_domains[b][k])
        end
    end
    return x
end

function update!(d::BlockCategorical, elites::AbstractMatrix{Int};
                 lr::Real, weights=nothing)
    n_elite = size(elites, 2)
    if weights === nothing
        w = ones(Float64, n_elite); total_w = Float64(n_elite)
    else
        w = Float64.(weights); total_w = sum(w)
        total_w > 0 || return d
    end
    tmp = Vector{Int}(undef, 0)
    @inbounds for b in eachindex(d.blocks)
        dom = d.block_domains[b]
        resize!(tmp, length(d.blocks[b]))
        freq_flat = zeros(Float64, prod(dom))
        for j in 1:n_elite
            for (k, pos) in enumerate(d.blocks[b])
                tmp[k] = elites[pos, j]
            end
            freq_flat[_multi_to_flat(tmp, dom)] += w[j]
        end
        freq_flat ./= total_w
        probs_flat = vec(d.probs[b])
        for k in eachindex(probs_flat)
            probs_flat[k] = max(d.pmin, (1 - lr) * probs_flat[k] + lr * freq_flat[k])
        end
        probs_flat ./= sum(probs_flat)
    end
    return d
end

snapshot(d::BlockCategorical) = deepcopy(d)
function restore!(d::BlockCategorical, snap::BlockCategorical)
    d.probs = [copy(p) for p in snap.probs]
    return d
end

# Propose a single-position resample. For BlockCategorical, this resamples
# the entire block containing the picked position. Returns:
#   (changed_positions::Vector{Int}, old_values::Vector{Int})
function propose(rng::AbstractRNG, x::AbstractVector{Int}, d::CategoricalVector)
    i = rand(rng, 1:length(d.domains))
    new_val = _sample_state(rng, d.probs[i])
    old_val = x[i]
    new_val == old_val && return Int[], Int[]
    x[i] = new_val
    return [i], [old_val]
end

function propose(rng::AbstractRNG, x::AbstractVector{Int}, d::BlockCategorical)
    b = rand(rng, 1:length(d.blocks))
    blk = d.blocks[b]; dom = d.block_domains[b]
    new_flat = _sample_state(rng, vec(d.probs[b]))
    new_idx = Vector{Int}(undef, length(dom))
    _flat_to_multi!(new_idx, new_flat, dom)
    changed = Int[]; olds = Int[]
    @inbounds for (k, pos) in enumerate(blk)
        if x[pos] != new_idx[k]
            push!(changed, pos); push!(olds, x[pos])
            x[pos] = new_idx[k]
        end
    end
    return changed, olds
end
