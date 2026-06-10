# SA-EDA algorithm specialized for UTGenome.
#
# Pattern mirrors sa-eda's `run_saeda` but operates directly on
# SAEDAGenomeState objects so we never copy a UTGenome more than necessary.
# Each trajectory has its own genome instance; the SA inner loop modifies
# individual CGPElement values in place and reverts on rejection.

import Random: AbstractRNG, MersenneTwister
import Statistics: mean, median
import ..UTCGP

# Block-categorical block factory for CGP genomes.
#
# Looking at MAGE's CGPNode layout: each node's node_material holds a
# sequence of CGPElements where the first is the function index and the rest
# are slot connections / types / parameters. The natural "block" is therefore
# one CGP node's worth of elements (so the joint distribution captures the
# function ↔ connection coupling). Outputs have their own per-element blocks.
function cgp_block_layout(state::SAEDAGenomeState)
    blocks = UnitRange{Int}[]
    block_doms = Vector{Vector{Int}}()
    pos = 1
    # Iterate the unaliased structure: chromosomes' node_material vectors,
    # then output nodes' node_material vectors.
    for chromosome in state.genome.genomes
        for node in chromosome.chromosome
            elem_count = length(node.node_material.material)
            elem_count == 0 && continue
            r = pos:(pos + elem_count - 1)
            push!(blocks, r)
            push!(block_doms, Int[state.domains[p] for p in r])
            pos += elem_count
        end
    end
    for output_node in state.genome.output_nodes
        elem_count = length(output_node.node_material.material)
        elem_count == 0 && continue
        r = pos:(pos + elem_count - 1)
        push!(blocks, r)
        push!(block_doms, Int[state.domains[p] for p in r])
        pos += elem_count
    end
    @assert pos - 1 == length(state.domains) "block layout doesn't cover genome"
    return BlockCategorical(blocks, block_doms)
end

Base.@kwdef struct SAEDAConfig
    pop_size::Int = 16
    sa_steps::Int = 200
    elite_size::Int = 5
    learning_rate::Float64 = 0.1
    T0::Float64 = 1.0
    Tend::Float64 = 0.01
    iterations::Int = 15
    uniform_fraction::Float64 = 0.25
    perturb_fraction::Float64 = 0.2
    elite_carryover::Int = 2
    seed::Union{Int,Nothing} = nothing
end

struct SAEDARunResult
    best_genome::UTGenome           # genome state at the best-fit point of the run
    best_loss::Float64              # MAGE loss (lower = better)
    history_best::Vector{Float64}   # best loss seen so far, per iteration
    history_mean::Vector{Float64}   # iteration mean loss
    evaluations::Int
end

# Geometric temperature schedule: T(k) = T0 * (Tend/T0)^(k/(K-1))
function _geometric_schedule(T0::Real, Tend::Real, K::Int)
    K <= 1 && return fill(Float64(T0), K)
    return Float64.(T0 .* (Tend / T0) .^ (range(0, 1, length=K)))
end

# Single SA Metropolis step. `loss` (lower = better) is MAGE's score; we
# accept moves that decrease loss, and tolerate increases with probability
# exp(-Δloss / T).
function sa_step!(rng::AbstractRNG, state::SAEDAGenomeState, loss::Float64,
                  ctx::SAEDAFitnessContext, dist, T::Float64,
                  genes::Vector{Int})
    # `genes` is a working buffer holding the current 1-based gene vector.
    changed, olds = propose(rng, genes, dist)
    isempty(changed) && return loss
    # Apply the proposed values into the actual UTGenome elements.
    @inbounds for (i, pos) in enumerate(changed)
        v = genes[pos] + state.elements[pos].lowest_bound - 1
        set_node_element_value!(state.elements[pos], v)
    end
    new_loss = saeda_fitness(ctx, state.genome)
    Δ = new_loss - loss              # positive = worse for MAGE-loss convention
    if Δ <= 0 || rand(rng) < exp(-Δ / T)
        return new_loss
    end
    # Revert.
    @inbounds for (i, pos) in enumerate(changed)
        old_v = olds[i]
        genes[pos] = old_v
        v = old_v + state.elements[pos].lowest_bound - 1
        set_node_element_value!(state.elements[pos], v)
    end
    return loss
end

function run_trajectory!(rng::AbstractRNG, state::SAEDAGenomeState,
                         ctx::SAEDAFitnessContext, dist,
                         schedule::AbstractVector{Float64},
                         init_genes::Vector{Int})
    # Initialise: write init_genes into the genome and score.
    set_genes_1based!(state.elements, init_genes)
    genes = copy(init_genes)
    loss = saeda_fitness(ctx, state.genome)
    best_genes = copy(genes)
    best_loss = loss
    for k in eachindex(schedule)
        loss = sa_step!(rng, state, loss, ctx, dist, schedule[k], genes)
        if loss < best_loss
            best_loss = loss
            best_genes .= genes
        end
    end
    return loss, best_genes, best_loss
end

"""
    fit_SAEDA(initial_genome, ctx, cfg) -> SAEDARunResult

Top-level entry point — analogous to mage-mcts-binary's `fit_MCTS`. Initialises
a pool of `pop_size` genomes derived from `initial_genome`, runs `iterations`
EDA generations of SA-polished trajectories, and returns the best genome seen.

The distribution defaults to `BlockCategorical` (one block per CGP node);
swap for `CategoricalVector(domains)` if you want the per-position marginal
(EdaFold-style).
"""
function fit_SAEDA(initial_genome::UTGenome, ctx::SAEDAFitnessContext,
                   cfg::SAEDAConfig; use_block_cat::Bool=true,
                   dist::Union{Nothing,AbstractSAEDADistribution}=nothing)
    rng = cfg.seed === nothing ? Random.default_rng() : MersenneTwister(cfg.seed)
    schedule = _geometric_schedule(cfg.T0, cfg.Tend, cfg.sa_steps)

    # Build the pool — one SAEDAGenomeState per trajectory.
    pool = [SAEDAGenomeState(initial_genome) for _ in 1:cfg.pop_size]
    n = length(pool[1].elements)
    if dist === nothing
        dist = use_block_cat ? cgp_block_layout(pool[1]) :
                                CategoricalVector(pool[1].domains)
    end
    n_uniform = round(Int, cfg.uniform_fraction * cfg.pop_size)
    n_carry   = clamp(cfg.elite_carryover, 0, cfg.pop_size)

    pool_genes = [zeros(Int, n) for _ in 1:cfg.pop_size]
    pool_loss  = fill(Inf, cfg.pop_size)
    prev_genes = [zeros(Int, n) for _ in 1:cfg.pop_size]
    carry_genes = n_carry > 0 ? [zeros(Int, n) for _ in 1:n_carry] : nothing
    have_carry = false

    history_best = Float64[]
    history_mean = Float64[]
    best_loss = Inf
    best_genes = zeros(Int, n)
    evaluations = 0

    for it in 1:cfg.iterations
        for k in 1:cfg.pop_size
            tr_rng = MersenneTwister(hash((cfg.seed === nothing ? 0 : cfg.seed,
                                           UInt64(it), UInt64(k))) % UInt64)
            # Decide starting genes for this trajectory.
            init_genes = if have_carry && k <= n_carry
                copy(carry_genes[k])
            elseif k <= n_carry + n_uniform
                uniform_sample(tr_rng, dist)
            elseif cfg.perturb_fraction > 0 && it > 1
                g = copy(prev_genes[k])
                u = uniform_sample(tr_rng, dist)
                @inbounds for j in 1:n
                    rand(tr_rng) < cfg.perturb_fraction && (g[j] = u[j])
                end
                g
            else
                sample(tr_rng, dist)
            end
            _final, traj_best, traj_loss = run_trajectory!(tr_rng, pool[k], ctx,
                                                          dist, schedule, init_genes)
            pool_genes[k] .= traj_best
            pool_loss[k]  = traj_loss
            evaluations  += 1 + cfg.sa_steps
            prev_genes[k] .= traj_best
        end

        # Elite selection: top-μ by lowest loss.
        order = sortperm(pool_loss)
        n_e   = min(cfg.elite_size, length(order))
        elites = Matrix{Int}(undef, n, n_e)
        for (j, idx) in enumerate(order[1:n_e])
            @inbounds for i in 1:n
                elites[i, j] = pool_genes[idx][i]
            end
        end
        update!(dist, elites; lr=cfg.learning_rate)

        # Carry-over: save top-k for next iteration's reseed slots.
        if n_carry > 0
            for kk in 1:n_carry
                carry_genes[kk] .= pool_genes[order[kk]]
            end
            have_carry = true
        end

        iter_best = pool_loss[order[1]]
        if iter_best < best_loss
            best_loss = iter_best
            best_genes .= pool_genes[order[1]]
        end
        push!(history_best, best_loss)
        push!(history_mean, mean(pool_loss))
    end

    # Build the result genome (one final SAEDAGenomeState seeded with best_genes).
    final_state = SAEDAGenomeState(initial_genome)
    set_genes_1based!(final_state.elements, best_genes)
    return SAEDARunResult(final_state.genome, best_loss,
                          history_best, history_mean, evaluations)
end
