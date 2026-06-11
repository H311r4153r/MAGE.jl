# Distributed SA-EDA via MPI. Master/worker pattern mirroring sa-eda-cgp's
# run_saeda_mpi: master owns the distribution + global elite tracking; each
# rank runs its slice of trajectories locally, with its own SAEDAGenomeState
# (so no UTGenome ever crosses the wire — only Vector{Int} gene vectors).
#
# Per iteration:
#   1. Master broadcasts the current distribution.
#   2. Master broadcasts the carry-over gene vectors (top-k elites), if any.
#   3. Each rank computes its trajectories and sends back
#      Vector{Tuple{Vector{Int}, Float64}}.
#   4. Master aggregates, selects elites, updates the distribution, saves
#      carry-over for the next iteration, and tracks history.
#
# MPI.Init() / MPI.Finalize() must be called by the caller — fit_SAEDA_mpi
# itself only operates within an active MPI environment.

import MPI

# Trajectory range owned by `rank` when splitting `pop_size` across `nprocs`.
function _local_range(pop_size::Int, rank::Int, nprocs::Int)
    base = pop_size ÷ nprocs
    rem  = pop_size % nprocs
    extra = rank < rem ? 1 : 0
    local_pop = base + extra
    start = rank * base + min(rank, rem) + 1
    return start, start + local_pop - 1
end

"""
    fit_SAEDA_mpi(initial_genome, ctx, cfg; comm=MPI.COMM_WORLD,
                  use_block_cat=true) -> SAEDARunResult | nothing

Distributed version of `fit_SAEDA`. All ranks must call this. Returns a
`SAEDARunResult` on the master rank (0), `nothing` on workers.

Communication pattern per iteration: one `MPI.bcast` for the distribution,
one `MPI.bcast` for carry-over (Vector{Vector{Int}} or nothing), and one
`MPI.gather` of trajectory results. The distribution is the only non-trivial
payload — gene vectors are small (a few hundred ints per genome).
"""
function fit_SAEDA_mpi(initial_genome::UTGenome, ctx::SAEDAFitnessContext,
                       cfg::SAEDAConfig; comm=MPI.COMM_WORLD,
                       use_block_cat::Bool=true,
                       dist::Union{Nothing,AbstractSAEDADistribution}=nothing)
    rank      = MPI.Comm_rank(comm)
    nprocs    = MPI.Comm_size(comm)
    is_master = rank == 0

    schedule  = _geometric_schedule(cfg.T0, cfg.Tend, cfg.sa_steps)
    n_uniform = round(Int, cfg.uniform_fraction * cfg.pop_size)
    n_carry   = clamp(cfg.elite_carryover, 0, cfg.pop_size)

    # Shared base seed: master picks (or uses cfg.seed), workers receive.
    base_seed = is_master ? (cfg.seed === nothing ? rand(UInt64) : UInt64(cfg.seed)) : zero(UInt64)
    base_seed = MPI.bcast(base_seed, comm; root=0)

    lo, hi = _local_range(cfg.pop_size, rank, nprocs)
    local_pop = hi - lo + 1

    # Each rank has its own genome template + cached element list.
    state = SAEDAGenomeState(initial_genome)
    n     = length(state.elements)

    # Build / accept the distribution. Master initializes; workers receive
    # via bcast each iteration.
    if is_master && dist === nothing
        dist = use_block_cat ? cgp_block_layout(state) : CategoricalVector(state.domains)
    elseif !is_master
        dist = nothing
    end

    # Per-rank persistent-trajectory storage (for perturb_fraction mode).
    local_prev_genes = [zeros(Int, n) for _ in 1:local_pop]

    # Master-only state.
    pool_genes   = is_master ? [zeros(Int, n) for _ in 1:cfg.pop_size] : nothing
    pool_loss    = is_master ? fill(Inf, cfg.pop_size)                 : nothing
    carry_genes  = (is_master && n_carry > 0) ?
                       [zeros(Int, n) for _ in 1:n_carry] : nothing
    have_carry   = false

    history_best = Float64[]
    history_mean = Float64[]
    best_loss    = Inf
    best_genes   = zeros(Int, n)
    evaluations  = 0

    for it in 1:cfg.iterations
        # Broadcast distribution. Master sends its dist; workers overwrite
        # their reference with the master's copy.
        iter_dist = MPI.bcast(dist, comm; root=0)

        # Broadcast carry-over vectors. Send `nothing` when no carry-over
        # is available yet (it == 1 or n_carry == 0).
        bcast_carry  = is_master ? (have_carry ? carry_genes : nothing) : nothing
        bcast_carry  = MPI.bcast(bcast_carry, comm; root=0)
        carry_active = bcast_carry !== nothing

        # Each rank computes its local trajectories.
        local_results = Vector{Tuple{Vector{Int},Float64}}(undef, local_pop)
        for (i, k) in enumerate(lo:hi)
            tr_rng = MersenneTwister(hash((base_seed, UInt64(it), UInt64(k))) % UInt64)
            init_genes = if carry_active && k <= n_carry
                copy(bcast_carry[k])
            elseif k <= n_carry + n_uniform
                uniform_sample(tr_rng, iter_dist)
            elseif cfg.perturb_fraction > 0 && it > 1
                g = copy(local_prev_genes[i])
                u = uniform_sample(tr_rng, iter_dist)
                @inbounds for j in 1:n
                    rand(tr_rng) < cfg.perturb_fraction && (g[j] = u[j])
                end
                g
            else
                sample(tr_rng, iter_dist)
            end

            _, traj_best, traj_loss = run_trajectory!(tr_rng, state, ctx,
                                                     iter_dist, schedule, init_genes)
            local_results[i] = (copy(traj_best), traj_loss)
            local_prev_genes[i] .= traj_best
        end

        # Gather to master. Each rank sends Vector{Tuple{...}}; master gets
        # Vector{Vector{Tuple{...}}} in rank order.
        gathered = MPI.gather(local_results, comm; root=0)
        evaluations += cfg.pop_size * (1 + cfg.sa_steps)

        if is_master
            # Flatten gathered results into the pool.
            offset = 1
            for rank_results in gathered
                for (genes, loss) in rank_results
                    pool_genes[offset] .= genes
                    pool_loss[offset]   = loss
                    offset += 1
                end
            end

            # Elite selection + distribution update.
            order = sortperm(pool_loss)
            n_e   = min(cfg.elite_size, length(order))
            elites = Matrix{Int}(undef, n, n_e)
            for (j, idx) in enumerate(order[1:n_e])
                @inbounds for i in 1:n
                    elites[i, j] = pool_genes[idx][i]
                end
            end
            update!(dist, elites; lr=cfg.learning_rate)

            # Save top-k for next iter's carry-over.
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
    end

    if is_master
        final_state = SAEDAGenomeState(initial_genome)
        set_genes_1based!(final_state.elements, best_genes)
        return SAEDARunResult(final_state.genome, best_loss,
                              history_best, history_mean, evaluations)
    else
        return nothing
    end
end
