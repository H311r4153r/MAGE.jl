# SA-EDA fit wrapper — analogous to `fit_MCTS` in utils_psb2.jl. Sets up the
# fitness context, runs the SA-EDA loop, and writes per-iteration tracking
# rows via the same jsonTracker / jsonTestTracker mechanism the MCTS path uses.
#
# Reads SA-EDA hyperparameters from environment variables to match the CALMIP
# SLURM-script convention used for the other comparison runs (`POP`, `LR`, ...).

using UTCGP
import UTCGP: SAEDA
using Dates
using Random
using Logging

"""
    fit_SAEDA_problem(shared_inputs, genome, model_arch, node_config,
                      run_config, meta_library, decoding_callbacks,
                      X_train, Y_train, X_test, Y_test, endpoint,
                      metric_tracker, test_tracker)

End-to-end run of SA-EDA on one PSB2 problem. Returns `(best_genome, best_loss,
history_best, history_mean, evaluations, wall_seconds)`.
"""
function fit_SAEDA_problem(
        shared_inputs::SharedInput,
        genome::UTGenome,
        model_architecture::modelArchitecture,
        node_config::nodeConfig,
        run_config::runConf,
        meta_library::MetaLibrary,
        decoding_callbacks::UTCGP.Mandatory_FN,
        X_train, Y_train, X_test, Y_test,
        endpoint,
        metric_tracker, test_tracker
    )

    # Read knobs from env (SLURM --export passes them through).
    pop_size    = parse(Int,    get(ENV, "POP",              "16"))
    sa_steps    = parse(Int,    get(ENV, "SA_STEPS",         "200"))
    iterations  = parse(Int,    get(ENV, "ITERS",            "15"))
    lr          = parse(Float64,get(ENV, "LR",               "0.1"))
    carry       = parse(Int,    get(ENV, "CARRY",            "2"))
    uniform_f   = parse(Float64,get(ENV, "UNIFORM_FRACTION", "0.25"))
    perturb_f   = parse(Float64,get(ENV, "PERTURB_FRACTION", "0.2"))
    use_blockc  = parse(Bool,   get(ENV, "USE_BLOCK_CAT",    "true"))
    seed_       = parse(Int,    get(ENV, "SEED",             "1"))
    elite_size  = parse(Int,    get(ENV, "ELITE_SIZE",       string(max(pop_size ÷ 3, 3))))

    @warn "SA-EDA config: pop=$pop_size K=$sa_steps iters=$iterations lr=$lr carry=$carry uniform_f=$uniform_f perturb_f=$perturb_f use_block_cat=$use_blockc seed=$seed_"

    ctx = SAEDA.SAEDAFitnessContext(
        shared_inputs=shared_inputs,
        model_architecture=model_architecture,
        node_config=node_config,
        run_config=run_config,
        meta_library=meta_library,
        decoding_callbacks=decoding_callbacks,
        X_train=X_train, Y_train=Y_train,
        endpoint=endpoint,
    )

    cfg = SAEDA.SAEDAConfig(
        pop_size=pop_size, sa_steps=sa_steps, elite_size=elite_size,
        learning_rate=lr, T0=1.0, Tend=0.01,
        iterations=iterations, uniform_fraction=uniform_f,
        perturb_fraction=perturb_f, elite_carryover=carry, seed=seed_,
    )

    start = now()
    result = SAEDA.fit_SAEDA(genome, ctx, cfg; use_block_cat=use_blockc)
    wall = (now() - start) / Second(1)

    @warn "SA-EDA finished: best_loss=$(result.best_loss)  evals=$(result.evaluations)  wall=$(wall) s"

    # Tracking: write history to the jsonTracker if available; the structure
    # accepts the same epoch-style logging the MCTS path uses. We avoid the
    # full epoch_callbacks mechanism since SA-EDA doesn't carry the same
    # population semantics — instead we just dump per-iteration best loss.
    if metric_tracker !== nothing
        try
            for (it, loss) in enumerate(result.history_best)
                UTCGP.affect_fitness_to_loss_tracker!(metric_tracker, it, loss)
            end
        catch e
            @warn "metric tracker write failed (continuing): $e"
        end
    end

    return (result.best_genome, result.best_loss, result.history_best,
            result.history_mean, result.evaluations, wall)
end
