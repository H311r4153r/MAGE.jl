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
# MPI is loaded lazily — only required when USE_MPI=true.

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
    # ELITE_SIZE may be set to empty string in the SLURM script to mean "auto" —
    # only call parse() when it's a non-empty digit string.
    elite_size_raw = get(ENV, "ELITE_SIZE", "")
    elite_size  = isempty(strip(elite_size_raw)) ? max(pop_size ÷ 3, 3) : parse(Int, elite_size_raw)

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

    # Dispatch: USE_MPI=true → distributed loop, else serial.
    use_mpi = parse(Bool, get(ENV, "USE_MPI", "false"))

    # Initialise MPI if needed. We accept being called inside an already-
    # initialised MPI environment too (some launchers do this for you).
    if use_mpi
        @eval Main using MPI
        if !Main.MPI.Initialized()
            Main.MPI.Init()
        end
    end
    is_master = !use_mpi || Main.MPI.Comm_rank(Main.MPI.COMM_WORLD) == 0

    start = now()
    result = if use_mpi
        is_master && @warn "SA-EDA: distributed mode (MPI), $(Main.MPI.Comm_size(Main.MPI.COMM_WORLD)) ranks"
        SAEDA.fit_SAEDA_mpi(genome, ctx, cfg; use_block_cat=use_blockc)
    else
        @warn "SA-EDA: serial mode"
        SAEDA.fit_SAEDA(genome, ctx, cfg; use_block_cat=use_blockc)
    end
    wall = (now() - start) / Second(1)

    # In MPI mode, only the master rank gets a non-nothing result. Workers
    # synchronize here, then return early so they don't try to write trackers
    # or CSV rows.
    if !is_master
        if use_mpi
            Main.MPI.Barrier(Main.MPI.COMM_WORLD)
            if !Main.MPI.Finalized()
                Main.MPI.Finalize()
            end
        end
        return (nothing, NaN, Float64[], Float64[], 0, wall)
    end

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

    # Held-out test loss: evaluate the best genome against X_test/Y_test using
    # the same decoder pipeline. This is the column we need for
    # paper-comparable results.
    test_loss = try
        test_ctx = SAEDA.SAEDAFitnessContext(
            shared_inputs=shared_inputs,
            model_architecture=model_architecture,
            node_config=node_config,
            run_config=run_config,
            meta_library=meta_library,
            decoding_callbacks=decoding_callbacks,
            X_train=X_test, Y_train=Y_test,    # reuse the same fitness machinery
            endpoint=endpoint,
        )
        SAEDA.saeda_fitness(test_ctx, result.best_genome)
    catch e
        @warn "test-loss evaluation failed: $e"
        NaN
    end

    # CSV emission — one append per run. Schema matches the previous
    # sa-eda-cgp full_scale_run.jl conventions so downstream aggregation
    # scripts can stay the same.
    csv_dir = joinpath(pwd(), "metrics")
    isdir(csv_dir) || mkpath(csv_dir)
    problem_name = get(ENV, "PROBLEM", "unknown")
    csv_path = joinpath(csv_dir, "$(problem_name)_saeda.csv")
    write_header = !isfile(csv_path)
    try
        open(csv_path, "a") do io
            if write_header
                println(io, join([
                    "problem", "method", "seed",
                    "n_train", "n_test",
                    "train_loss", "test_loss",
                    "evaluations", "wall_s",
                    "pop", "sa_steps", "iters", "lr",
                    "carry", "uniform_fraction", "perturb_fraction",
                    "use_block_cat", "elite_size",
                ], ","))
            end
            println(io, join([
                problem_name, "saeda", seed_,
                length(X_train), length(X_test),
                result.best_loss, test_loss,
                result.evaluations, round(wall; digits=3),
                pop_size, sa_steps, iterations, lr,
                carry, uniform_f, perturb_f,
                use_blockc, elite_size,
            ], ","))
        end
        @warn "CSV row appended to $csv_path"
    catch e
        @warn "CSV write failed (continuing): $e"
    end

    # Tear down MPI on the master rank too (workers already finalized above).
    if use_mpi
        Main.MPI.Barrier(Main.MPI.COMM_WORLD)
        if !Main.MPI.Finalized()
            Main.MPI.Finalize()
        end
    end

    return (result.best_genome, result.best_loss, result.history_best,
            result.history_mean, result.evaluations, wall)
end
