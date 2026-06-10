# Fitness evaluation: take a UTGenome, run it through MAGE's standard decoding
# + program execution pipeline against (X_train, Y_train), return the
# aggregated training loss.
#
# Pattern mirrors mage-mcts-binary/src/search.jl's `fitness(mcts, genome)`.

import ..UTCGP
import ..UTCGP: SharedInput, modelArchitecture, nodeConfig, runConf, MetaLibrary,
                Population, IndividualLossTracker, InputNode

"""
    SAEDAFitnessContext

Bundles every piece of state MAGE's `_make_decoding` /
`evaluate_individual_programs` /  endpoint pipeline needs to score a genome.
Built once at problem setup time and shared (read-only) across all
trajectories.
"""
struct SAEDAFitnessContext
    shared_inputs::SharedInput
    model_architecture::modelArchitecture
    node_config::nodeConfig
    run_config::runConf
    meta_library::MetaLibrary
    decoding_callbacks::UTCGP.Mandatory_FN
    X_train::AbstractArray
    Y_train::AbstractArray
    endpoint::Union{Type{<:UTCGP.BatchEndpoint},<:UTCGP.BatchEndpoint}
    fail_penalty::Float64
end

SAEDAFitnessContext(; shared_inputs, model_architecture, node_config,
                      run_config, meta_library, decoding_callbacks,
                      X_train, Y_train, endpoint, fail_penalty=10000.0) =
    SAEDAFitnessContext(shared_inputs, model_architecture, node_config,
                        run_config, meta_library, decoding_callbacks,
                        X_train, Y_train, endpoint, fail_penalty)

# Helper: build input nodes from a row of X. Identical to the MCTS wrapper's
# `make_input_nodes` so behaviour matches the paper's setup.
function _make_input_nodes(x::Vector, input_types::Vector)
    return [InputNode(value, pos, pos, input_types[pos]) for (pos, value) in enumerate(x)]
end

"""
    saeda_fitness(ctx, genome) -> Float64

Decode the genome to an individual program and score it on the training set,
matching the protocol used by mage-mcts-binary's `MCTS.fitness`. Returns
MAGE's loss (lower = better). SA-EDA negates this internally so its
"maximize fitness" convention works correctly.
"""
function saeda_fitness(ctx::SAEDAFitnessContext, genome::UTGenome)
    population = Population([genome])
    iteration = 0
    population_programs, _ = UTCGP._make_decoding(
        population, iteration,
        ctx.run_config, ctx.model_architecture,
        ctx.node_config, ctx.meta_library,
        ctx.shared_inputs, ctx.decoding_callbacks,
    )
    individual_program = population_programs[1]

    loss_tracker = IndividualLossTracker()
    for (x, y) in zip(ctx.X_train, ctx.Y_train)
        UTCGP.reset_programs!(individual_program)
        input_nodes = _make_input_nodes(x, ctx.model_architecture.inputs_types_idx)
        UTCGP.replace_shared_inputs!(individual_program, input_nodes)

        outputs = UTCGP.evaluate_individual_programs(
            individual_program,
            ctx.model_architecture.chromosomes_types,
            ctx.meta_library,
        )

        fitness_values = try
            output_fitness = ctx.endpoint([outputs], y)
            UTCGP.get_endpoint_results(output_fitness)
        catch
            [ctx.fail_penalty]
        end
        UTCGP.add_pop_loss_to_ind_tracker!(loss_tracker, fitness_values)
    end
    UTCGP.reset_programs!(individual_program)
    return UTCGP.resolve_ind_loss_tracker(loss_tracker)[1]
end
