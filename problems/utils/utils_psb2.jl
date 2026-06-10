import Pkg
import UUIDs

function get_python_path()
   
    python_path = ENV["UTCGP_PYTHON"]
    ENV["PYTHON"] = python_path
    return python_path
end

function get_psb2_path()
    dataset_path = ENV["UTCGP_PSB2_DATASET_PATH"]
    return dataset_path
end

function get_nruns()

    n_runs = ENV["UTCGP_NRUNS"]
    n_runs = parse(Int, n_runs)
    println("nruns from try is $n_runs")
    
end

function get_unique_id()
    return UUIDs.uuid4().value
end

function load_psb2_data(dataset_path::String, pb::String, n_train::Int, n_test::Int)
    PROBLEM = pb
    N_TRAIN = n_train
    N_test = n_test

    py"""
    import psb2 
    import numpy as np

    (train_data, test_data) = psb2.fetch_examples(
        $dataset_path, $PROBLEM, $N_TRAIN, $N_TEST, format="psb2", 
    )
    """
end

function make_rows(
    in_keys::Vector{String},
    in_casters::Vector,
    out_keys::Vector{String},
    out_casters::Vector,
    extra_inputs::Vector,
    df_in_python_memory::String)
    X = []
    Y = []
    df = df_in_python_memory
    for x in py"$$df"
        ins = [caster(x[k]) for (caster, k) in zip(in_casters, in_keys)]
        outs = [caster(x[k]) for (caster, k) in zip(out_casters, out_keys)]
        push!(X, Any[ins..., extra_inputs...])
        push!(Y, identity.([outs...]))
    end
    return X, Y
end


function fix_all_output_nodes!(ut_genome::UTGenome)
    for (ith_out_node, output_node) in enumerate(ut_genome.output_nodes)
        to_node = output_node[2].highest_bound + 1 - ith_out_node
        set_node_element_value!(output_node[2],
            to_node)
        set_node_freeze_state(output_node[2])
        println("Output node at $ith_out_node: $(output_node.id) pointing to $to_node")
        println("Output Node material : $(node_to_vector(output_node))")
    end
end


function args_parse()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--seed", "-s"
        help = "Random seed"
        arg_type = Int
        required = true
    end
    parsed_args = parse_args(s)
    return parsed_args
end


function fit_MCTS(
    shared_inputs::SharedInput,
    genome::UTGenome, 
    model_architecture::modelArchitecture,
    node_config::nodeConfig,
    run_config::runConf,
    mcts_config::MCTS.mctsConfig,
    meta_library::MetaLibrary,
    population_callbacks::UTCGP.Mandatory_FN,
    mutation_callbacks::UTCGP.Mandatory_FN,
    output_mutation_callbacks::UTCGP.Mandatory_FN,
    decoding_callbacks::UTCGP.Mandatory_FN,
    elite_selection_callbacks::UTCGP.Mandatory_FN,
    epoch_callbacks::UTCGP.Optional_FN
)

    start = now()

    ### INITIALISE GEN LOSS TRACKER
    M_gen_loss_tracker = UTCGP.GenerationLossTracker()

    ### INITIALISE MCTS
    mcts = MCTSearch(mcts_config, shared_inputs, model_architecture, 
                        run_config, node_config, meta_library, 
                        population_callbacks, mutation_callbacks,
                        output_mutation_callbacks, elite_selection_callbacks,
                        decoding_callbacks)

    initial_genome_fitness = MCTS.fitness(mcts, genome)
    root = MCTSNode(genome, 1, initial_genome_fitness)
    
    best_genome = deepcopy(root)
    best_genome_fitness = MCTS.fitness(mcts, best_genome)
    best_loss = best_genome_fitness

    i_stop = 1
    k_stop = 1
    for iteration = 1:run_config.generations
        
        if now() - start > Hour(98)
            best_loss = best_genome_fitness
            println("--- 98h passed, breaking the loop before iteration $iteration and returning best fitness")
            break
        end

        @warn "Iteration : $iteration"

        reset_genome!(root.state)

        parent = root
        while length(parent.child_list) != 0
            parent = MCTS.uct(mcts, parent)
        end
        if MCTS.is_leaf(parent)
            parent.child_list = MCTS.genetic_action_binary_random(mcts, parent)
            parent = rand(parent.child_list) 
        end
        parent.visit_count += 1

        best_aged, k = MCTS.rollout(mcts, parent) 
        MCTS.backtrack(best_aged)

        current_best = deepcopy(best_genome)
        fitness_current_best = MCTS.fitness(mcts, current_best)
        new_best = deepcopy(best_aged)
        fitness_new_best = MCTS.fitness(mcts, new_best)

        if fitness_new_best <= fitness_current_best
            best_genome = new_best
            best_genome_fitness = fitness_new_best
        end

        reset_genome!(current_best.state)
        reset_genome!(new_best.state)
        reset_genome!(best_genome.state) 

        println("After iteration $iteration, BEST fitness: $best_genome_fitness")

        # EPOCH CALLBACK
        ind_performances = [best_genome_fitness]
        population = UTCGP.Population([best_genome.state])
        individual_program, population_programs = MCTS.decode_single_genome_to_individual_program(mcts, best_genome.state)
        best_loss = best_genome_fitness
        best_program = individual_program
        elite_idx = 1 # we only have one program at a time

         if !isnothing(epoch_callbacks)
            UTCGP._make_epoch_callbacks_calls(
                ind_performances, # this generation best fitness
                population,  # this generation program
                iteration, # take
                run_config, # take
                model_architecture,  # take
                node_config,  # take
                meta_library,  # take
                shared_inputs,  # take
                population_programs, # overall best fitness
                best_loss, 
                best_program, # overall best program
                elite_idx,
                epoch_callbacks,
            )
        end
        
        # store iteration loss/fitness
        UTCGP.affect_fitness_to_loss_tracker!(M_gen_loss_tracker, iteration, best_loss)
        println(
            "Iteration $iteration loss: $(round(best_loss, digits = 10))"
        )

        if best_loss == 0.0
            i_stop = iteration
            k_stop = k
            println("Found complete solution at iteration $i_stop and rollout stop $k_stop, early stopping .....")
            break
        end
    end

    println("Best loss before returning: $best_loss")
    return (best_genome, root, M_gen_loss_tracker, i_stop, k_stop)
    
end