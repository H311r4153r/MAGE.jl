# SA-EDA on PSB2 Fuel Cost — mirrors the structure of mcts-fuel-cost.jl from
# mage-mcts-binary, but swaps the fit_MCTS call for fit_SAEDA_problem and uses
# the SA-EDA submodule (UTCGP.SAEDA) for the search strategy.
#
# All other elements — function bundles, model architecture, node config,
# initialise_genome!/correct_all_nodes!/fix_all_output_nodes!, dataset
# loading, endpoint, hyperparameters at the MAGE level — match the MCTS
# baseline so the comparison is library-faithful.

using Revise
using UTCGP
using UTCGP: SN_writer, sn_strictphenotype_hasher
import SearchNetworks as sn
import DataStructures: OrderedDict
using UUIDs
using Serialization
using Statistics
using Random

dir = @__DIR__
pwd_dir = pwd()

include(pwd_dir * "/problems/utils/imports.jl")
include(pwd_dir * "/problems/utils/utils_psb2.jl")
include(pwd_dir * "/problems/utils/saeda_fit.jl")

parsed_args = args_parse()
seed_ = parsed_args["seed"]
Random.seed!(seed_)
println("The random seed is : $(seed_)")

disable_logging(Logging.Error)
python_path = get_python_path()
dataset_path = get_psb2_path()
n_runs = 990                              # max iterations cap (SA-EDA's ITERS env var trims this further)
Pkg.build("PyCall")

hash = get_unique_id()

# ---- PARAMS ----
N_TRAIN = 200
N_TEST = 2000
PROBLEM = "fuel-cost"
EXPERIMENT_NAME = "FUEL COST"
STAGE = 1
MODE = "SAEDA"
N_types = 2
endpoint = EndpointBatchAbsDifference

# ---- DATA ----
load_psb2_data(dataset_path, PROBLEM, N_TRAIN, N_TEST)

extra_inputs = [0, 1, 2, 3]
X, Y = make_rows(
    ["input1"],
    [listinteger_caster],
    ["output1"],
    [integer_caster],
    extra_inputs,
    "train_data",
)
X_test, Y_test = make_rows(
    ["input1"],
    [listinteger_caster],
    ["output1"],
    [integer_caster],
    extra_inputs,
    "test_data",
)
X_test_behavior = deepcopy(X_test)
@assert unique(length.(X)) == unique(length.(X_test))
@assert length(X) == N_TRAIN
@assert length(X_test) == N_TEST
offset_by = length(X_test[1])

# ---- RUN CONF (only `mutation_rate` / `output_mutation_rate` matter at MAGE
# level here — SA-EDA reads its own hyperparams from env vars in saeda_fit.jl)
lmbd = 1
run_conf = runConf(lmbd, n_runs, 1.1, 0.2)

# ---- LIBRARIES + ARCHITECTURE ----
integer_bundles = get_integer_bundles()
listinteger_bundles = get_listinteger_bundles()
lib_integer = Library(integer_bundles)
lib_listinteger = Library(listinteger_bundles)
ml = MetaLibrary([lib_integer, lib_listinteger])

model_arch = modelArchitecture(
    [Vector{Int}, Int, Int, Int, Int],
    [2, 1, 1, 1, 1],
    [Int, Vector{Int}],
    [Int],
    [1],
)

N_nodes = 30
println("N Nodes : $N_nodes")
node_config = nodeConfig(N_nodes, 1, 3, offset_by)

# ---- GENOME ----
shared_inputs, ut_genome = make_evolvable_utgenome(model_arch, ml, node_config)
initialize_genome!(ut_genome)
correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
fix_all_output_nodes!(ut_genome)

# ---- TRACKING ----
h_params = Dict(
    "n_nodes" => node_config.n_nodes,
    "lambda" => run_conf.lambda_,
    "budget" => run_conf.generations,
    "mutation_rate" => run_conf.mutation_rate,
    "output_mutation_rate" => run_conf.output_mutation_rate,
    "mode" => MODE,
    "Correction" => "true",
    "output_node" => "fixed",
    "experiment" => EXPERIMENT_NAME,
    "stage" => STAGE,
    "n_train" => N_TRAIN,
    "n_test" => N_TEST,
    "problem" => PROBLEM,
    "seed" => seed_,
    "search" => "saeda",
    # SA-EDA-specific from env
    "saeda_pop" => get(ENV, "POP", "16"),
    "saeda_sa_steps" => get(ENV, "SA_STEPS", "200"),
    "saeda_iters" => get(ENV, "ITERS", "15"),
    "saeda_lr" => get(ENV, "LR", "0.1"),
    "saeda_carry" => get(ENV, "CARRY", "2"),
    "saeda_uniform_fraction" => get(ENV, "UNIFORM_FRACTION", "0.25"),
    "saeda_perturb_fraction" => get(ENV, "PERTURB_FRACTION", "0.2"),
    "saeda_use_block_cat" => get(ENV, "USE_BLOCK_CAT", "true"),
)
mkpath(pwd_dir * "/metrics/$PROBLEM")
f = open(pwd_dir * "/metrics/$PROBLEM/" * string(hash) * "_$(seed_).json", "a", lock=true)
metric_tracker = jsonTracker(h_params, f)
test_tracker = jsonTestTracker(metric_tracker, endpoint, X_test, Y_test)

# ---- FIT ----
best_genome, best_loss, history_best, history_mean, evaluations, wall_s = fit_SAEDA_problem(
    shared_inputs, ut_genome, model_arch, node_config, run_conf, ml,
    (:default_decoding_callback,),
    X, Y, X_test, Y_test, endpoint,
    metric_tracker, test_tracker,
)

println("Final best loss: $best_loss")
println("Total evaluations: $evaluations")
println("Wall clock: $wall_s s")

save_json_tracker(metric_tracker)
close(metric_tracker.file)

# ---- EXPORT BEST GENOME ----
mkpath(pwd_dir * "/exports/$PROBLEM")
open(pwd_dir * "/exports/$PROBLEM/$(seed_)_final_best_genome.bin", "w") do io
    serialize(io, best_genome)
end
open(pwd_dir * "/exports/$PROBLEM/$(seed_)_stop_info.txt", "w") do io
    println(io, "best_loss = $best_loss")
    println(io, "evaluations = $evaluations")
    println(io, "wall_seconds = $wall_s")
end
