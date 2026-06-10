# SA-EDA on PSB2 fizz-buzz — auto-generated from mage-mcts-binary/problems/mcts-fizz-buzz.jl
# Differences from MCTS baseline: search loop is fit_SAEDA_problem (UTCGP.SAEDA
# submodule). All MAGE-level config (libraries, architecture, node config,
# data loaders, endpoint) is identical to the MCTS baseline so the
# comparison is library-faithful.

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
n_runs = 990
Pkg.build("PyCall")

hash = get_unique_id()

# ---- PROBLEM-SPECIFIC BLOCK (lifted from mcts-fizz-buzz.jl) ----
# PARAMS --- --- 
N_TRAIN = 200
N_TEST = 2000
PROBLEM = "fizz-buzz"
EXPERIMENT_NAME = "FIZZ BUZZ"
STAGE = 1
MODE = "MCTS"
endpoint = EndpointBatchLevensthein

# LOAD THE DATA --- --- 
load_psb2_data(dataset_path, PROBLEM, N_TRAIN, N_TEST) # loads to python memory

extra_inputs = ["Fizz", "Buzz", "FizzBuzz", "", 0, 3, 5]
# TRAIN DATA --- ---- 
X, Y = make_rows(
    ["input1"],
    [integer_caster],
    ["output1"],
    [string_caster],
    extra_inputs,
    "train_data"
)

# TEST DATA --- ---- 
X_test, Y_test = make_rows(
    ["input1"],
    [integer_caster],
    ["output1"],
    [string_caster],
    extra_inputs,
    "test_data"
)

X_test_behavior = deepcopy(X_test)
@assert unique(length.(X)) == unique(length.(X_test))
@assert length(X) == N_TRAIN
@assert length(X_test) == N_TEST
offset_by = length(X_test[1])

# ---- ARCHITECTURE (lifted from mcts-fizz-buzz.jl) ----
# Bundles Integer
integer_bundles = get_integer_bundles()
string_bundles = get_string_bundles()

# Libraries
lib_integer = Library(integer_bundles)
lib_string = Library(string_bundles)

# MetaLibrary
ml = MetaLibrary([lib_integer, lib_string])

### Model Architecture ###
model_arch = modelArchitecture(
    [Int, String, String, String, String, Int, Int, Int],
    [1, 2, 2, 2, 2, 1, 1, 1],
    [Int, String],
    [String],
    [2]
)

### Node Config ###
N_nodes = 30
println("N Nodes : $N_nodes")
node_config = nodeConfig(N_nodes, 1, 3, offset_by)

# ---- GENOME ----
shared_inputs, ut_genome = make_evolvable_utgenome(model_arch, ml, node_config)
initialize_genome!(ut_genome)
correct_all_nodes!(ut_genome, model_arch, ml, shared_inputs)
fix_all_output_nodes!(ut_genome)

# ---- RUN CONF (only MAGE-level mutation rate / output mut rate matter here) ----
lmbd = 1
run_conf = runConf(lmbd, n_runs, 1.1, 0.2)

# ---- TRACKING ----
MODE = "SAEDA"
h_params = Dict(
    "n_nodes" => node_config.n_nodes,
    "lambda" => run_conf.lambda_,
    "budget" => run_conf.generations,
    "mutation_rate" => run_conf.mutation_rate,
    "output_mutation_rate" => run_conf.output_mutation_rate,
    "mode" => MODE,
    "experiment" => EXPERIMENT_NAME,
    "stage" => STAGE,
    "n_train" => N_TRAIN,
    "n_test" => N_TEST,
    "problem" => PROBLEM,
    "seed" => seed_,
    "search" => "saeda",
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

mkpath(pwd_dir * "/exports/$PROBLEM")
open(pwd_dir * "/exports/$PROBLEM/$(seed_)_final_best_genome.bin", "w") do io
    serialize(io, best_genome)
end
open(pwd_dir * "/exports/$PROBLEM/$(seed_)_stop_info.txt", "w") do io
    println(io, "best_loss = $best_loss")
    println(io, "evaluations = $evaluations")
    println(io, "wall_seconds = $wall_s")
end
