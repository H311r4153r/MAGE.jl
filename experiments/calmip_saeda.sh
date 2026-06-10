#!/bin/bash
# CALMIP SLURM script for SA-EDA-on-MAGE.
#
# Runs `problems/saeda-${PROBLEM}.jl` via the UTCGP.SAEDA submodule. SA-EDA
# uses the same MAGE function bundles + architecture + decoding pipeline as
# the MCTS / 1+λ baselines, so the comparison is library-faithful.
#
# Submit once per problem, sweep seeds via the array index:
#
#   sbatch --export=PROBLEM=fuel-cost                experiments/calmip_saeda.sh
#   sbatch --export=PROBLEM=fizz-buzz                experiments/calmip_saeda.sh
#
# Override any knob at submission time:
#
#   sbatch --export=PROBLEM=fuel-cost,POP=32,LR=0.05 experiments/calmip_saeda.sh
#
# Or change the seed range:
#
#   sbatch --array=1-25 --export=PROBLEM=fuel-cost   experiments/calmip_saeda.sh
#
# --------------------------------------------------------------------------
# ONE-TIME SETUP ON CALMIP (do this once on a login node before the first sbatch)
#
#   module purge && module load intel/18.2 intelmpi/18.2 julia/1.10.5
#   cd $HOME/<path>/MAGE.jl
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'  # requires team's private registry
#                                                       # for SearchNetworks etc.
#   # Build PyCall against your Python that has the `psb2` package installed:
#   export UTCGP_PYTHON=/path/to/python  # the one with psb2
#   julia --project=. -e 'using Pkg; ENV["PYTHON"]=ENV["UTCGP_PYTHON"]; Pkg.build("PyCall")'
#
# PSB2 dataset path must be set so the Python loader finds the JSON files:
#   export UTCGP_PSB2_DATASET_PATH=$HOME/datasets/psb2-data
#
# --------------------------------------------------------------------------

#SBATCH --job-name=saeda-mage
#SBATCH --array=1-10                  # seeds 1..10 (override: sbatch --array=1-25)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # single Julia process per array task (no MPI in this port)
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00               # generous for 200k evals; trim if your config is smaller
#SBATCH --output=logs/saeda-%x-%A_%a.out
#SBATCH --error=logs/saeda-%x-%A_%a.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.fr

set -euo pipefail
mkdir -p logs metrics exports

# ============================================================
# Tweakable parameters — each overridable via `sbatch --export=NAME=VALUE,...`
# ============================================================
: "${PROBLEM:=fuel-cost}"      # fuel-cost | fizz-buzz | coin-sums | luhn |
                                # mastermind | spin-words | square-digits | twitter
: "${POP:=16}"                 # SA-EDA population size
: "${SA_STEPS:=200}"           # SA inner-loop length (K)
: "${ITERS:=15}"               # outer iterations
: "${LR:=0.1}"                 # distribution learning rate
: "${CARRY:=2}"                # elite carry-over
: "${UNIFORM_FRACTION:=0.25}"  # forced-uniform fraction (paper EdaFold default)
: "${PERTURB_FRACTION:=0.2}"   # 1/5 of variables perturbed per iter boundary
: "${USE_BLOCK_CAT:=true}"     # block-categorical per CGP node (joint over function + slots)
: "${ELITE_SIZE:=}"            # auto = max(POP/3, 3) when empty

# Environment for MAGE's Python data loader. Override these in submission if
# CALMIP exposes them under different paths.
: "${UTCGP_PYTHON:=/usr/bin/python3}"
: "${UTCGP_PSB2_DATASET_PATH:=$HOME/datasets/psb2-data}"

SEED="${SLURM_ARRAY_TASK_ID:-1}"
JOBID="${SLURM_ARRAY_JOB_ID:-local}"

# ============================================================
# CALMIP module + environment setup
# ============================================================
module purge
module load intel/18.2 intelmpi/18.2
module load julia/1.10.5             # use the same Julia as for the sa-eda-cgp setup

echo "--- module status ---"
module list 2>&1
echo "--- julia on PATH? ---"
which julia 2>&1 || { echo "julia NOT on PATH"; exit 2; }
julia --version 2>&1

cd "$SLURM_SUBMIT_DIR"

# ============================================================
# Per-job config record (so we can map JOBID -> hyperparameters later)
# ============================================================
if [ "${SLURM_ARRAY_TASK_ID:-1}" = "1" ]; then
    mkdir -p metrics/_configs
    {
        echo "jobid=$JOBID"
        echo "problem=$PROBLEM"
        echo "pop=$POP  K=$SA_STEPS  iters=$ITERS  lr=$LR"
        echo "carry=$CARRY  uniform_fraction=$UNIFORM_FRACTION  perturb_fraction=$PERTURB_FRACTION"
        echo "use_block_cat=$USE_BLOCK_CAT  elite_size=${ELITE_SIZE:-auto}"
        echo "python=$UTCGP_PYTHON  psb2=$UTCGP_PSB2_DATASET_PATH"
    } > "metrics/_configs/${PROBLEM}_saeda_${JOBID}.config"
fi

# ============================================================
# Export everything the Julia script reads via ENV
# ============================================================
export PROBLEM SEED \
       POP SA_STEPS ITERS LR CARRY UNIFORM_FRACTION PERTURB_FRACTION \
       USE_BLOCK_CAT ELITE_SIZE \
       UTCGP_PYTHON UTCGP_PSB2_DATASET_PATH

echo "================================================================"
echo "JOB ${JOBID} / seed ${SEED}"
echo "  problem=$PROBLEM  pop=$POP  K=$SA_STEPS  iters=$ITERS  lr=$LR"
echo "  carry=$CARRY  uniform_fraction=$UNIFORM_FRACTION  perturb_fraction=$PERTURB_FRACTION"
echo "  use_block_cat=$USE_BLOCK_CAT  elite_size=${ELITE_SIZE:-auto}"
echo "================================================================"

# Run. `--seed N` is parsed by args_parse() inside problems/utils/utils_psb2.jl.
julia --project=. \
      "problems/saeda-${PROBLEM}.jl" \
      --seed "$SEED"
