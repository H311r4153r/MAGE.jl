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
# ONE-TIME SETUP (already done if you followed the README; here for reference)
#
#   module purge && module load intel/18.2 intelmpi/18.2 julia/1.10.5 python/3.11
#   python3 -m venv $HOME/envs/psb2-venv
#   source $HOME/envs/psb2-venv/bin/activate
#   pip install --only-binary=:all: "numpy<2"
#   pip install psb2
#   deactivate
#
#   cd $HOME/projects/MAGE-saeda
#   # Patch TestImages's broken build script (one-time):
#   TI_DIR=$(ls -d ~/.julia/packages/TestImages/*/ | head -1)
#   echo "# build skipped" > "$TI_DIR/deps/build.jl"
#
#   export UTCGP_PYTHON=$HOME/envs/psb2-venv/bin/python
#   julia --project=. -e '
#     using Pkg
#     Pkg.add(name="TestImages", version="1.9")
#     Pkg.add(url="https://github.com/camilodlt/SearchNetworks.jl")
#     Pkg.add("PyCall")
#     Pkg.instantiate()
#     ENV["PYTHON"] = ENV["UTCGP_PYTHON"]
#     Pkg.build("PyCall")
#   '
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
mkdir -p logs metrics exports metrics/_configs

# ============================================================
# Tweakable SA-EDA parameters — each overridable via `sbatch --export=...`
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

# Environment for MAGE's Python data loader. Defaults match the one-time-setup
# above; if you put psb2 somewhere else, override via --export or via your
# ~/.bashrc on CALMIP.
: "${UTCGP_PYTHON:=$HOME/envs/psb2-venv/bin/python}"
: "${UTCGP_PSB2_DATASET_PATH:=$HOME/datasets/psb2-data}"

SEED="${SLURM_ARRAY_TASK_ID:-1}"
JOBID="${SLURM_ARRAY_JOB_ID:-local}"

# ============================================================
# CALMIP module + environment setup
# Compute nodes don't inherit login-shell modules, so reload everything here.
# ============================================================
module purge
module load intel/18.2 intelmpi/18.2
module load julia/1.10.5
# (No python module load — the venv's python binary is self-contained.
#  Add a `module avail python` step and adjust here if your venv ever
#  starts complaining about a missing system Python.)

# Belt-and-braces: pin the julia binary absolute path. CALMIP's older srun
# was stripping $PATH inside MPI-launched tasks; we don't use srun here but
# this is harmless and lets you copy-paste the same script into an MPI variant
# later if the algorithm gains a distributed mode.
JULIA_BIN=$(which julia)

echo "--- module status ---"
module list 2>&1
echo "--- julia ---"
echo "  binary: $JULIA_BIN"
"$JULIA_BIN" --version
echo "--- python (UTCGP_PYTHON) ---"
echo "  binary: $UTCGP_PYTHON"
"$UTCGP_PYTHON" -c "import sys, psb2; print('python:', sys.version.split()[0], '  psb2 ok')" \
    || { echo "ERROR: $UTCGP_PYTHON cannot import psb2"; exit 3; }
echo "--- end diagnostics ---"

cd "$SLURM_SUBMIT_DIR"

# ============================================================
# Per-job config record — maps JOBID back to hyperparameters
# ============================================================
if [ "${SLURM_ARRAY_TASK_ID:-1}" = "1" ]; then
    {
        echo "jobid=$JOBID"
        echo "problem=$PROBLEM"
        echo "pop=$POP  K=$SA_STEPS  iters=$ITERS  lr=$LR"
        echo "carry=$CARRY  uniform_fraction=$UNIFORM_FRACTION  perturb_fraction=$PERTURB_FRACTION"
        echo "use_block_cat=$USE_BLOCK_CAT  elite_size=${ELITE_SIZE:-auto}"
        echo "python=$UTCGP_PYTHON  psb2=$UTCGP_PSB2_DATASET_PATH"
        echo "submitted=$(date -Is)"
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
"$JULIA_BIN" --project=. \
             "problems/saeda-${PROBLEM}.jl" \
             --seed "$SEED"
