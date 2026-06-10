# SA-EDA submodule inside UTCGP.
#
# Adds a third search strategy alongside the package's existing 1+λ EA and
# (in mage-mcts-binary) MCTS. The implementation lives entirely under
# `src/saeda/` so the main UTCGP module is untouched aside from a single
# `include` line.
#
# Public entry point:  fit_SAEDA(initial_genome, fitness_ctx, cfg)

module SAEDA

import Random
using Random: AbstractRNG

include("distribution.jl")
include("genome_adapter.jl")
include("fitness.jl")
include("algorithm.jl")

export AbstractSAEDADistribution, CategoricalVector, BlockCategorical
export sample, uniform_sample, propose, update!, dim, snapshot, restore!
export SAEDAFitnessContext, saeda_fitness
export SAEDAGenomeState, flatten_evolvable_elements, domain_sizes_from_elements,
       get_genes_1based, get_genes_1based!, set_genes_1based!
export SAEDAConfig, SAEDARunResult, fit_SAEDA, cgp_block_layout

end # module SAEDA
