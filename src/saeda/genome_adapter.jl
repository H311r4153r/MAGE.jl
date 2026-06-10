# Adapter layer: lets us treat a UTGenome as a flat Vector{Int} for SA-EDA
# purposes (sampling, mutation, distribution update), while delegating fitness
# evaluation back to MAGE's existing decoding + program execution pipeline.
#
# The adapter walks the genome once at construction to collect every evolvable
# CGPElement in document order, recording the bounds. After that, reads and
# writes go through the cached element vector — no further traversal of the
# UTGenome structure is needed during the SA inner loop.

import ..UTCGP: UTGenome, CGPElement, SingleGenome
import ..UTCGP: set_node_element_value!, get_node_element_value

"""
    flatten_evolvable_elements(genome) -> Vector{CGPElement}

Walk the UTGenome and collect every evolvable CGPElement (function, slot,
parameter, output) in a fixed document order. The same order is used to
project gene values back into the genome during proposals.
"""
function flatten_evolvable_elements(genome::UTGenome)
    elements = CGPElement[]
    # Chromosomes — each is a SingleGenome with a vector of CGP nodes; each
    # node has a node_material with one or more CGPElements.
    for chromosome in genome.genomes
        for node in chromosome.chromosome
            for elem in node.node_material.material
                push!(elements, elem)
            end
        end
    end
    # Output nodes
    for output_node in genome.output_nodes
        for elem in output_node.node_material.material
            push!(elements, elem)
        end
    end
    return elements
end

"""
    domain_sizes_from_elements(elements) -> Vector{Int}

Number of legal values for each gene position: `highest_bound - lowest_bound + 1`.
"""
function domain_sizes_from_elements(elements::Vector{CGPElement})
    return [e.highest_bound - e.lowest_bound + 1 for e in elements]
end

"""
    get_genes_1based(elements) -> Vector{Int}

Read the current gene values as a 1-based vector. The conversion makes the
representation match SA-EDA's CategoricalVector convention (values in
`1:domain_size`).
"""
function get_genes_1based(elements::Vector{CGPElement})
    return [e.value - e.lowest_bound + 1 for e in elements]
end

function get_genes_1based!(out::Vector{Int}, elements::Vector{CGPElement})
    @inbounds for i in eachindex(elements)
        out[i] = elements[i].value - elements[i].lowest_bound + 1
    end
    return out
end

"""
    set_genes_1based!(elements, x)

Write the 1-based gene vector back into the genome. Uses
`set_node_element_value!` so the frozen-element state is respected.
"""
function set_genes_1based!(elements::Vector{CGPElement}, x::AbstractVector{Int})
    @inbounds for (e, v) in zip(elements, x)
        set_node_element_value!(e, v + e.lowest_bound - 1)
    end
end

"""
    SAEDAGenomeState

Per-trajectory state: one UTGenome instance plus the cached element list and
its domain sizes. Used inside the SA inner loop — every trajectory gets its
own state so mutation in one trajectory doesn't race with another.
"""
mutable struct SAEDAGenomeState
    genome::UTGenome
    elements::Vector{CGPElement}
    domains::Vector{Int}
end

function SAEDAGenomeState(genome::UTGenome)
    g = deepcopy(genome)
    elements = flatten_evolvable_elements(g)
    domains = domain_sizes_from_elements(elements)
    return SAEDAGenomeState(g, elements, domains)
end
