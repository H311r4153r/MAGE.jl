make_cma_nodes!(args...) = @error "Should load MAGE_PYCMA to be olverloaded"
get_cma_nodes(args...) = @error "Should load MAGE_PYCMA to be olverloaded"
mutate_cma!(args...) = @error "Should load MAGE_PYCMA to be olverloaded"

"""
    llm_generated_function_client_backend(args...)

Return a readable backend name such as `\"ollama\"` for one generated-function
client.

Example: `llm_generated_function_client_backend(client)` may return `\"ollama\"`.
"""
llm_generated_function_client_backend(args...) = @error "No generated-function LLM backend is available for this client."

"""
    llm_generated_function_client_status(args...)

Extension hook returning one small status summary for a concrete generated-function
client. This keeps runtime-specific readiness checks out of UTCGP's core.

Example: an Ollama-backed client may report its model name and local host.
"""
llm_generated_function_client_status(args...) = @error "No generated-function LLM backend is available for this client."

export make_cma_nodes!, get_cma_nodes, mutate_cma!
export llm_generated_function_client_backend, llm_generated_function_client_status
