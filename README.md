# Hands-On-LLMS
**Hands-On-LLMS* is a curated, hands-on learning repository that documents my personal study path through the rapidly evolving landscape of large language model (LLM) tooling.

---

## 🎯 Purpose  
To document my hands-on journey through modern LLM ecosystems, capturing insights, and breakthroughs as I work with each tool.

---
## 🔧 Tools Covered  
- **LangChain**: Building and chaining prompts, agents, and RAG pipelines for retrieval-augmented workflows.  
- **Langraph**: Constructing and querying graph-based knowledge structures driven by an LLM.  
- **Unsloth**: Benchmarking and optimizing fine-tuning and inference performance across backends.  
- **LiveKit**: Integrating real-time audio/video streams for interactive, multimodal LLM demos.  
- **Ollama**: Running and managing open-source LLMs locally, experimenting with versioning and on-prem scaling.

---
| Parameter           | Purpose                                         | Affects Input or Output | Description                                                                 |
|---------------------|-------------------------------------------------|-------------------------|-----------------------------------------------------------------------------|
| `temperature`       | Controls randomness in text generation          | Output                  | Higher values (e.g., 1.0) make output more random; lower values (e.g., 0.2) make it more deterministic. Typical range: 0.0 to 2.0. |
| `top_p`             | Controls diversity via nucleus sampling         | Output                  | Selects tokens whose cumulative probability is at least `top_p` (e.g., 0.9 = top 90%), balancing diversity and relevance in output. |
| `top_k`             | Limits sampling to top K probable tokens        | Output                  | Randomly samples from the top K most probable tokens for the next token, ensuring focused output. |
| `seed`              | Enables reproducible text generation            | Output                  | Sets a random seed for reproducible outputs when using the same parameters and input. Ensures consistency across runs. |
| `repetition_penalty`| Reduces repeated tokens in output               | Output                  | Values >1.0 discourage repeated tokens; values <1.0 encourage repetition (rarely used). Typical values >1.0 reduce redundancy in output. |
| `num_predict`       | Limits length of generated response             | Output                  | Caps the generated response to this number of tokens, potentially truncating longer outputs. |
| `num_ctx`           | Sets context window for input and output        | Input + Output          | Sets the maximum number of tokens (input + output) in the context window, determining how much conversation history and response the model can process. |