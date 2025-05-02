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
| `temperature`       | Controls randomness in response generation      | Output                  | Higher values (e.g., 1.0) make output more random; lower values (e.g., 0.2) make it more deterministic. |
| `top_p`             | Nucleus sampling parameter                      | Output                  | Limits token selection to a subset with cumulative probability ≤ `top_p`.   |
| `top_k`             | Limits token selection to top K probabilities   | Output                  | Considers only the top K tokens with highest probability for next token selection. |
| `seed`              | Sets random seed for reproducibility            | Input                   | Ensures consistent outputs across runs when using the same parameters.      |
| `repetition_penalty`| Penalizes repeated tokens                       | Output                  | Values >1.0 discourage repetition; values <1.0 encourage it.                |
| `num_predict`       | Maximum number of tokens to generate            | Output                  | Caps the length of the generated response to this number of tokens.         |
| `num_ctx`           | Maximum number of tokens from context to consider | Input                 | Determines how much of the conversation history the model uses for generating a response. |

