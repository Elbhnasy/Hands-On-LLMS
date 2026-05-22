# Hands-On-LLMS

Hands-On-LLMS is a hands-on learning repository that documents practical experiments with large language model tooling, agent workflows, and supporting data pipelines. Most content lives in notebooks, with a few small Python scripts that demonstrate configuration patterns and client setup.

## Repository map

| Path | Focus |
| --- | --- |
| [01-LangChain](01-LangChain) | Introductory LangChain notebooks and patterns. |
| [Articles](Articles) | Background reading and comparisons. |
| [langgraph](langgraph) | LangGraph examples, including debugging flows. |
| [MCP](MCP) | Model Context Protocol client/server examples and walkthrough. |
| [notebooks](notebooks) | General experiments and prompt templates. |
| [pyspark](pyspark) | Data processing examples with Spark. |
| [src](src) | Helper utilities, configuration, and script prototypes. |
| [transformers](transformers) | Fine-tuning and training workflow notebooks. |

## Getting started

### Prerequisites
- Python 3.14 or newer
- uv installed

### Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r src/requirements.txt
```

### Environment configuration

Settings are loaded from a local `.env` file (see [src/helpers/config.py](src/helpers/config.py)). At minimum, set your API key and model identifiers as needed:

```bash
OPENAI_API_KEY=your_key_here
LLM_MODEL=your_model_name
MODEL_ID2=optional_secondary_model
OLLAMA_API_GENERATION=http://localhost:11434/api/generate
OLLAMA_API_CHAT=http://localhost:11434/api/chat
```

The Ollama endpoints above match the defaults used by the settings class and can be omitted if unchanged.

## Usage

### Notebooks
Open the notebooks in VS Code or Jupyter and run them cell-by-cell. Most experiments are organized by folder in the repository map above.

### MCP example
The [MCP](MCP) folder contains a minimal client/server setup showing how to expose tools and connect them to a LangChain agent. See [MCP/README.md](MCP/README.md) for the walkthrough and transport details.

### Scripts
The root [main.py](main.py) is a placeholder entry point. The [src/main.py](src/main.py) script contains a partial OpenAI client setup and is a starting point for expanding scripted workflows.

## LLM parameter reference

| Parameter | Purpose | Affects Input or Output | Description |
| --- | --- | --- | --- |
| `temperature` | Controls randomness in generation | Output | Higher values increase variability; lower values make output more deterministic. Typical range: 0.0 to 2.0. |
| `top_p` | Nucleus sampling threshold | Output | Samples from the smallest token set whose cumulative probability reaches `top_p`. |
| `top_k` | Limits sampling to top K tokens | Output | Restricts the candidate tokens to the top K probabilities before sampling. |
| `seed` | Enables reproducible generation | Output | Sets a random seed for deterministic outputs when other parameters match. |
| `repetition_penalty` | Reduces repeated tokens | Output | Values above 1.0 discourage repetition; values below 1.0 encourage it. |
| `num_predict` | Caps response length | Output | Limits the number of generated tokens. |
| `num_ctx` | Sets context window size | Input + Output | Maximum tokens available for the combined prompt and response. |