# Comparison Table of Major LLMs (Architectural Dimensions)

| Model / Family | Architecture | Parameters (Total / Active) | Number of Layers | Token Embedding Dim ($d_{\text{model}}$) | Max Input Context | Max Output Context | Key Architectural Specs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **OpenAI GPT-4** *(original)* | Sparse MoE | ~1.8T / ~220B *(est.)* | ~120 *(est.)* | ~8,192–8,448 *(est.)* | 8k / 32k tokens | 4,096 tokens | 16 experts (~110B–120B each; 2 routed per token) *(community estimate)*. |
| **OpenAI GPT-4 Turbo** | Sparse MoE | Proprietary *(est. ~1.8T)* | Proprietary | Proprietary *(est. ~8k)* | 128,000 tokens | 4,096 tokens | Native vision support, compressed KV-cache routing. |
| **OpenAI GPT-4o** | Native Multimodal MoE | Proprietary | Proprietary | Proprietary *(est. ~8k)* | 128,000 tokens | 16,384 tokens | Single end-to-end network across text, audio, and vision. Vocab size: 200,000 (tiktoken `o200k_base`). |
| **OpenAI o1 / o3** | Reasoning Transformer | Proprietary | Proprietary | Proprietary | 200,000 tokens | Up to 100,000 tokens *(incl. thinking)* | Native chain-of-thought (reasoning tokens deducted from output budget). |
| **Claude 3 Opus** | Transformer *(Dense/MoE)* | Proprietary | Proprietary | Proprietary | 200,000 tokens | 4,096 tokens | Flagship frontier model of the initial Claude 3 line. |
| **Claude 3.5 Sonnet** | Multimodal Transformer | Proprietary | Proprietary | Proprietary | 200,000 tokens | 8,192 tokens | Highly optimized for agentic coding and complex tool workflows. |
| **Claude 3.7 Sonnet** | Hybrid Reasoning | Proprietary | Proprietary | Proprietary | 200,000 tokens | 64k–128k tokens *(extended thinking)* | Dynamic controllable thinking budget; hybrid standard/extended execution. |
| **Google Gemini 1.5 Pro** | Multimodal MoE | Proprietary | Proprietary | Proprietary | 2,000,000 tokens | 8,192 tokens | Extreme-context needle-in-a-haystack retrieval with audio, video, code, and text support. |
| **Google Gemini 2.0 Flash / Pro** | Native Multimodal MoE | Proprietary | Proprietary | Proprietary | 1,000,000–2,000,000 tokens | 8,192–65,536 tokens | Real-time multimodal streaming, fast reasoning, deep agentic tools. |
| **Alibaba Qwen 2.5 (72B)** | Dense Transformer | **72.7B** *(all active)* | **80** | **8,192** | 128,000 tokens | 8,192 tokens | 64 Q heads, 8 KV heads (GQA), intermediate size 29,568, vocab size 152,064. |
| **Alibaba Qwen 2.5 (7B)** | Dense Transformer | **7.61B** *(all active)* | **28** | **3,584** | 128,000 tokens | 8,192 tokens | 28 Q heads, 4 KV heads (GQA), intermediate size 18,944, vocab size 152,064. |
| **Alibaba Qwen-Max** | Sparse MoE | Proprietary | Proprietary | Proprietary | 128,000 tokens | 8,192 tokens | Alibaba's proprietary frontier enterprise foundation model. |
| **Moonshot AI Kimi (Moonshot-v1)** | Dense / MoE | Proprietary | Proprietary | Proprietary | 200k–2,000,000 tokens | 4,096–8,192 tokens | Pioneer in million-token needle retrieval in consumer chat. |
| **Moonshot AI Kimi K2 / K2.5** | Sparse MoE | ~1.5T–2.8T / ~32B–48B *(est.)* | Proprietary | **7,168** | 200k–1,048,576 tokens | Up to 65,536 tokens | Multi-Head Latent Attention (MLA), Kimi Delta Attention, vocab size 163,840. |
| **Meta Llama 3.1 (405B)** | Dense Transformer | **405B** *(all active)* | **126** | **16,384** | 131,072 tokens | 4,096 / 8,192 tokens | 128 Q heads, 16 KV heads (GQA), intermediate size 53,248, vocab size 128,256. |
| **Meta Llama 3.1 / 3.3 (70B)** | Dense Transformer | **70.6B** *(all active)* | **80** | **8,192** | 131,072 tokens | 4,096 / 8,192 tokens | 64 Q heads, 8 KV heads (GQA), intermediate size 28,672, vocab size 128,256. |
| **DeepSeek-V3 / DeepSeek-R1** | Sparse MoE + MLA | **671B / 37B** | **61** *(+1 MTP layer)* | **7,168** | 128,000 tokens | 8,192 / 64,000 tokens *(R1 reasoning)* | 256 routed experts + 1 shared expert (8 routed active), Multi-Head Latent Attention (MLA), Multi-Token Prediction (MTP). |

## Key Structural Patterns

- **Dense frontier scaling limits:** Models above 70B parameters generally switch from dense architectures to sparse MoE to keep inference FLOPs per token manageable. Meta's **Llama 3.1 405B** represents the largest pure dense transformer deployed, scaling depth to **126 layers** and width to a **16,384 hidden dimension**.
- **MoE layer versus expert distribution:** Instead of scaling pure layer depth, modern MoE architectures such as DeepSeek-V3 and Kimi K2 scale horizontally across hundreds of fine-grained routed experts per layer while keeping the active parameter count per token between ~32B and ~37B.

## Attention Mechanics and KV-Cache Memory Dynamics

The critical bottleneck governing high-throughput serving across extended context horizons is the physical memory footprint of the key-value (KV) cache. In standard Multi-Head Attention (MHA), memory requirements scale linearly with sequence length $S$, transformer layer count $L$, hidden dimension $d_{\text{model}}$, and numerical precision byte size $b_p$:

$$
\mathrm{Memory}_{\mathrm{MHA}} = 2 \cdot b_p \cdot L \cdot S \cdot d_{\text{model}}
$$

For a 61-layer MHA architecture with $d_{\text{model}} = 7{,}168$, a 128,000-token context, and 16-bit precision ($b_p = 2$), a single active sequence consumes approximately 213.5 GB of HBM:

$$
2 \times 2 \times 61 \times 128{,}000 \times 7{,}168
\approx 2.238 \times 10^{11}\ \text{bytes}
\approx 213.5\ \text{GB}
$$

Because this exceeds the physical capacity of an 80 GB NVIDIA H100 GPU for one user request, frontier designs have shifted toward Grouped-Query Attention (GQA) and Multi-Head Latent Attention (MLA).

### Grouped-Query Attention

GQA reduces KV-cache dimensionality by allowing multiple query heads to share a single key-value head pair. Llama 3.1 405B maps 128 query heads to 8 or 16 KV heads, achieving an 8:1 to 16:1 memory compression ratio relative to MHA. Qwen 2.5 72B pairs 64 query heads with 8 KV heads, while Mistral Large 2 pairs 96 query heads with 8 KV heads.

### Multi-Head Latent Attention

MLA introduces low-rank projection compression into the attention computation. The hidden representation is down-projected into a compressed latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$, where $d_c = 512$. A decoupled RoPE key vector $k_t^R \in \mathbb{R}^{d_R}$, where $d_R = 64$, retains positional awareness:

$$
d_{\text{cache}} = d_c + d_R = 512 + 64 = 576
$$

During autoregressive generation, key and value up-projection matrices can be absorbed into the query projection operations. MLA therefore avoids materializing uncompressed key and value vectors, achieving more than a 93% reduction in KV-cache memory compared to standard MHA.

## Dense Parameter Scaling Versus Fine-Grained Sparse Routing

Dense models execute the same parameter set for every token, while sparse MoE models decouple total parameter capacity from per-token compute FLOPs.

### Dense Scaling

Meta's Llama 3.1 405B comprises 126 hidden layers, $d_{\text{model}} = 16{,}384$, and a SwiGLU feed-forward network with $d_{\text{ff}} = 53{,}248$. Processing a token activates the entire 405-billion-parameter network. Its unquantized 16-bit weights require over 810 GB of VRAM, necessitating tensor and pipeline parallelism or INT4/FP8 quantization.

### Fine-Grained MoE Routing

DeepSeek-V3 expands total capacity to 671 billion parameters while activating only 37 billion per token. DeepSeekMoE uses 1 shared expert and 256 routed experts, selecting 8 routed experts across 4 affinity groups for each token. An auxiliary-loss-free balancing mechanism dynamically adjusts gate-logit biases to prevent routing collapse.

At cluster scale, Expert Parallelism (EP) distributes experts across accelerator devices. Overlapped all-to-all dispatch and combine kernels reduce the communication cost of routing, allowing DeepSeek-V3 to retain ultra-large representational capacity with compact-model serving economics.

## Test-Time Compute, Extended Thinking, and Long Output Horizons

Frontier architectures increasingly shift computational focus from pre-training parameter expansion toward inference-time reasoning and test-time compute allocation. Modern reasoning architectures expand output limits up to 100,000 and 128,000 tokens, enabling internal reflection, code validation, and iterative error correction.

### Hybrid and Extended-Thinking Models

Claude 3.7 Sonnet combines standard response execution with an Extended Thinking mode. Its internal reasoning budget ranges from 1,024 tokens up to 128,000 tokens, while standard mode supports direct completions up to 8,192 tokens. Extended beta configurations use the `output-128k-2025-02-19` protocol.

### Reasoning Models and Decode-Time Costs

OpenAI's o1 and o3-mini models allocate up to 100,000 completion tokens per request through the `max_completion_tokens` parameter. Hidden reasoning tokens remain within the context window and consume sequence budget and memory bandwidth. In o3-mini, discrete effort settings (`low`, `medium`, and `high`) modulate reasoning depth.

Long outputs shift bottlenecks from compute-bound prefill to memory-bandwidth-bound decode. Multi-Token Prediction (MTP), as used in DeepSeek-V3, trains the model to predict multiple consecutive tokens ($k > 1$) per forward pass. Serving systems combine this with prefix caching to avoid recomputing persistent prompts and reasoning trajectories.

## Architectural Synthesis

The frontier LLM landscape has moved past simple parameter scaling toward structural efficiency, memory compression, and adaptive compute allocation. Dense models such as Llama 3.1 405B established the boundaries of pure scale, while their operational requirements accelerated adoption of sparse routing and compressed attention.

DeepSeek-V3 demonstrates that fine-grained sparse MoE routing combined with MLA can reduce KV-cache memory footprints and hardware costs while retaining frontier capabilities. Claude 3.7 Sonnet, OpenAI o1, and o3-mini show that allocating additional inference compute can improve performance in complex reasoning domains.

Future architectures will likely pair fine-grained sparse routing and latent attention compression with unified, test-time reasoning loops, establishing highly efficient long-context systems.
