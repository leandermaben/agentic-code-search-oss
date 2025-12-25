# Agentic Code Search OSS - Comprehensive Developer Documentation

**Last Updated:** 2025-12-25
**Author:** Comprehensive Repository Documentation
**Purpose:** Complete guide to understanding, modifying, and running experiments in this repository

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem & Solution](#2-problem--solution)
3. [Architecture & Design Principles](#3-architecture--design-principles)
4. [Technology Stack](#4-technology-stack)
5. [Directory Structure](#5-directory-structure)
6. [OpenHands SDK Integration](#6-openhands-sdk-integration)
7. [SkyRL Framework](#7-skyrl-framework)
8. [Training Pipeline](#8-training-pipeline)
9. [Environment & Dataset](#9-environment--dataset)
10. [Reward System](#10-reward-system)
11. [Configuration System](#11-configuration-system)
12. [Key Components Deep Dive](#12-key-components-deep-dive)
13. [Running Experiments](#13-running-experiments)
14. [Making Modifications](#14-making-modifications)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

### What Is This Project?

**Agentic Code Search OSS** is an open-source implementation of a **specialized AI agent** trained via **Reinforcement Learning (RL)** to solve the **code localization problem** - finding relevant files and code snippets in large codebases.

### Why Does It Matter?

Current LLM-based coding agents face a critical bottleneck: **slow and inefficient context retrieval**. When an agent needs to edit code, it first must find which files to modify. Existing approaches:
- Are **slow** (many sequential tool calls)
- Are **inefficient** (poor search strategies)
- Have **low recall** (miss relevant files)

This project trains a **small, fast, specialized model** (8B parameters) that can:
- **Quickly** locate relevant code (low latency)
- **Accurately** identify all needed files (high precision & recall)
- **Efficiently** search using parallel tool calls

### Inspiration

Heavily inspired by [Cognition AI's SWE-grep](https://cognition.ai/blog/swe-grep) blog post, which demonstrated that specialized agents trained with RL can dramatically outperform general-purpose LLMs on code localization.

---

## 2. Problem & Solution

### The Code Localization Problem

**Given:**
- A software repository (potentially millions of lines of code)
- A problem statement (e.g., "Fix the bug where users can't login after password reset")

**Find:**
- All files that need to be edited to solve the problem
- Ideally, the specific functions/classes/lines within those files

### Challenges

1. **Search Space:** Repositories can have thousands of files
2. **Latency Requirements:** Users expect fast responses
3. **Precision vs Recall Trade-off:** Must find all relevant files without overwhelming with irrelevant ones
4. **Tool Efficiency:** Sequential searches are slow; need parallel execution

### The RL-Based Solution

Instead of using a general-purpose LLM (like GPT-4 or Claude), this project:

1. **Starts with a small, efficient base model** (Qwen3-8B - only 8 billion parameters)
2. **Fine-tunes it with RL** on real software engineering tasks (SWE-bench dataset)
3. **Teaches it to use search tools efficiently** (bash commands like grep, ripgrep, find)
4. **Rewards good localization** (F1 score on predicted vs actual files)
5. **Encourages parallel tool calling** (multiple searches simultaneously)

**Result:** A specialized agent that's faster, more accurate, and more efficient than general LLMs at code localization.

---

## 3. Architecture & Design Principles

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop (SkyRL)                     │
│                                                              │
│  ┌────────────┐      ┌──────────────┐     ┌──────────────┐ │
│  │  Dataset   │─────▶│  Generator   │────▶│   Trainer    │ │
│  │ SWE-bench │      │ (OpenHands)  │     │    (PPO)     │ │
│  └────────────┘      └──────────────┘     └──────────────┘ │
│         │                    │                     │        │
│         │                    ▼                     │        │
│         │            ┌──────────────┐              │        │
│         └───────────▶│ Environment  │◀─────────────┘        │
│                      │  (SWEGrepEnv)│                       │
│                      └──────────────┘                       │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │    Reward    │                       │
│                      │  Computation │                       │
│                      └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘

Flow:
1. Dataset provides problem statements + ground truth patches
2. Generator creates agent with LLM to solve problems
3. Environment executes agent's tool calls (bash commands)
4. Reward system scores agent's file predictions
5. Trainer updates model weights via PPO algorithm
```

### Design Principles

#### 1. **Modularity**

The codebase is designed with clear separation of concerns:

- **Environment** (`swe_grep_oss_env.py`): Defines the RL environment, handles tool execution
- **Generator** (`src/generator/`): Manages agent creation and inference
- **Rewards** (`src/rewards/`): Computable, swappable reward functions
- **Agent** (`src/agent/`): Custom agent logic for OpenHands
- **Metrics** (`src/metrics/`): Performance measurement (separate from rewards)
- **Prompts** (`src/prompts/`): System prompts and instruction templates

**Why:** Easy to swap components, experiment with different rewards, or change the agent implementation.

#### 2. **Registry Pattern**

Key components use a **registry pattern** for extensibility:

**Reward Registry** (`src/rewards/__init__.py`):
```python
def get_reward_function(name: str):
    """
    Get reward function by name from registry.
    Allows adding new rewards without modifying core code.
    """
    registry = {
        "file_localization_f1_reward": file_localization_f1_reward,
        "tool_use_reward": tool_use_reward,
        # ... more rewards can be added here
    }
    return registry[name]
```

**Usage in Config** (`configs/rewards/tool_use.yaml`):
```yaml
reward:
  - fn: tool_use_reward        # Name looks up function in registry
  - fn: turn_efficiency
  - fn: multilevel_localization_f1_reward
```

**Why:** Add new reward functions without touching training code - just register and reference in config.

#### 3. **Configuration-Driven**

Almost everything is configurable via **TOML and YAML files**:

- **Training parameters** (learning rate, batch size, epochs): `configs/swe-grep-oss/rl/train.toml`
- **Inference settings** (model, temperature, max tokens): `configs/swe-grep-oss/rl/infer.toml`
- **Reward composition** (which rewards, in what combination): `configs/rewards/*.yaml`

**Why:** Experiment without code changes - just modify configs and re-run.

#### 4. **Asynchronous & Distributed**

The system is designed for **high throughput** using:

- **Ray** for distributed execution across multiple nodes/GPUs
- **Async Python** (`asyncio`) for concurrent operations
- **vLLM** for efficient batched inference
- **Parallel tool execution** (agent can run 5 bash commands simultaneously)

**Why:** Training on hundreds of instances is compute-intensive; parallelism is essential.

---

## 4. Technology Stack

### Core Frameworks

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **RL Training** | SkyRL | `git:69ca4d9` | PPO-based reinforcement learning framework |
| **Base Model** | Qwen3-8B | 8B params | Small, efficient LLM for code tasks |
| **Inference Engine** | vLLM | 0.11.0 | Fast, batched LLM inference with tool calling |
| **Agent Framework** | OpenHands SDK | Workspace | Agent abstraction, tool management, conversation handling |
| **Environment** | Verifiers (SWE-Gym) | >=0.1.6 | Standardized env for code tasks with reward validation |
| **Dataset** | SWE-bench Lite | 300 instances | Real-world software engineering tasks |
| **Config Management** | Hydra + OmegaConf | - | Hierarchical configuration composition |
| **Distributed Computing** | Ray | - | Parallel training across nodes/GPUs |

### Key Python Dependencies

```toml
# From pyproject.toml
dependencies = [
    "skyrl-train",              # RL training framework
    "transformers==4.57.3",     # HuggingFace models
    "openhands-tools",          # Tool definitions
    "openhands-agent-server",   # Agent server
    "openhands-workspace",      # Workspace management
    "vllm==0.11.0",            # Inference engine
    "verifiers>=0.1.6.post0",  # Environment & validation
    "datasets>=4.0.0",         # Dataset loading
    "gcsfs>=2025.3.0",         # Google Cloud Storage
    "lmcache",                 # LLM caching for efficiency
]
```

### Why These Choices?

**SkyRL:**
- Built specifically for training LLMs with RL
- Supports asynchronous training (faster than synchronous)
- Good integration with vLLM for inference

**Qwen3-8B:**
- **Small** (8B params → fast inference, ~10-20ms per token on GPU)
- **Code-specialized** (trained on code data)
- **Tool-calling capable** (supports function calling natively)

**vLLM:**
- **Fastest inference engine** for LLMs (PagedAttention, continuous batching)
- **Native tool calling** (parses function calls from model output)
- **Scalable** (can serve multiple models, handle high request rates)

**OpenHands SDK:**
- **Purpose-built for code agents** (by OpenHands/All-Hands-AI team)
- **Tool abstraction** (easy to add new tools)
- **Workspace isolation** (safe code execution in containers)

**SWE-bench:**
- **Real-world tasks** (from actual GitHub issues)
- **Verifiable** (can check if files are correct via patch application)
- **Standardized** (widely used benchmark)

---

## 5. Directory Structure

```
/home/user/agentic-code-search-oss/
│
├── src/                           # Main source code
│   ├── agent/
│   │   └── agent.py              # Custom agent extending OpenHands Agent
│   │
│   ├── rewards/
│   │   ├── __init__.py           # Reward registry
│   │   ├── file_localization.py  # F1-score based rewards
│   │   ├── tool_use.py           # Tool usage rewards
│   │   ├── cosine_rewards.py     # Alternative reward formulations
│   │   └── module_rewards.py     # Module/function-level rewards
│   │
│   ├── generator/
│   │   └── code_search_generator.py  # Manages agent creation & RL loop
│   │
│   ├── prompts/
│   │   ├── system_prompt.py      # Hard-coded system prompt
│   │   ├── prompt_builder.py     # Builds prompts from templates
│   │   └── templates/            # Jinja2 templates
│   │       ├── system_prompt.j2
│   │       ├── file_module.j2
│   │       └── file_localization.j2
│   │
│   ├── tools/
│   │   └── bash.py               # Bash tool for code search
│   │
│   ├── metrics/
│   │   ├── efficiency_metrics.py    # Token usage, latency, tool counts
│   │   └── trajectory_metrics.py    # Trajectory-level stats
│   │
│   ├── utils/
│   │   ├── instance.py           # Clone repositories at specific commits
│   │   ├── dataset.py            # Extract functions/modules from patches
│   │   └── parse_patch.py        # Parse git patch format
│   │
│   ├── train.py                  # Training entry point
│   ├── build_dataset.py          # Dataset building script
│   ├── async_trainer.py          # Custom async PPO trainer
│   └── constants.py              # Project constants
│
├── configs/                      # Configuration files
│   ├── swe-grep-oss/
│   │   └── rl/
│   │       ├── train.toml        # Training hyperparameters
│   │       └── infer.toml        # Inference settings
│   │
│   └── rewards/
│       ├── tool_use.yaml         # Tool use + efficiency rewards
│       └── cosine.yaml           # Cosine-based rewards
│
├── scripts/
│   ├── run_training.sh           # Main training script
│   └── clone_repos.py            # Clone SWE-bench repositories
│
├── data/                         # Training data (created by build_dataset.py)
│   ├── train.parquet
│   └── eval.parquet
│
├── tests/                        # Test suite
│   ├── test_single_prompt.py
│   ├── test_single_file_localization.py
│   └── metrics/
│       └── test_efficiency_metrics.py
│
├── software-agent-sdk/           # OpenHands SDK (git submodule)
│   ├── openhands-sdk/
│   ├── openhands-tools/
│   ├── openhands-workspace/
│   └── openhands-agent-server/
│
├── swe_grep_oss_env.py          # Environment definition (top-level)
├── pyproject.toml               # Python project metadata & dependencies
└── README.md                    # Project overview
```

### Key Files Explained

**`src/train.py`** - The main entry point for training
- Loads configuration via Hydra
- Initializes Ray for distributed training
- Creates `CodeSearchPPOExp` or `AsyncCodeSearchPPOExp` based on config
- Starts the training loop

**`swe_grep_oss_env.py`** - Defines the RL environment
- Extends `verifiers.StatefulToolEnv`
- Registers the `bash` tool
- Defines completion criteria (max turns, context limit, or `<files>` tag present)
- Executes tool calls and returns results
- Computes rewards based on file localization accuracy

**`src/generator/code_search_generator.py`** - The RL generator
- Creates OpenHands agent instances
- Runs conversations with the LLM
- Executes tool calls via the agent
- Computes rewards using reward functions
- Collects metrics for logging
- Returns trajectories for PPO training

**`src/agent/agent.py`** - Custom agent implementation
- Extends OpenHands `Agent` class
- Handles reasoning model support (extracts reasoning from `<think>` tags)
- Manages context window limits
- Supports parallel tool calling (up to 5 simultaneous bash calls)

---

## 6. OpenHands SDK Integration

### What is OpenHands SDK?

The **OpenHands Software Agent SDK** (from https://github.com/OpenHands/software-agent-sdk) is a Python framework for building agents that work with code. It provides:

- **Agent abstraction** - Base class for creating LLM-powered agents
- **Tool framework** - Define and execute tools (bash, file editor, etc.)
- **Conversation management** - Handle multi-turn interactions
- **Workspace isolation** - Safe code execution in containers
- **Event system** - Track agent actions and responses

### Why Use OpenHands?

Instead of building agent infrastructure from scratch, OpenHands provides:
- **Pre-built tools** (TerminalTool, FileEditorTool, etc.)
- **LLM integration** with multiple backends (OpenAI, vLLM, etc.)
- **Message handling** with proper formatting for tool calls
- **Security** via sandboxed execution

### How This Project Uses OpenHands

#### 1. Custom Agent Class

**Location:** `src/agent/agent.py`

```python
from openhands.sdk import Agent

class CustomAgent(Agent):
    def step(self, conversation, on_event, on_token=None):
        # Custom logic to handle reasoning models
        # Extract reasoning from <think> tags
        if "</think>" in content:
            reasoning_content = content.split('</think>')[0]...
            message.reasoning_content = reasoning_content

        # Handle tool calls and execute actions
        if message.tool_calls and len(message.tool_calls) > 0:
            for tool_call in message.tool_calls:
                action_event = self._get_action_event(tool_call, ...)
                # Execute the tool
```

**Key Customizations:**
- **Reasoning extraction** - Qwen3 models can output reasoning in `<think>` tags; this code extracts and separates it
- **Token tracking** - Captures prompt and response token IDs for RL training
- **Context window management** - Handles condensation when context limit is reached

#### 2. Agent Creation in Generator

**Location:** `src/generator/code_search_generator.py:107`

```python
agent = CustomAgent(
    llm=LLM(
        usage_id="agent",
        model="litellm_proxy/willcb/Qwen3-8B",  # Model served by vLLM
        base_url="http://127.0.0.1:8000/v1/",   # vLLM server
        api_key="sk-xxx",                        # Dummy key
        temperature=1.0,                         # High temp for exploration
        litellm_extra_body={
            "return_token_ids": True,            # For RL training
            "include_stop_str_in_output": True,
        }
    ),
    tools=[Tool(name=TerminalTool.name)],       # Only bash tool
    security_analyzer=None,                      # No security checks (trusted env)
    system_prompt_filename="prompts/templates/system_prompt.j2"
)
```

#### 3. Conversation Management

**Location:** `src/generator/code_search_generator.py:125`

```python
conversation = Conversation(
    agent=agent,
    max_iteration_per_run=10,     # Max 10 turns
    visualizer=None,              # No UI visualization
    workspace=str(working_dir),   # Repository path
)

# Send initial problem statement
input_message = get_instruction(instance, prompt_template, working_dir)
conversation.send_message(input_message)

# Run agent until completion
conversation.run()

# Extract results
messages = list(map(lambda event: event.model_dump(), conversation.state.events))
final_message = get_agent_final_response(conversation.state.events)
```

**Flow:**
1. Create conversation with agent and workspace
2. Send user message (problem statement)
3. Run agent loop (agent generates tool calls → tools execute → agent sees results → repeat)
4. Extract final response and all events for reward computation

#### 4. Tools Integration

**Location:** `src/tools/bash.py`

```python
from openhands.tools.terminal import TerminalTool

# In environment setup:
self.add_tool(tools.bash, args_to_skip=["cwd"])

# The bash tool definition:
def bash(command: str, cwd: Optional[str] = None) -> str:
    """Execute bash command in the repository."""
    # Implementation details...
```

The OpenHands SDK automatically:
- Converts Python functions to tool schemas
- Formats them for LLM tool calling
- Validates tool call arguments
- Executes tools and returns results

### Event Types in OpenHands

When you run a conversation, it generates **events**:

```python
# Sample event stream:
[
    {"kind": "MessageEvent", "source": "user", "content": "Find files for bug fix"},
    {"kind": "TokenEvent", "prompt_token_ids": [...], "response_token_ids": [...]},
    {"kind": "ActionEvent", "action": "bash", "args": {"command": "rg 'def login'"}},
    {"kind": "ObservationEvent", "content": "auth/login.py:42:def login(user):"},
    {"kind": "TokenEvent", ...},
    {"kind": "ActionEvent", "action": "bash", "args": {"command": "rg 'password reset'"}},
    {"kind": "MessageEvent", "source": "agent", "content": "<files>auth/login.py\nauth/password.py</files>"},
]
```

**Event Types:**
- **MessageEvent** - User or agent text messages
- **TokenEvent** - LLM generation with token IDs (for RL)
- **ActionEvent** - Tool calls from the agent
- **ObservationEvent** - Tool execution results

### Token Events for RL Training

The key innovation for RL training is **capturing token IDs**:

```python
# In CustomAgent, when LLM generates a response:
message = {
    "kind": "TokenEvent",
    "prompt_token_ids": [151643, 8948, ...],     # Input tokens
    "response_token_ids": [151644, 9023, ...],   # Generated tokens
}
```

These token IDs are used to compute **log probabilities** for PPO training:
- **Prompt tokens** → context for generation
- **Response tokens** → what to optimize (increase prob if reward high, decrease if low)

---

## 7. SkyRL Framework

### What is SkyRL?

**SkyRL** (from https://github.com/NovaSky-AI/SkyRL) is a reinforcement learning framework specifically designed for training **large language models (LLMs)** using **Proximal Policy Optimization (PPO)**.

### Why SkyRL?

Other RL frameworks (like OpenAI's RL toolkit or RLlib) are designed for traditional RL (e.g., game playing, robotics). SkyRL is optimized for:
- **LLM-specific PPO** - Handles text generation, token-level rewards
- **Distributed training** - Multi-GPU, multi-node via Ray
- **Async execution** - Faster than synchronous PPO
- **Integration with vLLM** - Efficient inference engine

### PPO Overview (Quick Primer)

**PPO (Proximal Policy Optimization)** is an RL algorithm that improves a policy (in this case, the LLM's text generation) by:

1. **Collect trajectories** - Run the current policy (LLM) on tasks
2. **Compute rewards** - Score how well the policy did
3. **Update policy** - Adjust model weights to increase probability of high-reward actions
4. **Repeat** - Iterate to improve

**Key PPO Concepts:**
- **Policy (π)** - The LLM that generates text (tool calls, responses)
- **Reward (R)** - Score for a trajectory (e.g., F1 score for file localization)
- **Value function (V)** - Estimates expected future reward
- **Advantage (A)** - How much better an action was than expected
- **Clipped objective** - Prevents too-large policy updates (stability)

### How This Project Uses SkyRL

#### 1. Training Entry Point

**Location:** `src/train.py`

```python
from skyrl_train.entrypoints.main_base import BasePPOExp

class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        # Create custom generator (OpenHands-based)
        return CodeSearchGenerator(...)

@hydra.main(config_path=config_dir, config_name="ppo_base_config")
def main(cfg: DictConfig):
    # Load reward configuration from YAML
    if hasattr(cfg.generator, "reward"):
        with open(cfg.generator.reward, "r") as f:
            reward_cfg = OmegaConf.load(f)
        cfg.generator.reward = reward_cfg.reward

    # Initialize Ray for distributed training
    initialize_ray(cfg)

    # Launch training experiment
    ray.get(skyrl_entrypoint.remote(cfg))
```

**What `BasePPOExp` Provides:**
- Dataset loading
- Tokenizer initialization
- Inference engine setup (vLLM)
- Trainer creation (PPO optimizer)
- Checkpointing
- Logging

#### 2. Custom Generator

SkyRL expects a **generator** that:
- Takes prompts (problem statements)
- Generates completions (agent trajectories)
- Computes rewards
- Returns token IDs and rewards for PPO

**Location:** `src/generator/code_search_generator.py`

```python
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator

class CodeSearchGenerator(SkyRLGymGenerator):
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        # For each problem in batch:
        for i in range(len(prompts)):
            # 1. Create agent + conversation
            agent = CustomAgent(llm=...)
            conversation = Conversation(agent=agent, ...)

            # 2. Run agent on problem
            conversation.send_message(problem_statement)
            conversation.run()

            # 3. Extract messages and final response
            messages = conversation.state.events
            final_message = get_agent_final_response(messages)

            # 4. Compute reward
            reward = compute_reward(final_message, ground_truth)

            # 5. Extract token IDs from TokenEvents
            token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
            response_ids = [msg["response_token_ids"] for msg in token_messages]
            prompt_ids = [msg["prompt_token_ids"] for msg in token_messages]

            # 6. Return for PPO training
            rollout_list.append((response_ids, reward, ...))

        return GeneratorOutput(
            response_ids=response_ids,
            rewards=rewards,
            prompt_token_ids=prompt_ids,
            ...
        )
```

#### 3. Async vs Sync Training

**Synchronous PPO (Default):**
- Generate batch of trajectories → Wait for all to finish → Update policy → Repeat
- Simple but slow (blocked waiting for slowest trajectory)

**Asynchronous PPO (Faster):**
- Generate trajectories continuously in background
- Update policy as soon as enough data is collected
- Don't wait for slow trajectories

**Location:** `src/async_trainer.py`

```python
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer

class CustomFullyAsyncRayPPOTrainer(FullyAsyncRayPPOTrainer):
    # Custom async training logic
    # Overlaps trajectory generation with policy updates
```

**To enable async training:**
```yaml
# In config
run_async_trainer: true
```

#### 4. PPO Update Loop

**What SkyRL does behind the scenes:**

```python
# Simplified PPO loop:
for epoch in range(num_epochs):
    # 1. Generate trajectories with current policy
    rollouts = generator.generate(problems)

    # 2. Compute advantages (how good were actions?)
    advantages = compute_gae(rewards, values)

    # 3. Multiple optimization epochs on collected data
    for _ in range(ppo_epochs):
        # 4. Compute PPO loss
        # Compare new policy to old policy
        ratio = π_new(action) / π_old(action)
        clipped_ratio = clip(ratio, 1-ε, 1+ε)
        loss = -min(ratio * advantage, clipped_ratio * advantage)

        # 5. Update model weights
        optimizer.step()

    # 6. Save checkpoint
    if epoch % checkpoint_interval == 0:
        save_checkpoint()
```

---

## 8. Training Pipeline

### End-to-End Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATASET PREPARATION                                          │
│    └─ build_dataset.py: Load SWE-bench, extract ground truth   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. ENVIRONMENT SETUP                                            │
│    └─ Clone repositories at specific commits                   │
│    └─ Define tools (bash) and completion criteria              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TRAINING INITIALIZATION                                      │
│    ├─ Load config (Hydra)                                      │
│    ├─ Initialize Ray cluster                                   │
│    ├─ Start vLLM inference engine                              │
│    ├─ Load base model (Qwen3-8B) + LoRA adapters              │
│    └─ Create PPO trainer                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. TRAJECTORY GENERATION (Generator)                            │
│    For each problem:                                            │
│    ├─ Create agent with LLM                                    │
│    ├─ Send problem statement                                   │
│    ├─ Agent generates tool calls (bash commands)               │
│    ├─ Execute tools in repository                              │
│    ├─ Agent sees results, generates more calls                 │
│    └─ Continue until <files> tag or max turns                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. REWARD COMPUTATION                                           │
│    ├─ Extract predicted files from agent's final message       │
│    ├─ Compare to ground truth files (from patch)               │
│    ├─ Compute F1 score (precision & recall)                    │
│    ├─ Add tool efficiency rewards                              │
│    └─ Assign reward to each step (discounted)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. PPO UPDATE (Trainer)                                         │
│    ├─ Collect batch of trajectories + rewards                  │
│    ├─ Compute advantages (GAE)                                 │
│    ├─ Run multiple PPO optimization epochs                     │
│    ├─ Update model weights (increase prob of high reward)      │
│    └─ Update value function                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. EVALUATION & CHECKPOINTING                                   │
│    ├─ Run evaluation on held-out set                           │
│    ├─ Log metrics (rewards, F1, tool usage)                    │
│    ├─ Save checkpoint every N steps                            │
│    └─ Continue training or terminate                           │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Component Breakdown

#### Dataset Building

**Script:** `src/build_dataset.py`

```bash
uv run src/build_dataset.py --output ../data/
```

**What it does:**
1. Loads SWE-bench_Lite dataset (300 instances)
2. For each instance:
   - Extracts files modified in the patch (ground truth)
   - Extracts functions/modules modified (for module-level rewards)
3. Splits into train (80%) and validation (20%)
4. Saves as Parquet files

**Output:**
```
data/
├── train.parquet  # 240 training instances
└── eval.parquet   # 60 validation instances
```

#### Repository Cloning

**Utility:** `src/utils/instance.py:clone_instance()`

```python
def clone_instance(repo_name, commit_id, instance_id, workspace):
    """
    Clone repository at specific commit for an instance.

    Args:
        repo_name: e.g., "django/django"
        commit_id: Git commit SHA
        instance_id: Unique instance identifier
        workspace: Path to clone into (e.g., /tmp/testbed/uuid/)

    Returns:
        (status, working_dir)
    """
    # Clone from GitHub
    # Checkout specific commit
    # Return path to repository
```

**Why clone per-instance?**
- Each SWE-bench task targets a specific commit
- Parallel training needs isolated workspaces
- Prevents conflicts between concurrent trajectories

---

## 9. Environment & Dataset

### SWE-bench Dataset

**What is SWE-bench?**
- A benchmark of **real software engineering tasks** from GitHub
- Each task is a bug/feature from an actual pull request
- Includes problem statement + patch (ground truth solution)
- **SWE-bench Lite** = 300 carefully curated instances

**Dataset Structure:**
```python
{
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "base_commit": "abc123def456...",
    "problem_statement": "The login form crashes when...",
    "patch": "diff --git a/auth/login.py ...",
    "test_patch": "diff --git a/tests/test_auth.py ..."
}
```

**How Ground Truth is Extracted:**
```python
# From swe_grep_oss_env.py:194
def transform_row(row):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["problem_statement"]},
        ],
        "answer": json.dumps(parse_patch(row["patch"])),  # Files from patch
    }
```

**`parse_patch()` Function:**
```python
# From src/utils/parse_patch.py
def parse_patch(patch_string: str) -> list[str]:
    """
    Extract list of files modified in a git patch.

    Example patch:
    diff --git a/src/main.py b/src/main.py
    --- a/src/main.py
    +++ b/src/main.py
    ...

    Returns: ["src/main.py"]
    """
    # Uses regex to extract file paths from diff headers
```

### Environment Definition

**Location:** `swe_grep_oss_env.py`

```python
from verifiers import StatefulToolEnv

class SWEGrepEnv(StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(tools.bash, args_to_skip=["cwd"])

    async def is_completed(self, messages, state, **kwargs) -> bool:
        """
        Episode ends when:
        1. Agent returns <files> XML tags, OR
        2. Max turns (8) reached, OR
        3. Context window limit exceeded
        """
        max_turns_reached = await self.max_turns_reached(state)
        prompt_too_long = await self.prompt_too_long(state)

        # Check for <files> tag in last message
        has_files_tag = False
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.get("role") == "assistant":
                content = last_message.get("content", "")
                if "<files>" in content and "</files>" in content:
                    has_files_tag = True

        return has_files_tag or max_turns_reached or prompt_too_long

    async def env_response(self, messages, state, **kwargs):
        """
        Execute tool calls and return results.
        """
        tool_messages = []
        tool_calls = messages[-1].get("tool_calls", [])
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args = json.loads(tool_call.get("function", {}).get("arguments", ""))

            # Inject repository path as working directory
            if tool_name == "bash":
                tool_args["cwd"] = get_instance_path(state["info"])

            # Execute tool
            tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
            tool_messages.append(tool_message)

        return tool_messages, state
```

**Key Design Decisions:**

1. **Single Tool (Bash Only)**
   - Simplifies the agent's decision space
   - Bash is flexible enough for all search operations
   - Aligns with human software engineering workflow

2. **Completion via `<files>` Tags**
   - Clear signal that agent is done
   - Easy to parse final answer
   - Prevents ambiguity

3. **Repository Injection via `cwd`**
   - Agent doesn't need to know repository path
   - Tools automatically execute in correct directory
   - Isolates different concurrent episodes

### System Prompt

**Location:** `src/prompts/system_prompt.py`

The system prompt is **critical** - it defines the agent's behavior. Key directives:

**Core Objective:**
```
You are a specialized code localization agent. Your sole objective is to
identify and return the files in the codebase that are relevant to the
user's query.
```

**Tool Usage:**
```
- You MUST use the bash tool to search and explore the codebase
- Execute bash commands like: rg, grep, find, ls, cat, head, tail, sed
- Use parallel tool calls: invoke bash tool up to 5 times concurrently
- NEVER exceed 5 parallel tool calls per turn
```

**Critical Context Management:**
```
- NEVER read entire large files with `cat`
- ALWAYS check file size first: `wc -l path/to/file.py`
- For files > 100 lines, read in chunks:
  * Use `sed -n '1,100p' file.py` to read lines 1-100
  * Use `sed -n '101,200p' file.py` to read lines 101-200
```

**Output Format:**
```
<files>
src/main.py
src/utils/helper.py
tests/test_main.py
</files>
```

**Why This Matters:**
- Teaches agent to use tools efficiently
- Prevents context window overflow
- Encourages parallel exploration
- Provides clear success criteria

---

## 10. Reward System

The reward system is **the heart of RL training** - it defines what the agent learns to optimize.

### Reward Architecture

**Design Pattern:** Modular, composable rewards via registry

```python
# configs/rewards/tool_use.yaml
reward:
  - fn: tool_use_reward               # Ratio of tool calls to turns
  - fn: turn_efficiency               # Penalty for too many turns
  - fn: multilevel_localization_f1_reward  # File localization accuracy
```

Each reward function is:
- **Independent** - Can be tested and debugged separately
- **Composable** - Multiple rewards can be combined
- **Configurable** - Weights and parameters via YAML

### Reward Registry Pattern

**Location:** `src/rewards/__init__.py`

```python
REWARD_REGISTRY = {}

def reward(name: str):
    """Decorator to register a reward function."""
    def decorator(func):
        REWARD_REGISTRY[name] = func
        return func
    return decorator

def get_reward_function(reward_name: str):
    """Get a reward function by name."""
    if reward_name not in REWARD_REGISTRY:
        raise ValueError(f"Reward function '{reward_name}' not found")
    return REWARD_REGISTRY[reward_name]

# Auto-discover and import all reward modules
_auto_load_rewards()
```

**Adding a New Reward:**

1. Create file `src/rewards/my_reward.py`
2. Import decorator and use it:
```python
from src.rewards import reward

@reward("my_custom_reward")
def my_custom_reward(messages, final_message, instance, **kwargs):
    # Compute reward based on trajectory
    return reward_value
```

3. Reference in config:
```yaml
reward:
  - fn: my_custom_reward
    args:
      some_param: value
```

### Core Reward Functions

#### 1. File Localization F1 Reward

**Location:** `src/rewards/result_tool_f1.py` (used via multilevel wrapper)

```python
def multilevel_localization_f1_reward(final_message, instance, **kwargs):
    """
    F1 score for file localization.

    Ground truth: Files extracted from patch
    Predicted: Files extracted from agent's <files> tags
    """
    # Parse agent's response
    predicted_files = extract_files_from_xml(final_message)

    # Get ground truth from instance
    ground_truth_files = instance["target"]

    # Normalize paths (remove leading ./)
    predicted = set(normalize_path(f) for f in predicted_files)
    actual = set(normalize_path(f) for f in ground_truth_files)

    # Compute F1
    if len(predicted) == 0 and len(actual) == 0:
        return 1.0  # Perfect if both empty
    if len(predicted) == 0 or len(actual) == 0:
        return 0.0  # Zero if one is empty

    true_positives = len(predicted & actual)
    precision = true_positives / len(predicted)
    recall = true_positives / len(actual)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

**F1 Score Intuition:**
- **Precision** = (Correct files found) / (Total files returned)
  - High precision → Few false positives
- **Recall** = (Correct files found) / (Total correct files)
  - High recall → Few false negatives
- **F1** = Harmonic mean of precision and recall
  - Balances both metrics

**Example:**
```
Ground truth: [auth/login.py, auth/password.py, auth/session.py]
Agent predicts: [auth/login.py, auth/password.py, utils/helper.py]

True positives: 2 (login.py, password.py)
False positives: 1 (helper.py)
False negatives: 1 (session.py)

Precision = 2/3 = 0.67
Recall = 2/3 = 0.67
F1 = 2 * (0.67 * 0.67) / (0.67 + 0.67) = 0.67
```

#### 2. Tool Use Reward

**Location:** `src/rewards/tool_use.py`

```python
@reward("tool_use_reward")
def tool_use_reward(messages, **kwargs) -> float:
    """
    Encourage agent to use tools frequently.
    Reward = (Number of tool calls) / (Number of turns)
    """
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]

    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)

    if num_turns == 0:
        return 0.0

    return num_tool_calls / num_turns
```

**Intuition:**
- Agents should use tools actively (not just generate text)
- Higher ratio → More exploration
- Encourages parallel tool calling (5 tools in 1 turn → ratio of 5)

**Example:**
```
Turn 1: Generate 3 tool calls → ratio = 3
Turn 2: Generate 5 tool calls → ratio = 5
Turn 3: Generate 2 tool calls → ratio = 2
Average: (3 + 5 + 2) / 3 = 3.33
```

#### 3. Turn Efficiency Reward

**Location:** `src/rewards/tool_use.py`

```python
@reward("turn_efficiency")
def turn_efficiency(messages, max_turns=5, **kwargs) -> float:
    """
    Penalize agents for using too many turns.
    Encourages solving tasks quickly.
    """
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]

    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)

    if num_turns <= 1:
        return 0.0

    if num_tool_calls > 1:
        # Reward if within max_turns, penalize if exceeds
        if num_turns <= max_turns:
            return 1.0
        else:
            return max(0.0, 1.0 - (num_turns - max_turns) * 0.1)

    return 0.0
```

**Intuition:**
- Fewer turns → Faster localization → Lower latency
- Penalty grows linearly after `max_turns`
- Balances exploration (need tools) with efficiency (minimize turns)

### Reward Combination

**Location:** `src/generator/code_search_generator.py:246`

```python
reward = 0
reward_dict = {}

for reward_fn_args in self.generator_cfg.reward:
    reward_fn = get_reward_function(reward_fn_args["fn"])
    input_args = {
        "final_message": final_message,
        "messages": messages,
        "instance": instance,
        **reward_fn_args.get("args", {})
    }

    reward_value = reward_fn(**input_args)
    reward += reward_value  # Simple sum
    reward_dict[reward_fn_args["fn"]] = reward_value

print(f"Reward details: {reward_dict}, Total reward: {reward}")
```

**Example Output:**
```
Reward details: {
    'tool_use_reward': 3.2,
    'turn_efficiency': 1.0,
    'multilevel_localization_f1_reward': 0.75
}, Total reward: 4.95
```

### Reward Discounting

**Location:** `src/generator/code_search_generator.py:302`

```python
# Assign reward to each step with gamma discounting
gamma = 0.9
num_steps = len(token_messages)
for idx, message in enumerate(token_messages):
    step_reward = reward * gamma**(num_steps - idx - 1)
    rollout_list.append((response_ids, step_reward, ...))
```

**Why Discount?**
- **Credit assignment** - Later steps get higher reward (closer to outcome)
- **Temporal structure** - Earlier actions have less direct impact
- **Standard RL practice** - gamma ∈ [0, 1]

**Example:**
```
Total reward: 5.0
3 steps in trajectory
gamma = 0.9

Step 0 reward: 5.0 * 0.9^(3-0-1) = 5.0 * 0.9^2 = 5.0 * 0.81 = 4.05
Step 1 reward: 5.0 * 0.9^(3-1-1) = 5.0 * 0.9^1 = 5.0 * 0.9 = 4.5
Step 2 reward: 5.0 * 0.9^(3-2-1) = 5.0 * 0.9^0 = 5.0 * 1 = 5.0
```

---

## 11. Configuration System

### Hydra + OmegaConf

This project uses **Hydra** for hierarchical configuration management.

**Why Hydra?**
- **Composition** - Combine multiple config files
- **Overrides** - Change parameters from command line
- **Type safety** - Structured configs with validation
- **Sweeps** - Run multiple experiments with different configs

### Configuration Files

#### 1. Training Config

**Location:** `configs/swe-grep-oss/rl/train.toml`

```toml
max_steps = 150  # Total training steps

[model]
name = "willcb/Qwen3-8B"

[model.ac]
freq = 1  # Actor-critic update frequency

[model.experimental.lora]
rank = 64  # LoRA rank (higher = more parameters)
alpha = 512  # LoRA scaling factor
dropout = 0.0
target_modules = [
    "q_proj",      # Attention projections
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",   # MLP projections
    "up_proj",
    "down_proj"
]

[optim]
lr = 1e-5  # Learning rate

[ckpt]
interval = 10  # Save checkpoint every 10 steps
```

**LoRA Explained:**
- **LoRA (Low-Rank Adaptation)** - Efficient fine-tuning method
- Instead of updating all 8B parameters, update small low-rank matrices
- **rank = 64** → Only ~0.1% of parameters are trainable
- **Faster training, less memory, prevents overfitting**

**Target Modules:**
- `q_proj, k_proj, v_proj, o_proj` - Attention mechanism
- `gate_proj, up_proj, down_proj` - MLP (feedforward) layers
- These are the most impactful layers for fine-tuning

#### 2. Inference Config

**Location:** `configs/swe-grep-oss/rl/infer.toml`

```toml
gpu_memory_utilization = 0.7  # Use 70% of GPU memory

[model]
name = "willcb/Qwen3-8B"
enforce_eager = true  # Disable CUDA graphs (for debugging)
enable_auto_tool_choice = true  # Enable tool calling
tool_call_parser = "hermes"  # Parser for tool call format
```

**vLLM Parameters:**
- `gpu_memory_utilization` - Controls memory allocation (higher = more throughput)
- `enable_auto_tool_choice` - Model automatically generates tool calls
- `tool_call_parser = "hermes"` - Qwen3 uses Hermes tool calling format

#### 3. Reward Config

**Location:** `configs/rewards/tool_use.yaml`

```yaml
reward:
  - fn: tool_use_reward
  - fn: turn_efficiency
  - fn: multilevel_localization_f1_reward
```

**Simple, composable design:**
- List of reward functions to apply
- Functions are looked up in reward registry
- Results are summed

### Using Configurations

**Training Script:** `scripts/run_training.sh`

```bash
uv run src/train.py \
    generator=code_search_generator \
    generator.reward=configs/rewards/tool_use.yaml \
    trainer=ppo_trainer \
    trainer.policy.model.path=willcb/Qwen3-8B \
    trainer.policy.model.lora.rank=64 \
    trainer.optimizer.lr=1e-5 \
    trainer.epochs=20 \
    generator.batch_size=4
```

**Hydra Override Syntax:**
```bash
# Override nested config
trainer.policy.model.path=different/model

# Override top-level
trainer.epochs=50

# Specify reward file
generator.reward=configs/rewards/cosine.yaml
```

### Configuration Loading

**Location:** `src/train.py:66`

```python
@hydra.main(config_path=config_dir, config_name="ppo_base_config")
def main(cfg: DictConfig):
    # Load reward config from YAML
    if hasattr(cfg.generator, "reward"):
        with open(cfg.generator.reward, "r") as f:
            reward_cfg = OmegaConf.load(f)
        cfg.generator.reward = reward_cfg.reward
    else:
        # Default reward
        with open_dict(cfg):
            cfg.generator.reward = [
                {"fn": "multilevel_localization_f1_reward"},
            ]
```

**Flow:**
1. Hydra loads base config (`ppo_base_config.toml`)
2. Applies overrides from command line
3. Custom logic loads reward YAML and merges
4. Final config passed to experiment

---

## 12. Key Components Deep Dive

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                            │
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Dataset    │────────┐                                      │
│  │ (SWE-bench) │        │                                      │
│  └──────────────┘        ▼                                      │
│                   ┌──────────────┐                              │
│  ┌──────────────┐ │  Generator   │ ┌──────────────┐           │
│  │ Inference    │◀│ (CodeSearch) │▶│   Trainer    │           │
│  │ Engine (vLLM)│ └──────────────┘ │    (PPO)     │           │
│  └──────────────┘        │         └──────────────┘           │
│         │                │                │                     │
│         │                ▼                │                     │
│         │        ┌──────────────┐         │                     │
│         └───────▶│   Agent +    │         │                     │
│                  │ Conversation │         │                     │
│                  └──────────────┘         │                     │
│                         │                 │                     │
│                         ▼                 │                     │
│                  ┌──────────────┐         │                     │
│                  │ Environment  │         │                     │
│                  │  (SWEGrepEnv)│         │                     │
│                  └──────────────┘         │                     │
│                         │                 │                     │
│                         ▼                 │                     │
│                  ┌──────────────┐         │                     │
│                  │ Reward       │─────────┘                     │
│                  │ Computation  │                               │
│                  └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Generator (`CodeSearchGenerator`)

**Location:** `src/generator/code_search_generator.py`

**Responsibilities:**
- Create and manage agent instances
- Run conversations (trajectories)
- Execute tool calls via environment
- Compute rewards
- Extract token IDs for PPO
- Log metrics
- Save trajectories to disk/GCS

**Key Methods:**

**`generate(input_batch)`** - Main entry point
```python
async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
    # Input: Batch of problems
    # Output: Token IDs + rewards for PPO

    for i in range(len(prompts)):
        # Run one trajectory
        rollout = self.code_search_loop(
            prompts[i],
            env_extras[i],
            trajectory_id=trajectory_ids[i],
            ...
        )

    # Aggregate results
    return GeneratorOutput(
        response_ids=responses,
        rewards=rewards,
        prompt_token_ids=prompts,
        ...
    )
```

**`code_search_loop()`** - Single trajectory
```python
async def code_search_loop(self, prompt, env_extras, ...):
    # 1. Clone repository
    workspace = Path(f"/tmp/testbed/{uuid_str}/")
    clone_instance(repo_name, commit_id, workspace)

    # 2. Create agent
    agent = CustomAgent(llm=LLM(...), tools=[TerminalTool])
    conversation = Conversation(agent=agent, workspace=working_dir)

    # 3. Run agent
    conversation.send_message(problem_statement)
    conversation.run()

    # 4. Extract results
    messages = conversation.state.events
    final_message = get_agent_final_response(messages)

    # 5. Compute reward
    for reward_fn_args in self.generator_cfg.reward:
        reward_fn = get_reward_function(reward_fn_args["fn"])
        reward_value = reward_fn(final_message=final_message, messages=messages, ...)
        reward += reward_value

    # 6. Extract token IDs
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    rollout_list = []
    for idx, message in enumerate(token_messages):
        step_reward = reward * gamma**(num_steps - idx - 1)
        rollout_list.append((
            message["response_token_ids"],
            step_reward,
            message["prompt_token_ids"],
            ...
        ))

    # 7. Save trajectory
    save_to_disk(rollout_list, trajectory_id, ...)

    return rollout_list
```

### 2. Agent (`CustomAgent`)

**Location:** `src/agent/agent.py`

**Extends:** `openhands.sdk.Agent`

**Key Customization: Reasoning Extraction**

```python
class CustomAgent(Agent):
    def step(self, conversation, on_event, on_token=None):
        # ... (prepare messages, call LLM)

        # Extract reasoning from <think> tags (Qwen3 feature)
        content = message.content[0].text
        if "</think>" in content:
            reasoning_content = content.split('</think>')[0].split('<think>')[-1]
            content = content.split('</think>')[-1]
            message.content[0].text = content
            message.reasoning_content = reasoning_content

        # Handle tool calls
        if message.tool_calls:
            for i, tool_call in enumerate(message.tool_calls):
                action_event = self._get_action_event(
                    tool_call,
                    reasoning_content=message.reasoning_content if i == 0 else None
                )
                self._execute_actions(conversation, action_events, on_event)

        # Emit token IDs (for RL training)
        self._maybe_emit_vllm_tokens(llm_response, on_event)
```

**Why Reasoning Extraction?**
- Qwen3 models can output reasoning in `<think>` tags
- This separates reasoning from final response
- Allows training on both reasoning and action generation

### 3. Environment (`SWEGrepEnv`)

**Location:** `swe_grep_oss_env.py`

**Extends:** `verifiers.StatefulToolEnv`

**Core Methods:**

**`is_completed()`** - Check if episode should end
```python
async def is_completed(self, messages, state, **kwargs) -> bool:
    # Three termination conditions:
    # 1. Agent returns <files> tags
    has_files_tag = "<files>" in last_message and "</files>" in last_message

    # 2. Max turns reached (8 turns)
    max_turns_reached = await self.max_turns_reached(state)

    # 3. Context window exceeded
    prompt_too_long = await self.prompt_too_long(state)

    return has_files_tag or max_turns_reached or prompt_too_long
```

**`env_response()`** - Execute tools
```python
async def env_response(self, messages, state, **kwargs):
    tool_calls = messages[-1].get("tool_calls", [])

    for tool_call in tool_calls:
        # Parse tool arguments
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])

        # Inject repository path
        if tool_name == "bash":
            tool_args["cwd"] = get_instance_path(state["info"])

        # Execute tool
        tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
        tool_messages.append(tool_message)

    return tool_messages, state
```

### 4. Trainer (`CustomFullyAsyncRayPPOTrainer`)

**Location:** `src/async_trainer.py`

**Extends:** `skyrl_train.fully_async_trainer.FullyAsyncRayPPOTrainer`

**What it does:**
- Manages PPO training loop
- Coordinates generator and policy
- Computes advantages (GAE)
- Runs optimization epochs
- Saves checkpoints
- Logs metrics

**Async Training Benefits:**
- Overlaps trajectory generation with policy updates
- Doesn't wait for slow trajectories
- Higher throughput (2-3x faster than sync)

### 5. Metrics (`efficiency_metrics.py`, `trajectory_metrics.py`)

**Location:** `src/metrics/`

**Efficiency Metrics:**
```python
def compute_all_efficiency_metrics(messages, wall_clock_duration, ...):
    token_events = [m for m in messages if m["kind"] == "TokenEvent"]
    action_events = [m for m in messages if m["kind"] == "ActionEvent"]

    return {
        "num_turns": len(token_events),
        "num_tool_calls": len(action_events),
        "total_prompt_tokens": sum(len(m["prompt_token_ids"]) for m in token_events),
        "total_response_tokens": sum(len(m["response_token_ids"]) for m in token_events),
        "wall_clock_duration": wall_clock_duration,
        "tokens_per_second": total_tokens / wall_clock_duration,
        ...
    }
```

**Trajectory Metrics:**
```python
def compute_trajectory_metrics(messages):
    return {
        "has_final_response": has_files_tag(messages),
        "trajectory_length": len(messages),
        "unique_tool_calls": len(set(tool_names)),
        ...
    }
```

---

## 13. Running Experiments

### Prerequisites

**Hardware Requirements:**
- **GPU**: 2x A100 (80GB) recommended for training
  - Minimum: 1x A100 (40GB) for smaller experiments
- **CPU**: 32+ cores
- **RAM**: 512GB recommended
- **Storage**: 1TB+ (for datasets, checkpoints, trajectories)

**Software Requirements:**
```bash
# Python 3.13+
python --version

# CUDA 12.8 (for PyTorch + vLLM)
nvcc --version

# UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 1: Setup Environment

```bash
# Clone repository
git clone https://github.com/All-Hands-AI/agentic-code-search-oss
cd agentic-code-search-oss

# Initialize submodules (OpenHands SDK)
git submodule update --init --recursive

# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Step 2: Build Dataset

```bash
# Create data directory
mkdir -p data

# Build training/validation split
uv run src/build_dataset.py --output data/

# Output:
# data/train.parquet (240 instances)
# data/eval.parquet (60 instances)
```

**What this does:**
- Downloads SWE-bench_Lite dataset
- Extracts ground truth files from patches
- Splits 80/20 train/validation
- Saves as Parquet for fast loading

### Step 3: Clone Repositories (Optional)

For faster training, pre-clone all repositories:

```bash
# Clone all SWE-bench repositories
uv run scripts/clone_repos.py --output-dir ./swebench_repos

# This creates:
# swebench_repos/
#   ├── django__django/
#   ├── scikit-learn__scikit-learn/
#   └── ... (all repos at specific commits)
```

**Why pre-clone?**
- Avoids cloning during training (saves time)
- More deterministic (no network issues)
- Optional - training will clone on-demand if not present

### Step 4: Run Training

**Basic Training (Local):**

```bash
# Simple training run
uv run src/train.py \
    data.train_data="['data/train.parquet']" \
    data.val_data="['data/eval.parquet']" \
    trainer.policy.model.path=willcb/Qwen3-8B \
    trainer.epochs=20 \
    trainer.train_batch_size=4 \
    generator.reward=configs/rewards/tool_use.yaml
```

**Full Training (Production):**

```bash
# Use the provided script
bash scripts/run_training.sh -m willcb/Qwen3-8B -n 4 -d data
```

**Script Parameters:**
- `-m MODEL` - Model name (e.g., `willcb/Qwen3-8B`)
- `-n N_ROLLOUTS` - Rollouts per problem (default: 4)
- `-d DATA_PATH` - Path to dataset directory (default: `data/swe_smith`)
- `-s CKPT_PATH` - Checkpoint save directory (default: `ckpts/{model}`)

**Key Training Parameters Explained:**

```bash
# Data
data.train_data="['data/train.parquet']"  # Training data
data.val_data="['data/eval.parquet']"     # Validation data

# Model & LoRA
trainer.policy.model.path=willcb/Qwen3-8B  # Base model
trainer.policy.model.lora.rank=64          # LoRA rank
trainer.policy.model.lora.alpha=512        # LoRA alpha

# Training
trainer.epochs=20                          # Total epochs
trainer.train_batch_size=4                 # Batch size
trainer.eval_batch_size=100                # Eval batch size
trainer.policy.optimizer_config.lr=1e-6    # Learning rate

# Generation
generator.max_turns=20                     # Max agent turns
generator.num_inference_engines=2          # Parallel inference
generator.n_samples_per_prompt=4           # Rollouts per problem
generator.gpu_memory_utilization=0.6       # vLLM memory usage

# Checkpointing
trainer.ckpt_interval=10                   # Save every 10 steps
trainer.ckpt_path=ckpts/qwen3-8b           # Save directory
```

### Step 5: Monitor Training

**Weights & Biases (Recommended):**

```bash
# Set up W&B
export WANDB_API_KEY=your_key_here

# Training will automatically log to W&B
# View at: https://wandb.ai/your_username/code_search
```

**Metrics Logged:**
- **Rewards**: F1 score, tool usage, turn efficiency
- **Training**: Loss, gradient norms, learning rate
- **Efficiency**: Tokens/sec, wall clock time
- **Trajectory**: Num turns, tool calls, success rate

**Local Logs:**

```bash
# Check training logs
tail -f logs/*.out

# View trajectories
ls ckpts/qwen3-8b/trajectories/step_10/train/
# Each file: instance_id_repetition.json
```

**Sample Trajectory File:**
```json
{
  "instance_id": "django__django-12345",
  "target": ["auth/login.py", "auth/password.py"],
  "total_reward": 4.95,
  "reward_dict": {
    "tool_use_reward": 3.2,
    "turn_efficiency": 1.0,
    "multilevel_localization_f1_reward": 0.75
  },
  "parsed_final_message": "auth/login.py\nauth/password.py\nutils/helper.py",
  "messages": [...],
  "metrics_dict": {
    "num_turns": 3,
    "num_tool_calls": 9,
    "wall_clock_duration": 12.5
  }
}
```

### Step 6: Evaluate Checkpoints

```bash
# Evaluate a specific checkpoint
uv run vf-eval swe-grep-oss-env \
    --model-path ckpts/qwen3-8b/checkpoint_50 \
    --dataset-split test \
    --output-dir eval_results/

# View results
cat eval_results/metrics.json
```

**Evaluation Metrics:**
- **F1 Score** - Primary metric (precision & recall)
- **Precision** - % of predicted files that are correct
- **Recall** - % of correct files that were found
- **Latency** - Average time to localize
- **Tool Usage** - Avg tool calls per problem

### Step 7: Deploy Model

**Serve with vLLM:**

```bash
# Start vLLM server
vllm serve ckpts/qwen3-8b/checkpoint_best \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8000

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Find files for login bug"}
    ],
    "tools": [{"type": "function", "function": {"name": "bash", ...}}]
  }'
```

---

## 14. Making Modifications

### Adding a New Reward Function

**1. Create reward file:** `src/rewards/my_reward.py`

```python
from src.rewards import reward

@reward("my_reward")
def my_reward(messages, final_message, instance, **kwargs) -> float:
    """
    Custom reward based on your criteria.

    Args:
        messages: List of all conversation events
        final_message: Agent's final response
        instance: Dataset instance with ground truth

    Returns:
        Reward value (float)
    """
    # Example: Reward based on number of unique commands used
    action_events = [m for m in messages if m["kind"] == "ActionEvent"]
    unique_commands = set()

    for event in action_events:
        command = event.get("args", {}).get("command", "")
        # Extract command name (e.g., "rg" from "rg pattern")
        cmd_name = command.split()[0] if command else ""
        unique_commands.add(cmd_name)

    # Reward diversity of search strategies
    diversity_score = len(unique_commands) / 5.0  # Normalize by max expected
    return min(diversity_score, 1.0)
```

**2. Create reward config:** `configs/rewards/my_config.yaml`

```yaml
reward:
  - fn: my_reward
  - fn: multilevel_localization_f1_reward
```

**3. Use in training:**

```bash
uv run src/train.py \
    generator.reward=configs/rewards/my_config.yaml \
    ...
```

### Modifying the System Prompt

**Location:** `src/prompts/system_prompt.py` or `src/prompts/templates/system_prompt.j2`

**Option 1: Hardcoded (Python)**

Edit `src/prompts/system_prompt.py`:
```python
SYSTEM_PROMPT = """
You are a code localization agent with ENHANCED CAPABILITIES.

NEW DIRECTIVE:
- Use semantic search to find related code
- Consider file dependencies and imports
- ... (your modifications)
"""
```

**Option 2: Template (Jinja2)**

Edit `src/prompts/templates/system_prompt.j2`:
```jinja2
You are a specialized code localization agent.

Your objective is to {{ objective }}.

## TOOL USAGE
{% if enable_parallel %}
- Use parallel tool calls (up to {{ max_parallel }})
{% endif %}
...
```

Then pass variables in code:
```python
system_prompt_filename="prompts/templates/system_prompt.j2"
# OpenHands will render with context
```

### Adding a New Tool

**1. Define tool function:** `src/tools/my_tool.py`

```python
def semantic_search(query: str, top_k: int = 5, cwd: str = None) -> str:
    """
    Search codebase using semantic similarity.

    Args:
        query: Natural language query
        top_k: Number of results to return
        cwd: Working directory (injected by environment)

    Returns:
        Search results as formatted string
    """
    # Implementation using embeddings, vector DB, etc.
    # ...
    return results
```

**2. Register tool in environment:**

Edit `swe_grep_oss_env.py`:
```python
import src.tools as tools

class SWEGrepEnv(StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.semantic_search, args_to_skip=["cwd"])  # Add new tool
```

**3. Update system prompt** to teach agent about new tool:
```
## TOOLS AVAILABLE
- bash: Execute shell commands (rg, grep, find, etc.)
- semantic_search: Find code using natural language queries
```

**4. Test:**
```bash
# Run single instance to verify tool works
uv run tests/test_single_prompt.py --instance-id django__django-12345
```

### Changing Model

**Option 1: Different base model**

```bash
# Use different Qwen variant
uv run src/train.py \
    trainer.policy.model.path=Qwen/Qwen3-14B \
    ...

# Use different model family entirely
uv run src/train.py \
    trainer.policy.model.path=meta-llama/Llama-3.1-8B \
    generator.engine_init_kwargs="{enable_auto_tool_choice:true,tool_call_parser:llama3}" \
    ...
```

**Option 2: Different LoRA config**

```bash
# Larger LoRA rank (more parameters, slower)
uv run src/train.py \
    trainer.policy.model.lora.rank=128 \
    trainer.policy.model.lora.alpha=1024 \
    ...

# Disable LoRA (full fine-tuning - requires more memory)
uv run src/train.py \
    trainer.policy.model.lora.enabled=false \
    ...
```

### Adjusting Training Hyperparameters

**Learning Rate:**
```bash
# Higher LR for faster convergence (risk of instability)
trainer.policy.optimizer_config.lr=5e-6

# Lower LR for more stable training
trainer.policy.optimizer_config.lr=1e-7
```

**Batch Size:**
```bash
# Larger batch (more stable gradients, more memory)
trainer.train_batch_size=8
trainer.policy_mini_batch_size=8

# Smaller batch (less memory, more updates)
trainer.train_batch_size=2
trainer.policy_mini_batch_size=2
```

**Generation Settings:**
```bash
# More rollouts per problem (better exploration, slower)
generator.n_samples_per_prompt=8

# Longer context window
generator.max_input_length=32000
trainer.max_prompt_length=8192

# More turns (longer search, more thorough)
generator.max_turns=30
```

### Debugging Tips

**1. Test Single Instance:**

```bash
# Run agent on one problem
uv run tests/test_single_prompt.py \
    --instance-id django__django-12345 \
    --model willcb/Qwen3-8B
```

**2. Check Tool Execution:**

```bash
# Test bash tool directly
cd /path/to/repo
rg "pattern" -t py  # Verify command works

# Check tool output parsing
uv run tests/test_messages.py
```

**3. Validate Rewards:**

```bash
# Test reward computation
uv run tests/test_file_localization.py

# Check reward values
python -c "
from src.rewards import get_reward_function
reward_fn = get_reward_function('tool_use_reward')
print(reward_fn(messages=[...]))
"
```

**4. Profile Performance:**

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Profile inference
uv run vllm benchmark \
    --model willcb/Qwen3-8B \
    --input-len 2048 \
    --output-len 512
```

**5. Inspect Trajectories:**

```bash
# View saved trajectories
cat ckpts/qwen3-8b/trajectories/step_10/train/django__django-12345_0.json | jq .

# Count successful episodes
grep -l '"total_reward"' ckpts/qwen3-8b/trajectories/step_10/train/*.json | wc -l
```

---

## 15. Troubleshooting

### Common Issues

**Issue: Out of Memory (OOM)**

```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce GPU memory utilization
generator.gpu_memory_utilization=0.4  # Default: 0.6

# Smaller batch size
trainer.train_batch_size=2
trainer.micro_train_batch_size_per_gpu=1

# Enable gradient checkpointing
trainer.policy.model.gradient_checkpointing=true

# Smaller LoRA rank
trainer.policy.model.lora.rank=32  # Default: 64
```

**Issue: vLLM Tool Calling Errors**

```
Error: Tool call parsing failed
```

**Solutions:**
```bash
# Verify tool call parser matches model
generator.engine_init_kwargs="{tool_call_parser:hermes}"  # For Qwen3

# Check model supports tool calling
vllm serve willcb/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes

# Test with simple tool call
curl http://localhost:8000/v1/chat/completions -d '{"tools": [...]}'
```

**Issue: Slow Training**

**Solutions:**
```bash
# Enable async training
run_async_trainer=true

# More inference engines
generator.num_inference_engines=4  # Default: 2

# Fewer rollouts per problem
generator.n_samples_per_prompt=2  # Default: 4

# Smaller eval batch (faster eval)
trainer.eval_batch_size=50  # Default: 100
```

**Issue: Low Rewards**

**Debugging:**
```bash
# Check individual reward components
cat ckpts/.../trajectories/step_10/train/*.json | jq '.reward_dict'

# Verify ground truth extraction
python -c "
from src.utils.parse_patch import parse_patch
patch = '''diff --git a/file.py ...'''
print(parse_patch(patch))
"

# Test agent on easy instances
uv run tests/test_single_prompt.py --instance-id <easy_instance>
```

**Solutions:**
```bash
# Adjust reward composition
# Edit configs/rewards/tool_use.yaml
reward:
  - fn: multilevel_localization_f1_reward
  # Remove other rewards to focus on F1

# Simplify task (fewer files)
# Filter dataset to single-file changes

# Better system prompt
# Edit src/prompts/system_prompt.py with clearer instructions
```

### FAQ

**Q: How long does training take?**

A: On 2x A100 GPUs:
- 150 steps with batch_size=4: ~6-8 hours
- Full 20 epochs: ~48-72 hours

**Q: How much does it cost?**

A: Approximate costs (USD):
- Cloud GPUs (2x A100): $6-10/hour
- Full training run: $300-700
- Dataset storage: ~$10/month

**Q: Can I train on smaller GPUs?**

A: Yes, with modifications:
- **Single A100 40GB**: Reduce batch_size to 2, LoRA rank to 32
- **V100 32GB**: Use smaller model (Qwen3-0.6B), batch_size=1
- **Consumer GPUs (RTX 4090)**: Use QLoRA (4-bit quantization)

**Q: How do I resume training?**

```bash
uv run src/train.py \
    trainer.resume_mode=latest \
    trainer.ckpt_path=ckpts/qwen3-8b
```

**Q: Can I use my own dataset?**

Yes! Format as Parquet with columns:
```python
{
    "instance_id": str,
    "repo": str,
    "base_commit": str,
    "problem_statement": str,
    "target": list[str],  # Ground truth files
}
```

Then:
```bash
uv run src/train.py \
    data.train_data="['my_data/train.parquet']" \
    data.val_data="['my_data/val.parquet']"
```

---

## Conclusion

You now have a comprehensive understanding of the **Agentic Code Search OSS** project:

- **What it does**: Trains specialized agents for code localization using RL
- **How it works**: PPO training on SWE-bench with OpenHands agents
- **Why it matters**: Faster, more accurate code search than general LLMs
- **How to use it**: Run experiments, modify configs, add custom rewards
- **How to extend it**: Add tools, change models, tune hyperparameters

### Next Steps

1. **Run your first experiment** - Start with `scripts/run_training.sh`
2. **Analyze results** - Check W&B logs and trajectory files
3. **Experiment** - Try different rewards, models, or prompts
4. **Contribute** - Share findings in #agentic-code-search-oss Slack

### Resources

- **GitHub**: https://github.com/All-Hands-AI/agentic-code-search-oss
- **Slack**: #agentic-code-search-oss (All-Hands-AI workspace)
- **SWE-bench**: https://www.swebench.com/
- **OpenHands SDK**: https://github.com/OpenHands/software-agent-sdk
- **SkyRL**: https://github.com/NovaSky-AI/SkyRL
- **vLLM**: https://docs.vllm.ai/

**Happy experimenting!**
