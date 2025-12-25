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

