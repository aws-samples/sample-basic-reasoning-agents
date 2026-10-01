# Basic Reasoning Agents

The simplest form of agentic AI: a query goes in, the LLM reasons over it, and a response comes back. No tools, no memory, no orchestration.

This sample covers building an agent with the [Strands Agents SDK](https://strandsagents.com/) and is based off of the [AWS Prescriptive Guidance - Basic Reasoning Agents pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/basic-reasoning-agents.html).

## Table of Contents

- [Quick Start](#quick-start)
- [Simple Agent](#simple-agent)
  - [How It Works](#how-it-works)
  - [Creating the Agent](#creating-the-agent)
  - [Invoking the Agent](#invoking-the-agent)
  - [Choosing a Model](#choosing-a-model)
- [RAG Agent](#rag-agent)
  - [Building the Knowledge Base](#building-the-knowledge-base)
  - [Retrieving and Injecting Context](#retrieving-and-injecting-context)
- [Error Handling](#error-handling)
- [AWS Implementation Patterns](#aws-implementation-patterns)
- [Reference](#reference)

## Quick Start

**Prerequisites:** 
- Python 3.10+
- An AWS account with Amazon Bedrock access
- AWS credentials configured (`aws configure`) with permission to invoke models on Bedrock


```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Point the sample at your AWS profile and region (loaded by shared/model.py)
cp .env.example .env
# Edit .env: set AWS_PROFILE and AWS_REGION. Optionally pin a model with STRANDS_MODEL_ID.

# Run the simple agent
python simple_agent.py

# Run the RAG agent
python rag_agent.py
```

**Try these exercises:**
1. **Reshape the agent with the system prompt.** Swap in a technical-explainer, code-reviewer, or customer-service persona and observe how the same model behaves differently.
2. **Swap the model.** Point the agent at a faster, cheaper model (e.g., Claude Haiku) and compare latency and answer quality.
3. **Grow the knowledge base.** Add entries to `KNOWLEDGE_BASE` and confirm the agent uses retrieved context for those topics while falling back to general knowledge otherwise.
4. **Add the error handling.** Wrap the invocation in the try/except block above and force a failure (e.g., a very long prompt) to see graceful recovery.

---

## Simple Agent

### How It Works

1. **Receives input**: User or system submits a query or instruction
2. **Invokes the LLM**: Agent transforms the query into a structured prompt and sends it to the LLM
3. **Returns response**: Generated output is formatted and returned to the user

[Strands Agents](https://strandsagents.com/) is an open-source SDK developed by AWS for building production-ready AI agents. (Comparable frameworks include LangChain/LangGraph, CrewAI, and Pydantic AI.) Import it with:

<img src="images/basic-reasoning-agents.png" width="600" alt="Diagram of a basic reasoning agent: a query flows into an LLM and a response flows back out." />


```python
from strands import Agent
```

### Creating the Agent

An agent needs two things to start: a **system prompt** that sets its role and behavior, and a **callback handler** that controls how execution events are surfaced.

#### System Prompt
The [system prompt](https://strandsagents.com/docs/user-guide/concepts/agents/prompts/) is your primary tool for customization. It's a persistent instruction that gets sent with every user message, so it shapes the agent's voice, scope, and constraints across the entire conversation. Think of it as a job description: it tells the model who it is, what it should focus on, what it should avoid, and how to format its responses. A vague prompt like "You are a helpful assistant" gives the model wide latitude — fine for general Q&A, but unpredictable. A specific prompt like "You are a customer service agent for a cloud storage product. Always confirm the user's plan tier before suggesting features. Never quote pricing — direct billing questions to support" produces consistent, scoped behavior. Good system prompts tend to define the role, the domain, the tone, any guardrails (what not to do), and the desired output format.

#### Callback Handler
[Callback handlers](https://strandsagents.com/docs/user-guide/concepts/streaming/callback-handlers/) let you intercept execution events for real-time monitoring, custom output formatting, or integration with external systems. By default, Strands uses `PrintingCallbackHandler`, which streams [agent events](https://strandsagents.com/docs/user-guide/concepts/streaming/#event-types) (model tokens, tool calls, lifecycle events) to stdout as the agent runs. Setting `callback_handler=None` disables that streaming so you only see the final response. This can be useful when you want quick clean output.

```python
agent = Agent(
    system_prompt="""You are a helpful assistant that provides clear, 
    concise answers. Focus on being accurate and informative.""",
    callback_handler=None
)
```

### Invoking the Agent

Invoking the agent is a single call:

```python
response = agent("What are the three laws of thermodynamics?")
```

That's it. Everything below is just a `while` loop so you can chat with it interactively. The only Strands-specific line is `agent(user_input)`.

```python
def main():
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        response = agent(user_input)  # ← the actual agent call
        print(f"\nAgent: {response}\n")
```

### Choosing a Model

Strands defaults to a Bedrock model provider. Check which model your agent is using:

```python
print(agent.model.config)
# eg. {'model_id': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'}
```

To use a different model, pass a `BedrockModel` instance to the agent:

```python
from strands.models.bedrock import BedrockModel

model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
agent = Agent(model=model, system_prompt="...", callback_handler=None)
```

Strands isn't limited to Bedrock. It has built-in support for Anthropic direct, OpenAI, Google Gemini, Ollama, LiteLLM, and others. See [Model Providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/) in the Strands docs for the full list and configuration examples for each.

> **Note:** Bedrock models released after mid-2024 require inference profile IDs (prefixed with `us.` or `global.`) rather than direct model IDs. List them with `aws bedrock list-inference-profiles`. See [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) for the full list.


---

## RAG Agent

A basic reasoning agent only knows what the model learned at training time. The Retrieval-Augmented Generation (RAG) variant retrieves relevant context *before* calling the LLM, grounding responses in your own data.

<img src="images/agent-rag.png" width="600" alt="Diagram of a RAG agent: a query is used to retrieve context from a knowledge base, the context is injected into the prompt, and the LLM generates a grounded response." />

### Building the Knowledge Base

For this sample, the knowledge base is a simple dictionary mapping topics to facts. This keeps the retrieval logic straight forward and simple, but larger systems would want to use a vector store such as the walkthrough describes in [Amazon Bedrock Knowledge Bases](https://aws.amazon.com/blogs/aws/knowledge-bases-now-delivers-fully-managed-rag-experience-in-amazon-bedrock/).

> **Note:** We explore the benefits of Vector Databases in more detail in the [companion blog](Building%20Basic%20Reasoning%20Agents%20with%20Amazon%20Bedrock%20and%20Strands%20SDK.md)

```python
KNOWLEDGE_BASE = {
    "agentcore": "Amazon Bedrock AgentCore is a platform to build, deploy, and operate AI agents at scale.",
    "strands": "Strands Agents SDK is a framework for building AI agents with tools, memory, and multi-agent patterns.",
    "bedrock": "Amazon Bedrock is a fully managed service for foundation models from AI21, Anthropic, Meta, and more.",
    "tim": "Tim is a fun loving guy with great hair.",
    "bits": "bits are the best thing since sliced bread."
}
```

### Retrieving and Injecting Context

If a knowledge-base context match appears in the query, its added to the LLM prompt:

```python
def retrieve_context(query: str) -> str:
    """Simple keyword-based retrieval for JSON map."""
    query_lower = query.lower()
    relevant = [value for key, value in KNOWLEDGE_BASE.items() if key in query_lower]
    return "\n".join(relevant)
```

> **Note:** When no context matches, the agent falls back to its general knowledge:

```python
context = retrieve_context(user_input)

if context:
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {user_input}"
else:
    augmented_prompt = user_input

response = agent(augmented_prompt)
```

The injection step, retrieve->augment->generate, is the entire RAG pattern

---

## Error Handling

Agent calls fail in agent-specific ways. The Strands SDK surfaces [typed exceptions](https://strandsagents.com/docs/api/python/strands.types.exceptions/) that deserve targeted handling rather than a generic catch-all:

```python
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
    MaxTokensReachedException,
)

try:
    response = agent(user_input)
    print(f"Agent: {response}")

except ContextWindowOverflowException:
    # Conversation exceeded the model's context window — reset or summarize
    print("Conversation too long — starting fresh.")
    agent = Agent(system_prompt=SYSTEM_PROMPT, callback_handler=None)

except ModelThrottledException:
    # Bedrock rate-limited the request — back off and retry
    import time
    print("Rate limited — retrying in a few seconds.")
    time.sleep(5)
    response = agent(user_input)

except MaxTokensReachedException:
    # Response was cut off before completing
    print("Response truncated — ask the agent to continue.")
```

Other exceptions worth knowing:

| Exception | When it fires | What to do |
|-----------|---------------|------------|
| `ToolProviderException` | A tool failed during execution | Log the error, check tool inputs |
| `MCPClientInitializationError` | MCP server connection failed | Verify the server is running |
| `ModelTimeoutException` (Bedrock) | Model took too long to respond | Retry with a simpler prompt |
| `ServiceQuotaExceededException` (Bedrock) | Hit account-level limits | Request a quota increase |

---

## AWS Implementation Patterns

| Pattern | Description | Reference |
|---------|-------------|-----------|
| Strands Agents SDK Quick Start | Build a basic reasoning agent with Strands using a model-driven approach on Amazon Bedrock | [Introducing Strands Agents, an Open Source AI Agents SDK](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/) |
| RAG with Bedrock Knowledge Bases | Implement retrieval-augmented generation using managed vector stores and automatic chunking | [Evaluate and improve performance of Amazon Bedrock Knowledge Bases](https://aws.amazon.com/blogs/machine-learning/evaluate-and-improve-performance-of-amazon-bedrock-knowledge-bases/) |
| Model-Driven Agent Architecture | Design agents that leverage LLM reasoning for planning and tool selection without hardcoded workflows | [Strands Agents and the Model-Driven Approach](https://aws.amazon.com/blogs/opensource/strands-agents-and-the-model-driven-approach/) |
| Contextual Retrieval for RAG | Enhance RAG accuracy using contextual retrieval techniques with Amazon Bedrock Knowledge Bases | [Contextual retrieval in Anthropic using Amazon Bedrock Knowledge Bases](https://aws.amazon.com/blogs/machine-learning/contextual-retrieval-in-anthropic-using-amazon-bedrock-knowledge-bases/) |



## Reference

- [Companion blog post: Building Basic Reasoning Agents](Building%20Basic%20Reasoning%20Agents%20with%20Amazon%20Bedrock%20and%20Strands%20SDK.md) — the concepts and history behind this sample
- [AWS Prescriptive Guidance - Basic Reasoning Agents](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/basic-reasoning-agents.html)
- [Strands Agents Documentation](https://strandsagents.com/)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)

### The series

This sample is one of eleven, one per pattern in the [AWS Prescriptive Guidance on agentic AI patterns](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/). Each has a hands-on sample repository and a companion blog post explaining the concepts.

| # | Pattern | Sample | Blog |
|---|---|---|---|
| 01 | Basic Reasoning Agents | this repository | [Building Basic Reasoning Agents with Amazon Bedrock and Strands SDK](Building%20Basic%20Reasoning%20Agents%20with%20Amazon%20Bedrock%20and%20Strands%20SDK.md) |
| 02 | Tool-Based Agents (Functions) | [sample-tool-based-agents-functions](https://github.com/aws-samples/sample-tool-based-agents-functions) | [Extending AI Agents with Custom Tools and Functions](https://github.com/aws-samples/sample-tool-based-agents-functions/blob/main/Extending%20AI%20Agents%20with%20Custom%20Tools%20and%20Functions.md) |
| 03 | Tool-Based Agents (Servers) | [sample-tool-based-agents-servers](https://github.com/aws-samples/sample-tool-based-agents-servers) | [Delegating Work: Tool Servers and the Model Context Protocol](https://github.com/aws-samples/sample-tool-based-agents-servers/blob/main/Delegating%20Work%20-%20Tool%20Servers%20and%20the%20Model%20Context%20Protocol.md) |
| 04 | Computer-Use Agents | [sample-computer-use-agents](https://github.com/aws-samples/sample-computer-use-agents) | [Agents That Use Computers: Browsers, Desktops, and the GUI Frontier](https://github.com/aws-samples/sample-computer-use-agents/blob/main/Agents%20That%20Use%20Computers%20-%20Browsers%2C%20Desktops%2C%20and%20the%20GUI%20Frontier.md) |
| 05 | Coding Agents | [sample-coding-agents](https://github.com/aws-samples/sample-coding-agents) | [Coding Agents: From Autocomplete to Autonomous Software Work](https://github.com/aws-samples/sample-coding-agents/blob/main/Coding%20Agents%20-%20From%20Autocomplete%20to%20Autonomous%20Software%20Work.md) |
| 06 | Speech and Voice Agents | [sample-speech-voice-agents](https://github.com/aws-samples/sample-speech-voice-agents) | [Giving Agents a Voice: Speech-to-Speech and the STT/TTS Pipeline](https://github.com/aws-samples/sample-speech-voice-agents/blob/main/Giving%20Agents%20a%20Voice%20-%20Speech-to-Speech%20and%20the%20STT-TTS%20Pipeline.md) |
| 07 | Workflow Orchestration Agents | [sample-workflow-orchestration-agent](https://github.com/aws-samples/sample-workflow-orchestration-agent) | [Orchestrating Agents: Sequential, Parallel, and Conditional Workflows](https://github.com/aws-samples/sample-workflow-orchestration-agent/blob/main/Orchestrating%20Agents%20-%20Sequential%2C%20Parallel%2C%20and%20Conditional%20Workflows.md) |
| 08 | Memory-Augmented Agents | [sample-memory-augmented-agents](https://github.com/aws-samples/sample-memory-augmented-agents) | [Agents That Remember: Context Windows, Summaries, and Persistent Sessions](https://github.com/aws-samples/sample-memory-augmented-agents/blob/main/Agents%20That%20Remember%20-%20Context%20Windows%2C%20Summaries%2C%20and%20Persistent%20Sessions.md) |
| 09 | Simulation and Test-Bed Agents | [sample-simulation-testbed-agents](https://github.com/aws-samples/sample-simulation-testbed-agents) | [Practice Worlds: Simulation and Test-Bed Agents](https://github.com/aws-samples/sample-simulation-testbed-agents/blob/main/Practice%20Worlds%20-%20Simulation%20and%20Test-Bed%20Agents.md) |
| 10 | Observer and Monitoring Agents | [sample-observer-monitoring-agents](https://github.com/aws-samples/sample-observer-monitoring-agents) | [Watching the Watched: Observer and Monitoring Agents](https://github.com/aws-samples/sample-observer-monitoring-agents/blob/main/Watching%20the%20Watched%20-%20Observer%20and%20Monitoring%20Agents.md) |
| 11 | Multi-Agent Collaboration | [sample-multi-agent-collaboration](https://github.com/aws-samples/sample-multi-agent-collaboration) | [When Multi-Agent Collaboration Earns Its Cost](https://github.com/aws-samples/sample-multi-agent-collaboration/blob/main/When%20Multi-Agent%20Collaboration%20Earns%20Its%20Cost.md) |

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
