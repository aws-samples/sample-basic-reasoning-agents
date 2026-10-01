"""
Basic Reasoning Agent - RAG Agent

Extends the basic agent with retrieval-augmented generation (RAG).
Retrieves relevant context before generating responses.

Learning objectives:
- Understand how RAG extends agent capabilities
- Practice context injection into prompts
- See how external knowledge improves responses
- Watch the model stream its grounded answer in real time
"""

import time
from shared.model import get_model
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent

# Sample knowledge base (in production, use vector DB like OpenSearch)
KNOWLEDGE_BASE = {
    "agentcore": "Amazon Bedrock AgentCore is a platform to build, deploy, and operate AI agents at scale.",
    "strands": "Strands Agents SDK is a framework for building AI agents with tools, memory, and multi-agent patterns.",
    "bedrock": "Amazon Bedrock is a fully managed service for foundation models from AI21, Anthropic, Meta, and more.",
    "tim": "Tim is a fun loving guy with great hair.",
    "bits": "bits are the best thing since sliced bread."
}


def retrieve_context(query: str) -> str:
    """Simple keyword-based retrieval (replace with vector search in production)."""
    query_lower = query.lower()
    relevant = []
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            relevant.append(value)
    return "\n".join(relevant) if relevant else ""


# Streaming callback handler prints tokens as the model emits them so you can
# watch the agent "think". Pass callback_handler=None instead to suppress it.
stream_handler = StreamingCallbackHandler()


def create_rag_agent():
    """Create an agent that uses retrieved context."""
    return Agent(
        model=get_model(),
        system_prompt="""You are a knowledgeable assistant about AWS AI services 
        and some specific topics around Tim and Computers.
        Use the provided context to answer questions accurately and efficiently.""",
        callback_handler=stream_handler,
    )


def main():
    """Run the RAG agent interactively."""
    agent = create_rag_agent()

    print("RAG Agent - Knowledge-Enhanced Responses")
    print("=" * 40)
    print("This agent retrieves context from a knowledge base before responding.")
    print("Type 'quit' to exit")
    print("Tip: You can paste multi-line prompts!\n")

    print("Example prompts to try:")
    print("  - What is Amazon Bedrock AgentCore?")
    print("  - Tell me about the Strands Agents SDK")
    print("  - How does Amazon Bedrock work?")
    print("  - Who is Tim?\n")

    while True:
        user_input = get_multiline_input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            # Retrieve relevant context first. Print whether a hit was found
            # so the viewer can see retrieval happening before generation.
            context = retrieve_context(user_input)
            if context:
                print(f"  [retrieval] matched {len(context.splitlines())} entry(ies)")
                augmented_prompt = (
                    f"Context:\n{context}\n\nQuestion: {user_input}"
                )
            else:
                print("  [retrieval] no match — answering from model's general knowledge")
                augmented_prompt = user_input

            # Reset streaming state, print prefix, then stream the response.
            stream_handler.reset()
            print("\nAgent: ", end="", flush=True)

            start_time = time.time()
            agent(augmented_prompt)
            elapsed = time.time() - start_time

            print(f"\n({elapsed:.1f}s)\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
