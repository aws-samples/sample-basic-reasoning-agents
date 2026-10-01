"""
Basic Reasoning Agent - Simple Agent

A stateless agent that performs single-step reasoning using Amazon Bedrock.
This is the foundation for all other agent patterns.

Learning objectives:
- Understand the agent loop
- Practice prompt engineering
- See how agents transform queries into responses
- Watch the model stream tokens in real time
"""

import time
from shared.model import get_model
from shared.input_utils import get_multiline_input
from shared.streaming import StreamingCallbackHandler
from strands import Agent

# Default: Bedrock (uncomment to pin a specific model)
# from strands.models.bedrock import BedrockModel
#
# model = BedrockModel(
#     model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0"  # any Bedrock inference profile
# )

# Alternative: OpenAI (key read from the environment, never written in code)
# import os
# from strands.models.openai import OpenAIModel
#
# model = OpenAIModel(
#     client_args={"api_key": os.environ["OPENAI_API_KEY"]},
#     model_id="gpt-4o"
# )

# Streaming callback handler prints tokens as the model emits them so you can
# watch the agent "think". Pass callback_handler=None instead to suppress it.
stream_handler = StreamingCallbackHandler()

agent = Agent(
    model=get_model(),
    system_prompt="""You are a helpful assistant that provides clear, 
    concise answers. Focus on being accurate and informative.""",
    callback_handler=stream_handler,
)


def main():
    """Run the basic reasoning agent interactively."""
    print("Basic Reasoning Agent")
    print("=" * 40)
    print("Type 'quit' to exit")
    print("Tip: You can paste multi-line prompts!\n")

    while True:
        user_input = get_multiline_input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Reset per-turn streaming state, print the prefix, then let the
        # callback handler stream the response in place.
        stream_handler.reset()
        print("\nAgent: ", end="", flush=True)

        start_time = time.time()
        agent(user_input)
        elapsed = time.time() - start_time

        print(f"\n({elapsed:.1f}s)\n")


if __name__ == "__main__":
    main()
