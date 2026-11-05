import os
from anthropic import Anthropic

# The SDK automatically looks for the ANTHROPIC_API_KEY environment variable
client = Anthropic()

# Send a message to Claude
response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a one-sentence greeting to a developer."}
    ]
)

# Print Claude's reply
print(response.content[0].text)
