import anthropic

client = anthropic.Anthropic(
    api_key="sk-ant-your-key-here"
)

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Hello Claude"}
    ]
)

print(response.content[0].text)