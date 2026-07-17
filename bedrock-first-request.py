import anthropic

client = anthropic.Anthropic()

candidates = [
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-haiku-20241022",
    "claude-haiku-4-5",
]

for model in candidates:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello"}],
        )
        print(f"✅ WORKS: {model}")
        print(f"   → {response.content[0].text}\n")
    except Exception as e:
        print(f"❌ FAILS: {model}")
        print(f"   → {type(e).__name__}: {str(e)[:100]}\n")