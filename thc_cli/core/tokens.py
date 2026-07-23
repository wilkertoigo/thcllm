def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken
    except ImportError:
        return max(0, len(text) // 4)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


def count_tokens_messages(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += count_tokens(content)
    return total
