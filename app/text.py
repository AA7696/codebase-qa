import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "how", "what",
    "when", "where", "which", "who", "why", "of", "to", "in", "on", "for", "and", "or", "it",
    "its", "this", "that", "with", "by", "as", "at", "from", "can", "i", "you", "my",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words. SessionRedirectMixin -> sessionredirectmixin, session, redirect, mixin."""
    tokens = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ")
        parts = [p for p in spaced.lower().split() if p]
        joined = "".join(parts)
        if joined:
            tokens.append(joined)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens