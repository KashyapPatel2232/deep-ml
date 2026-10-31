import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    out = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [token.strip() for token in out if token.strip()]
