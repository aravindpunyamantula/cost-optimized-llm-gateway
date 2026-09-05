import re 

def preprocess_prompt(prompt: str) -> str:
    """
    Preprocess the prompt by removing extra whitespace and normalizing it.
    """
    prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")
    prompt = "".join(
        char for char in prompt
        if char.isprintable() or char == "\n"
    )
    prompt = "\n".join(
        line.rstrip()
        for line in prompt.split("\n")
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    
    return prompt.strip()