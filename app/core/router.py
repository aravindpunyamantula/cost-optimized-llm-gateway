import tiktoken
from app.config import config

def estimate_tokens(prompt : str) -> int :
    """
    Estimate the number of tokens in a prompt using tiktoken """

    try:
        encoding = tiktoken.encoding_for_model(config["models"]["high_tier"])
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(prompt))

def contains_keyword(prompt: str, keywords: list[str]) -> bool:
    """
    Check if the prompt contains any of the specified keywords.
    """
    prompt_lower = prompt.lower()
    return any(keyword.lower() in prompt_lower for keyword in keywords)


def route_prompt(prompt: str) -> str:
    """ 
    Select an LLM model according to the confirgued routing rules.
    Rules are evaluated in order 
    The first matching rule wins
    """

    token_count = estimate_tokens(prompt)

    for rule in config["routing_rules"]:
        condition = rule["condition"]

        if condition == "contains_keyword":
            if contains_keyword(prompt, rule["keywords"]):
                return config["models"][rule["target_tier"]]
            
        elif condition == "max_tokens_exceeded":
            if token_count > rule["threshold"]:
                return config["models"][rule["target_tier"]]

        elif condition == "catch_all":
            return config["models"][rule["target_tier"]]
    raise ValueError("No routing rule matched the prompt")