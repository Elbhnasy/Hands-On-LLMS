from helpers import get_settings
from openai import OpenAI
import os 

setting = get_settings()

client = OpenAI(
    api_key = setting.OPENAI_API_KEY
)

llm_model = setting.LLM_MODEL

def get_completion(prompt: str,
                    model: str = llm_model,
                    temperature: float = 0.0,
                    max_tokens: int = 500) -> str:
    """
    Get the completion from the OpenAI API.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message['content']

