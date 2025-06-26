from dataclasses import dataclass

@dataclass(frozen=True)
class GeneratorConfig:
    vec_size: int
    iterations: int
    output_path: str
    llm_model: str = "google/gemini-2.0-flash-exp:free"
    temperature: float = 0.7
    base_url: str = "https://openrouter.ai/api/v1"
    max_retries: int = 2
