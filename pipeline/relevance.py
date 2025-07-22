from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pathlib import Path

# Convert local Windows path to POSIX
model_path = Path(r"C:\Users\btriki003\Desktop\llmops_project\gemma-3-1b-it").as_posix()

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

gemma_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def is_relevant(text_chunk: str) -> bool:
    prompt = f"""
You are a language model trained to evaluate training data for finance-related models.

Given the following chunk, answer with 'Yes' if it's relevant to finance,
or 'No' otherwise.

Text:
\"\"\"{text_chunk}\"\"\"

Answer with only: Yes or No.
"""
    result = gemma_pipe(prompt, max_new_tokens=10, do_sample=False)
    output = result[0]['generated_text'].lower()
    return "yes" in output


