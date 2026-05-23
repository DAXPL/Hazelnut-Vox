from ollama import Client
import asyncio

class LLMService:
    def __init__(self, model, hostAddress="127.0.0.1"):
        print("Ładowanie modelu LLM ...")
        self.model = model
        self.ollama_client = Client(host=hostAddress)
        self.ollama_client.generate(model=self.model, keep_alive="1h")
        print("Ok")

    def think(self, data, system_prompt=None):
        messages = []

        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
            
        messages.append({'role': 'user', 'content': data})

        response = self.ollama_client.chat(
            model=self.model,
            messages=messages,
            keep_alive="1h",
            options={
                "temperature": 0.9,
                "top_p": 0.95,
                "num_ctx": 2048,
                "num_predict": 250,
            }
        )

        content = remove_think_tags(response.message.content)
        return content
    
def remove_think_tags(text: str) -> str:
    result = ""
    start = 0
    while True:
        open_idx = text.find("<think>", start)
        if open_idx == -1:
            result += text[start:]
            break
        result += text[start:open_idx]
        close_idx = text.find("</think>", open_idx)
        if close_idx == -1:
            break
        start = close_idx + len("</think>")
    return result.strip()