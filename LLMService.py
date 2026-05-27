from ollama import AsyncClient
import asyncio

class LLMService:
    def __init__(self, model, hostAddress="127.0.0.1", max_history_pairs=10):
        self.model = model
        self.ollama_client = AsyncClient(host=hostAddress, timeout=None)
        self.conversation_history = [] 
        self.max_history_pairs = max_history_pairs

    async def initialize(self):
        print("Ładowanie modelu LLM ...")
        await self.ollama_client.generate(model=self.model, keep_alive="1h")
        print("Ok")

    async def think(self, data, system_prompt=None):
        if system_prompt and not self.conversation_history:
            self.conversation_history.append({'role': 'system', 'content': system_prompt})
            
        self.conversation_history.append({'role': 'user', 'content': data})
        max_messages = (self.max_history_pairs * 2) + (1 if system_prompt else 0)
        
        while len(self.conversation_history) > max_messages:
            if system_prompt:
                self.conversation_history.pop(1)
            else:
                self.conversation_history.pop(0)

        response = await self.ollama_client.chat(
            model=self.model,
            messages=self.conversation_history,
            keep_alive="1h",
            options={
                "temperature": 0.7,
                "top_p": 0.95,
                "num_ctx": 4096, 
                "num_predict": 250,
            }
        )

        content = remove_think_tags(response.message.content)
        self.conversation_history.append({'role': 'assistant', 'content': content})
        
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