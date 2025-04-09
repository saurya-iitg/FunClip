import logging
import requests
from typing import Optional
from g4f.client import Client
from lmstudio import llm  # Import the local model library

if __name__ == '__main__':
    from llm.demo_prompt import demo_prompt
    client = Client()
    response = client.chat.completions.create(
        model="Qwen2.5 7B Instruct 1M",
        messages=[{"role": "user", "content": "你好你的名字是什么"}],
    )
    print(response.choices[0].message.content)
 

def g4f_openai_call(model: str, user_content: str, system_content: Optional[str] = None) -> str:
    """Call LMStudio local server API"""
    try:
        # LMStudio server URL
        url = "http://127.0.0.1:1234/v1/chat/completions"
        
        # Prepare messages
        messages = []
        if system_content and system_content.strip():
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        
        # API request payload
        payload = {
            "messages": messages,
            "model": model,
            "temperature": 0.7,
            "max_tokens": 100000
        }
        
        # Make API call
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Extract and return response content
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        logging.error(f"Failed to call LMStudio API: {str(e)}")
        return f"Error: {str(e)}"