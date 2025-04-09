import logging
import requests
from typing import Optional
from g4f.client import Client
from lmstudio import llm  # Import the local model library

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    from llm.demo_prompt import demo_prompt
    client = Client()
    response = client.chat.completions.create(
        model="qwen2.5 7b instruct 1m",
        messages=[{"role": "user", "content": "你好你的名字是什么"}],
    )
    print(response.choices[0].message.content)

def g4f_openai_call(model: str, user_content: str, system_content: Optional[str] = None) -> str:
    """Call LMStudio local server API"""
    try:
        url = "http://localhost:1234/v1/chat/completions"
        
        # Prepare messages according to LMStudio's format
        messages = []
        if system_content and system_content.strip():
            messages.append({
                "role": "system",
                "content": system_content.strip()
            })
        messages.append({
            "role": "user",
            "content": user_content.strip()
        })
        
        payload = {
            "model": model.lower(),  # Must be lowercase
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,  # -1 means no limit
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        logging.info(f"Sending request to LMStudio with payload: {payload}")
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"LMStudio API error: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}"
            
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return f"Error: {str(e)}"