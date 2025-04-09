# test_lmstudio.py
import requests

def test_lmstudio_connection():
    try:
        # Test models endpoint
        models_response = requests.get("http://127.0.0.1:1234/v1/models")
        print("Available models:", models_response.json())
        
        # Test chat completion
        chat_url = "http://127.0.0.1:1234/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "qwen2.5-7b-instruct-1m",  # Use an available model name
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        
        chat_response = requests.post(chat_url, json=payload, headers=headers)
        print("Chat response:", chat_response.json())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_lmstudio_connection()