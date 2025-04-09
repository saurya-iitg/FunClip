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
 

def g4f_openai_call(model, user_content, system_content=None):
    try:
        # Initialize the local model
        local_model = llm(model)
        
        # Prepare the messages for the local model
        if system_content is not None and len(system_content.strip()):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        else:
            messages = [
                {"role": "user", "content": user_content}
            ]
        
        # Call the local model with the prepared messages
        result = local_model.chat(messages)
        
        # Return the content of the response
        return result.message.content
    except Exception as e:
        logging.error(f"Failed to load or call the model: {model}. Error: {e}")
        return f"Error: {e}"