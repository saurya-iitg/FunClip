#local_llm_studio.py
import lmstudio as lms

def call_local_model(
    model="C:/Users/saura/.cache/lm-studio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF",  # Replace with your loaded model in LM Studio
    user_content="如何做西红柿炖牛腩？", 
    system_content=None
):
    # Initialize the model
    llm_model = lms.llm(model)
    
    # Prepare the prompt with system content if provided
    if system_content is not None and len(system_content.strip()):
        # Use the system content as part of the prompt
        result = llm_model.chat([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ])
    else:
        # Just use the user content
        result = llm_model.chat([
            {"role": "user", "content": user_content}
        ])
    
    print(result)
    return result.message.content

if __name__ == '__main__':
    call_local_model()
