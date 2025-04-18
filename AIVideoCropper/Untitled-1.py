    combined_content = user_content + '\n' + srt_text
    
    # Use local LM endpoint if model is present in available models
    if model.lower() in [m.lower() for m in lmstudio_models]:
        actual_model = next(m for m in lmstudio_models if m.lower() == model.lower())
        logging.info(f"Using LMStudio local model: {actual_model}")
        response = g4f_openai_call(actual_model, combined_content, system_content)
        logging.info(f"Local model response: {response}")
        return response
    elif model.startswith('gpt') or model.startswith('moonshot'):
        logging.info("Using OpenAI API call")
        response = openai_call(apikey, model, system_content, combined_content)
        logging.info(f"OpenAI model response: {response}")
        return response
    else:
        error_msg = f"Unsupported model: {model}. Available models: {lmstudio_models}"
        logging.error(error_msg)
        return error_msg
except Exception as e:
    error_msg = f"LLM inference error: {str(e)}"
    logging.error(error_msg)
    return error_msg