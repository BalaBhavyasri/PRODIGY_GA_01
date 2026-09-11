import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
def load_model_and_tokenizer(model_path_or_name):
    """
    Loads causal LM model and tokenizer from local directory or Hugging Face Hub.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
    model.eval() # Put model in evaluation mode
    return model, tokenizer
def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    num_return_sequences=1
):
    """
    Generates continuation text based on the provided prompt using top-k, top-p, 
    and temperature sampling parameters.
    """
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = len(inputs["input_ids"][0])
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_length=max_length + prompt_len, # limit generation to max_length tokens *after* prompt
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_texts = []
    for seq in output_sequences:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        generated_texts.append(text)
        
    return generated_texts