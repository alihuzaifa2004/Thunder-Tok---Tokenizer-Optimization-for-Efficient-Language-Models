import torch
from transformers import AutoModelForCausalLM

class LLaMAWrapper:
    def __init__(self, model_name, vocab_size, token=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32, 
            device_map="auto",
            token=token
        )
        
    def generate(self, input_ids, max_new_tokens=50):
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # 1. MANUALLY CREATE ATTENTION MASK
        # This tells the model exactly which Korean tokens to focus on
        attention_mask = torch.ones(input_ids.shape, device=device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, # <-- Add this line
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9, # Helps with better language flow
                pad_token_id=self.model.config.eos_token_id 
            )
        return output