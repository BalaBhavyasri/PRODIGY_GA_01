import os
import sys
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
class TextDataset(Dataset):
    """
    Custom PyTorch Dataset that loads a text file, tokenizes it, 
    and chunks it into sequences of block_size with overlap (stride)
    to maximize training samples from small datasets.
    """
    def __init__(self, tokenizer, file_path, block_size=128):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        tokenized_text = tokenizer.encode(text)
        self.examples = []
        
        # If the token count is smaller than block_size, pad to block_size
        if len(tokenized_text) < block_size:
            pad_len = block_size - len(tokenized_text)
            padded = tokenized_text + [tokenizer.eos_token_id] * pad_len
            self.examples.append(padded)
        else:
            # Overlapping chunking (50% overlap)
            stride = block_size // 2
            for i in range(0, len(tokenized_text) - block_size + 1, stride):
                self.examples.append(tokenized_text[i:i + block_size])
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        item = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": item, "labels": item}
class StreamlitProgressCallback(TrainerCallback):
    """
    Hugging Face Trainer Callback to write progress metadata and
    loss history to local JSON files for real-time visualization in Streamlit.
    """
    def __init__(self, progress_filepath):
        self.progress_filepath = progress_filepath
        dir_name = os.path.dirname(progress_filepath) or ""
        base_name = os.path.basename(progress_filepath)
        name, ext = os.path.splitext(base_name)
        self.history_filepath = os.path.join(dir_name, f"{name}_history{ext}")
        self.last_loss = None
        
        # Clear/initialize history file
        with open(self.history_filepath, "w") as f:
            json.dump([], f)
    def on_step_end(self, args, state, control, **kwargs):
        # Update current training progress metrics
        data = {
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": state.epoch,
            "loss": self.last_loss,
            "percentage": (state.global_step / state.max_steps) * 100 if state.max_steps > 0 else 0,
            "status": "training"
        }
        with open(self.progress_filepath, "w") as f:
            json.dump(data, f)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_loss = logs["loss"]
            
            # Read, append, and save loss history
            history = []
            if os.path.exists(self.history_filepath):
                try:
                    with open(self.history_filepath, "r") as f:
                        history = json.load(f)
                except Exception:
                    history = []
            
            history.append({
                "step": state.global_step,
                "loss": logs["loss"],
                "learning_rate": logs.get("learning_rate", 0.0)
            })
            
            with open(self.history_filepath, "w") as f:
                json.dump(history, f)
    def on_train_end(self, args, state, control, **kwargs):
        # Mark training as completed
        data = {
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": state.epoch,
            "loss": self.last_loss,
            "percentage": 100.0,
            "status": "completed"
        }
        with open(self.progress_filepath, "w") as f:
            json.dump(data, f)
def run_fine_tuning(args):
    # Ensure progress file is written immediately to signal start
    start_data = {
        "step": 0,
        "max_steps": 100, # default placeholder
        "epoch": 0.0,
        "loss": None,
        "percentage": 0.0,
        "status": "loading_model"
    }
    with open(args.progress_file, "w") as f:
        json.dump(start_data, f)
        
    try:
        print(f"Loading tokenizer & model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # GPT-2 needs a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(args.model)
        
        print(f"Loading and processing dataset: {args.dataset}")
        dataset = TextDataset(tokenizer, args.dataset, block_size=args.block_size)
        print(f"Dataset contains {len(dataset)} examples.")
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=1, # Log progress on every step
            save_strategy="no", # Do not save intermediate large checkpoints to save disk/time
            max_steps=args.max_steps,
            fp16=False, # CPUs don't support standard FP16
            use_cpu=True, # Explicitly run on CPU
            report_to="none" # Disable wandb/tensorboard logging
        )
        
        # Progress callback
        progress_callback = StreamlitProgressCallback(args.progress_file)
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[progress_callback]
        )
        
        # Run fine-tuning
        print("Starting training...")
        trainer.train()
        
        # Save fine-tuned model and tokenizer
        print(f"Saving fine-tuned model to: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Training successfully finished!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training failed with error: {str(e)}")
        error_data = {
            "status": "failed",
            "error": str(e)
        }
        with open(args.progress_file, "w") as f:
            json.dump(error_data, f)
        sys.exit(1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 CPU Fine-Tuning Backend")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Pretrained model identifier")
    parser.add_argument("--dataset", type=str, required=True, help="Path to text dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/latest", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to train (overrides epochs if > 0)")
    parser.add_argument("--block_size", type=int, default=128, help="Token block size")
    parser.add_argument("--progress_file", type=str, default="progress.json", help="Path to write progress JSON")
    
    args = parser.parse_args()
    run_fine_tuning(args)