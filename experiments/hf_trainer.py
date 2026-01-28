import torch
import transformers as tr
import datasets as hf_datasets
from typing import Dict, Any, Union
import glob
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

import models.config as config
from models.vision_language_model import VisionLanguageModel
from data.datasets import VQADataset, MMStarDataset
from data.collators import VQACollator, MMStarCollator
from data.processors import get_image_processor, get_tokenizer
from experiments.experiment import Experiment
from models.utils import check_multiple_choice_with_regex

class MMStarCallback(tr.TrainerCallback):
    def __init__(self, test_loader, tokenizer, device, use_grayscale=False):
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self.use_grayscale = use_grayscale

    def on_evaluate(self, args, state, control, model, **kwargs):
        # This calls the test_mmstar logic during evaluation
        print("Running MMStar Evaluation...")
        if model.training:
             model.eval()

        total_examples = 0
        correct_predictions = 0

        # Ensure model is on the right device
        model.to(self.device)

        with torch.no_grad():
            for batch in self.test_loader:
                image = batch['images'].to(self.device)
                if self.use_grayscale:
                    gray = image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114
                    image = gray.repeat(1, 3, 1, 1)
                image = (image - 0.5) / 0.5
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                correct_answer = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                gen = model.generate(input_ids, image, attention_mask)
                model_output = self.tokenizer.batch_decode(gen, skip_special_tokens=True)

                is_correct = check_multiple_choice_with_regex(model_output, correct_answer)

                total_examples += len(is_correct)
                if is_correct:
                    correct_predictions += sum(is_correct)

        accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        print(f"MMStar Accuracy: {accuracy:.4f}")

        # Log to trainer
        # Note: We can access the trainer instance via kwargs if needed, but standard logging is handled via return in compute_metrics usually.
        # However, since this is a callback, we can modify state or log specifically if we have access to the trainer.
        # But 'model' is passed.
        # Standard way in Callback allows logging?
        # Actually TrainerCallback doesn't return metrics to log directly to history easily without accessing trainer.
        # But we can print it for now as requested.

@dataclass
class HFTrainerExperiment(Experiment):
    """
    Experiment that uses HuggingFace Trainer to train the VisionLanguageModel.
    Uses a small subset of data for quick testing/experimentation on MPS.
    """
    batch_size: int = 2
    max_steps: int = 10
    learning_rate: float = 1e-4
    output_dir: str = "output/hf_trainer_experiment"

    def run(self):
        print("Starting HFTrainer Experiment...")

        # 1. Setup Configuration
        vlm_cfg = config.VLMConfig()
        # Use smaller values for lightweight test
        vlm_cfg.vit_model_type = 'google/siglip-base-patch16-224' # Default in new config
        vlm_cfg.lm_model_type = 'HuggingFaceTB/SmolLM2-135M' # Smallest LM
        vlm_cfg.lm_tokenizer = 'HuggingFaceTB/cosmo2-tokenizer'

        train_cfg = config.TrainConfig()

        # 2. Prepare Data (Tiny Subset)
        print("Loading dataset subset...")
        # Use 'ai2d' from the_cauldron as it's a standard VQA task compatible with the pipeline
        ds_stream = hf_datasets.load_dataset(
            "HuggingFaceM4/the_cauldron",
            name="ai2d",
            split="train",
            streaming=True
        )

        # Take a small number of samples
        data_list = list(ds_stream.take(20))

        # Convert back to a HF Dataset object
        try:
            tiny_dataset = hf_datasets.Dataset.from_list(data_list)
        except Exception as e:
            print(f"Failed to create HF Dataset from list: {e}. Using list directly.")
            tiny_dataset = data_list

        print("Loading MMStar dataset for eval...")
        # Using the actual MMStar dataset for "test" part
        mmstar_ds = hf_datasets.load_dataset(train_cfg.test_dataset_path, split="val")
        # Take a tiny subset for speed
        mmstar_tiny = mmstar_ds.select(range(10))

        # 3. Setup Processors
        print("Setting up processors...")
        # New signatures from processors.py
        image_processor = get_image_processor(vlm_cfg.vit_img_size)
        tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

        # 4. Initialize Datasets
        print("Initializing Datasets...")
        vqa_dataset = VQADataset(
            tiny_dataset,
            tokenizer,
            image_processor
        )
        train_dataset = vqa_dataset

        mmstar_dataset = MMStarDataset(
            mmstar_tiny,
            tokenizer,
            image_processor
        )

        # 5. Initialize Model
        print("Initializing Model...")
        model = VisionLanguageModel(vlm_cfg, load_backbone=True)

        # 6. Initialize Collator
        collate_fn = VQACollator(tokenizer, vlm_cfg.lm_max_length)
        mmstar_collate_fn = MMStarCollator(tokenizer)

        # 7. Setup Training Arguments
        print("Setting up Trainer...")
        training_args = tr.TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            max_steps=self.max_steps,
            logging_steps=1,
            learning_rate=self.learning_rate,
            save_strategy="no",
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            label_names=["labels"],
            eval_strategy="steps", # Evaluate every 'eval_steps'
            eval_steps=5, # Run eval twice during the 10 steps
            per_device_eval_batch_size=2
        )

        # 8. Setup Evaluation Objects
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        test_loader = DataLoader(
            mmstar_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=mmstar_collate_fn
        )

        mmstar_callback = MMStarCallback(
            test_loader=test_loader,
            tokenizer=tokenizer,
            device=device
        )

        # 9. Run Trainer
        trainer = tr.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset, # Required for eval_strategy, even if we use callback
            data_collator=collate_fn,
            callbacks=[mmstar_callback]
        )

        print("Starting Training...")
        trainer.train()
        print("Training complete!")
