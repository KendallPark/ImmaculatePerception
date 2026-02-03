import torch
import os
import transformers as tr
import wandb
from dataclasses import dataclass

import models.config as config
from models.vision_language_model import VisionLanguageModel
from data.datasets import ColorShapeDataset
from data.collators import VQACollator
from data.processors import get_image_processor, get_tokenizer
from data.color_shape import create_color_shape_dataset_dict
from experiments.basic_training import BasicTraining, VQATrainer
from experiments.experiment import Experiment

@dataclass
class ShapeColorSFT(BasicTraining):
    """
    Supervised Fine-Tuning on Synthetic Shape/Color Dataset

    Inherits from BasicTraining to reuse model loading and configuration logic.
    Overrides dataset loading to use synthetic shape/color generation.
    """
    # Override BasicTraining defaults
    wandb_project: str = "maryVLM_shape_color_sft_aug"
    run_name: str = "shape_color_sft_aug_10k_v1"

    # Training
    epochs: int = 3 # Increased from 3 to 5 to help learn EOS token
    batch_size: int = 32

    # In BasicTraining, we have lr_mp and lr_backbones.
    # We set both to 1e-4 for SFT (increased from 5e-5).
    lr_mp: float = 1e-4
    lr_backbones: float = 1e-4

    warmup_ratio: float = 0.1
    # ... (comments)

    max_grad_norm: float = 1.0
    eval_steps: int = 200 # More frequent eval
    save_steps: int = 1000 # Save less often
    logging_steps: int = 25

    # Model
    resume_from_vlm_checkpoint: bool = True
    vlm_checkpoint_path: str = "output/maryVLM_bs256_rgb_ms1_ds1/checkpoint-22000"

    # System
    train_dataloader_num_workers: int = 4
    val_dataloader_num_workers: int = 2

    # SFT Specific
    num_train_per_combo: int = 100 # Increased to ~66k examples
    num_val_per_combo: int = 1
    # img_size is vit_img_size in BasicTraining (default 224)
    # Use 'random' prompts to train model on "The color is", "The shape is", etc.
    prompt_style: str = 'random'

    def load_datasets(self, tokenizer, image_processor, vlm_cfg):
        """Create and return synthetic datasets."""
        print("\nCreating synthetic dataset...")
        ds_dict = create_color_shape_dataset_dict(
            num_train_per_combo=self.num_train_per_combo,
            num_val_per_combo=self.num_val_per_combo,
            img_size=self.vit_img_size,
            seed=self.data_seed # Use experiment's data_seed
        )

        print(f"Train examples: {len(ds_dict['train'])}")
        print(f"Validation examples: {len(ds_dict['validation'])}")

        print(f"Shuffling training dataset with seed {self.data_seed}...")
        # Explicitly shuffle to ensure batches are random, even if Trainer usually does it
        shuffled_train = ds_dict['train'].shuffle(seed=self.data_seed)

        print("Wrapping datasets...")
        train_dataset = ColorShapeDataset(
            shuffled_train,
            tokenizer,
            image_processor,
            prompt_style=self.prompt_style
        )

        print(f"Train examples: {len(ds_dict['train'])}")
        print(f"Validation examples: {len(ds_dict['validation'])}")

        print("Wrapping validation dataset...")
        val_dataset = ColorShapeDataset(
            ds_dict['validation'],
            tokenizer,
            image_processor,
            prompt_style=self.prompt_style
        )



        # We don't use MMStar for this fine-tuning
        return train_dataset, val_dataset, None

    def get_trainer(self, model, args, train_dataset, eval_dataset, collator, processing_class, mmstar_dataset=None):
        """Return standard Trainer for SFT."""

        print("\nInitializing Trainer (Standard)...")

        # Override specific args for SFT from our class fields
        args.warmup_ratio = self.warmup_ratio
        args.warmup_steps = 0 # Disable fixed steps
        args.max_grad_norm = self.max_grad_norm

        # Define Custom Callback for Text Generation
        class TextGenerationCallback(tr.TrainerCallback):
            def __init__(self, dataset, tokenizer, num_examples=5):
                self.dataset = dataset
                self.tokenizer = tokenizer
                self.num_examples = num_examples

            def on_evaluate(self, args, state, control, model, **kwargs):
                print(f"\n\n*** Generative Evaluation (Step {state.global_step}) ***")

                # Determine device from model
                device = next(model.parameters()).device
                model.eval()

                indices = range(min(len(self.dataset), self.num_examples))

                logs = []
                for i in indices:
                    item = self.dataset[i]
                    # Dataset returns preprocessed tensors
                    img_tensor = item['image'].unsqueeze(0).to(device)
                    prompt = item['text_data']
                    label = item['answer']

                    # Tokenize prompt manually since we aren't using collator here
                    if prompt:
                        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                    else:
                        # Use BOS token if prompt is empty
                        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
                        input_ids = torch.tensor([[bos_id]], device=device)

                    with torch.no_grad():
                        # Generate
                        gen_ids = model.generate(input_ids, img_tensor, max_new_tokens=30)
                        gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

                        # Clean up prompt from generation if it's there (sometimes decode keeps it)
                        # Usually gen_ids includes input_ids for decoder-only? No, generate usually returns full seq
                        # Let's simple-strip prompt if it persists
                        # Actually model.generate returns FULL sequence (prompt + new tokens) usually?
                        # It depends on model config, usually yes for CausalLM.
                        pass

                    print(f"Ex {i}: Prompt='{prompt}' | Label='{label.strip()}' | Pred='{gen_text.strip()}'")
                    logs.append(f"Ex {i}: {gen_text.strip()} (Ref: {label.strip()})")

                print("******************************************************\n")

                if wandb.run is not None:
                    # Log as a simple text table or alert
                    wandb.log({"eval_samples": wandb.Html("<br>".join(logs))})

        # Register Callback
        gen_callback = TextGenerationCallback(eval_dataset, processing_class, num_examples=5)

        # Use VQATrainer
        trainer = VQATrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            processing_class=processing_class,
            callbacks=[gen_callback] # Add our custom callback
        )

        return trainer


@dataclass
class ShapeColorSFTAllModels(Experiment):
    """
    Meta-experiment: Fine-tune all 5 analysis models on synthetic dataset.

    Creates instances of ShapeColorSFT for each model and runs them sequentially.
    """

    # Model configs from analysis
    model_configs = [
        {
            "name": r"MaryVLM$_{rgb,0}$ (rgb)",
            "checkpoint": "output/maryVLM_bs256_rgb_ms0_ds0/checkpoint-22000",
            "run_name": "sft_rgb0_aug_10k",
            "use_grayscale": False,
        },
        {
            "name": r"MaryVLM$_{rgb,1}$ (rgb)",
            "checkpoint": "output/maryVLM_bs256_rgb_ms1_ds1/checkpoint-22000",
            "run_name": "sft_rgb1_aug_10k",
            "use_grayscale": False,
        },
        {
            "name": r"MaryVLM$_{gray,0}$ (gray)",
            "checkpoint": "output/maryVLM_bs256_gray_ms0_ds0/checkpoint-22000",
            "run_name": "sft_gray0_gray_aug_10k",
            "use_grayscale": True,

        },
        {
            "name": r"MaryVLM$_{gray,0}$ (rgb)",
            "checkpoint": "output/maryVLM_bs256_gray_ms0_ds0/checkpoint-22000",
            "run_name": "sft_gray0_rgb_aug_10k",
            "use_grayscale": False,
        },
        {
            "name": r"MaryVLM$_{gray,0}^{cont}$ (rgb)",
            "checkpoint": "output/maryVLM_bs256_gray_ms0_ds0_22000_cont_rgb_1_lr_mp0.001_lr_backbone0.005/checkpoint-5000",
            "run_name": "sft_gray0_cont_rgb_aug_10k",
            "use_grayscale": False,
        },
    ]

    def run(self):
        """Run SFT for all models sequentially."""
        print("=" * 80)
        print("Fine-Tuning All Analysis Models on Synthetic Dataset")
        print(f"Total models: {len(self.model_configs)}")
        print("=" * 80)

        results = {}

        for i, model_cfg in enumerate(self.model_configs, 1):
            print(f"\n{'=' * 80}")
            print(f"Model {i}/{len(self.model_configs)}: {model_cfg['name']}")
            print(f"{'=' * 80}")

            # Create experiment instance for this model
            experiment = ShapeColorSFT(
                vlm_checkpoint_path=model_cfg['checkpoint'],
                run_name=model_cfg['run_name'],
                wandb_project="maryVLM_color_shape_sft",
                use_grayscale=model_cfg['use_grayscale'],
            )

            # Run training
            trainer = experiment.run()

            # Store results
            results[model_cfg['name']] = {
                'checkpoint': model_cfg['checkpoint'],
                'output_dir': f"output/{model_cfg['run_name']}",
            }

            # Clean up GPU memory
            del trainer
            torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print("All Models Fine-Tuned Successfully!")
        print("=" * 80)
        for name, result in results.items():
            print(f"{name}: {result['output_dir']}")

        return results
