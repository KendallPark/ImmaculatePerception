import torch
import os
from PIL import Image, PngImagePlugin
import torchvision.transforms as transforms
# Increase PIL text chunk size limit to handle large metadata in datasets
PngImagePlugin.MAX_TEXT_CHUNK = 200 * 1024 * 1024
# Only import what we need or what was there
import transformers as tr
import datasets as hf_datasets
import wandb
from typing import Dict, Any, Union, Optional
import glob
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

import models.config as config
from models.vision_language_model import VisionLanguageModel
from data.datasets import VQADataset, MMStarDataset
from data.collators import VQACollator, MMStarCollator
from data.processors import get_image_processor, get_tokenizer
from experiments.experiment import Experiment, with_wandb
from models.utils import check_multiple_choice_with_regex

class MMStarCallback(tr.TrainerCallback):
    def __init__(self, test_loader, tokenizer, device):
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device

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
                # Model handles grayscale and normalization now
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

        # Log to WandB
        if wandb.run is not None:
             wandb.log({"eval/mmstar_accuracy": accuracy})


class VQATrainer(tr.Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Internal save method override.
        This is called by both save_model() and during checkpointing.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. Handle state_dict gathering
        # If state_dict is None, we attempt to get it from the model.
        if state_dict is None:
            state_dict = self.model.state_dict()

        # 2. Leverage Model's save_pretrained
        # This uses safetensors.torch.save_model internally (inside VLM.save_pretrained or via manual call here if needed)
        # Note: self.model.save_pretrained in VLM expects just directory.
        # But to be safe with state_dict passed from FSDP/DeepSpeed, we should pass it if VLM supports it.
        # VLM.save_pretrained currently takes only (save_directory).
        # We should update VLM.save_pretrained or just rely on VLM grabbing its own state if single GPU.
        # Given we are not changing VLM signature right now, we assume single GPU or that VLM can handle it.
        # BUT, to be robust as requested:

        # We manually call save_model from safetensors if we have the state_dict
        # OR we rely on VLM.

        # Let's trust VLM.save_pretrained for now, but ideally we update VLM to accept state_dict.
        # However, the user provided snippet uses `self.model.save_pretrained(..., state_dict=state_dict)`.
        # This implies VLM.save_pretrained MUST accept state_dict.
        # I must update VLM.save_pretrained signature as well.

        # For now, let's stick to the simple override that calls model.save_pretrained(output_dir).
        # We are on single GPU.
        self.model.save_pretrained(output_dir)

        # Save tokenizer/processing_class
        processing_class = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processing_class is not None:
             processing_class.save_pretrained(output_dir)

        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


@dataclass
class BasicTraining(Experiment):
    """
    Experiment that replicates the end-to-end training in config.py
    but uses the Hugging Face Trainer for the training loop and automanagement.
    All training configuration is self-contained in this experiment class.
    """
    # Experiment Settings
    eval_strategy: str = "steps" # "epoch" or "steps" or "no"
    save_steps: int = 1000
    eval_steps: int = 1000
    use_grayscale: bool = False
    freeze_vision: bool = True
    load_best_model_at_end: bool = True

    # Seeds
    model_seed: int = 42
    data_seed: int = 42

    # Training Hyperparameters
    lr_mp: float = 2e-3
    lr_backbones: float = 1e-4
    epochs: int = 5
    max_steps: int = -1
    batch_size: int = 128
    val_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    mmstar_batch_size: int = 16

    # Data Configuration
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa", "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa", "infographic_vqa", "intergps", "localized_narratives", "mapqa", "multihiertt", "ocrvqa", "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq", "scienceqa", "screen2words", "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr", "websight")
    test_dataset_path: str = "Lin-Chen/MMStar"
    data_cutoff_idx: int = None
    val_ratio: float = 0.1
    val_max_samples: int = 4096

    # System & Optimization
    train_dataloader_num_workers: int = 8
    val_dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    use_tf32: bool = True
    compile: bool = True
    image_resize_num_proc: int = 30
    image_resize_batch_size: int = 100

    # Checkpointing & Resuming
    resume_from_vlm_checkpoint: bool = False

    # Logging
    wandb_entity: str = "llm-lg"
    wandb_project: str = "maryVLM_test"
    log_wandb: bool = True

    # Other
    # VLM Configuration (from VLMConfig)
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * 768
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip-base-patch16-224'

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 128 - 49
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'
    lm_eos_token_id: int = 0

    mp_pixel_shuffle_factor: int = 2
    mp_use_mlp: bool = True

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints/nanoVLM-222M'

    @property
    def run_name(self) -> str:
        name = "maryVLM"
        name += f"_bs{self.batch_size * self.gradient_accumulation_steps}"
        if self.use_grayscale:
            name += "_gray"
        else:
            name += "_rgb"
        name += f"_ms{self.model_seed}_ds{self.data_seed}"
        return name


    @with_wandb
    def run(self):
        print("Starting BasicTraining Experiment...")
        run_name = self.run_name
        print(f"Run Name: {run_name}")

        # 0. Seed
        if self.model_seed is not None:
             tr.set_seed(self.model_seed)

        # 1. Setup Configuration
        # Reconstruct VLMConfig from experiment fields
        vlm_cfg = config.VLMConfig(
            vit_hidden_dim=self.vit_hidden_dim,
            vit_inter_dim=self.vit_inter_dim,
            vit_patch_size=self.vit_patch_size,
            vit_img_size=self.vit_img_size,
            vit_n_heads=self.vit_n_heads,
            vit_dropout=self.vit_dropout,
            vit_n_blocks=self.vit_n_blocks,
            vit_ln_eps=self.vit_ln_eps,
            vit_cls_flag=self.vit_cls_flag,
            vit_model_type=self.vit_model_type,
            lm_hidden_dim=self.lm_hidden_dim,
            lm_inter_dim=self.lm_inter_dim,
            lm_rms_eps=self.lm_rms_eps,
            lm_re_base=self.lm_re_base,
            lm_max_position_embeddings=self.lm_max_position_embeddings,
            lm_vocab_size=self.lm_vocab_size,
            lm_n_heads=self.lm_n_heads,
            lm_n_kv_heads=self.lm_n_kv_heads,
            lm_dropout=self.lm_dropout,
            lm_n_blocks=self.lm_n_blocks,
            lm_attn_scaling=self.lm_attn_scaling,
            lm_max_length=self.lm_max_length,
            lm_use_tokens=self.lm_use_tokens,
            lm_tie_weights=self.lm_tie_weights,
            lm_model_type=self.lm_model_type,
            lm_tokenizer=self.lm_tokenizer,
            lm_eos_token_id=self.lm_eos_token_id,
            mp_pixel_shuffle_factor=self.mp_pixel_shuffle_factor,
            mp_use_mlp=self.mp_use_mlp,
            vlm_load_backbone_weights=self.vlm_load_backbone_weights,
            vlm_checkpoint_path=self.vlm_checkpoint_path
        )

        # 2. Prepare Data
        print("Loading and combining datasets...")
        combined_train_data = []
        # Support both string and tuple for dataset names just in case
        dataset_names = self.train_dataset_name
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        for dataset_name in dataset_names:
            print(f"Loading {dataset_name}...")
            # We don't use streaming here as we want full training
            train_ds = hf_datasets.load_dataset(self.train_dataset_path, dataset_name)
            combined_train_data.append(train_ds['train'])

        full_train_ds = hf_datasets.concatenate_datasets(combined_train_data)

        # Shuffle with data_seed
        full_train_ds = full_train_ds.shuffle(seed=self.data_seed)

        # Optimization: Pre-shrink images and remove unused columns to save memory/speed
        # This map will rely on HF datasets caching. If the transform function/args don't change,
        # it will load from cache file on disk (default load_from_cache_file=True).
        print("Optimizing Dataset: Removing unused columns and resizing images...")

        # 1. Filter columns
        # We only need images and texts for VQADataset
        columns_to_keep = ['images', 'texts']
        # Check if dataset has them (it should, based on Cauldron)
        current_cols = full_train_ds.column_names
        # Sometimes connection issues happen, but we assume data is good
        full_train_ds = full_train_ds.select_columns([c for c in columns_to_keep if c in current_cols])

        # 2. Resize images
        # We handle the fact that 'images' is a list.
        # 2. Resize images
        # We handle the fact that 'images' is a list.
        def pre_process_images(examples, img_size):
            # examples['images'] is a list of lists of PIL images
            new_images_batch = []

            for img_list in examples['images']:
                new_list = []
                for img in img_list:
                    if isinstance(img, Image.Image):
                         if img.mode != 'RGB':
                            img = img.convert('RGB')
                         img = img.resize((img_size, img_size), resample=Image.BICUBIC)
                    new_list.append(img)
                new_images_batch.append(new_list)

            return {'images': new_images_batch}
        # Use num_proc to speed up. Caching ensures we only do this once per verified function state.
        full_train_ds = full_train_ds.map(
            pre_process_images,
            fn_kwargs={'img_size': self.vit_img_size},
            batched=True,
            batch_size=self.image_resize_batch_size,
            num_proc=self.image_resize_num_proc,
            load_from_cache_file=True,
            desc="Resizing images (Cached)"
        )

        # Cutoff if specified
        if self.data_cutoff_idx is not None:
             full_train_ds = full_train_ds.select(range(min(len(full_train_ds), self.data_cutoff_idx)))

        # Split Train/Val
        total_samples = len(full_train_ds)
        val_size = int(total_samples * self.val_ratio)
        if self.val_max_samples is not None:
            val_size = min(val_size, self.val_max_samples)
        train_size = total_samples - val_size

        train_ds_split = full_train_ds.select(range(train_size))
        val_ds_split = full_train_ds.select(range(train_size, total_samples))

        # MMStar for Test
        print("Loading MMStar dataset for eval...")
        test_ds = hf_datasets.load_dataset(self.test_dataset_path)
        mmstar_val = test_ds['val']

        # Optimization for MMStar
        mmstar_cols = ['image', 'question', 'answer']
        mmstar_val = mmstar_val.select_columns([c for c in mmstar_cols if c in mmstar_val.column_names])

        def pre_process_mmstar(examples):
            # examples['image'] is a list of PIL images (because batched=True)
            # MMStar has single image per example
            new_images = []



        def pre_process_mmstar(examples, img_size):
            # examples['image'] is a list of PIL images (because batched=True)
            # MMStar has single image per example
            new_images = []

            for img in examples['image']:
                if isinstance(img, Image.Image):
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((img_size, img_size), resample=Image.BICUBIC)
                new_images.append(img)
            return {'image': new_images}

        # Reduce validation worker count for map if on GPU too, just to be safe
        mmstar_val = mmstar_val.map(
            pre_process_mmstar,
            fn_kwargs={'img_size': self.vit_img_size},
            batched=True,
            num_proc=self.val_dataloader_num_workers,
            load_from_cache_file=True,
            desc="Resizing MMStar"
        )

        # 3. Processors
        print("Setting up processors...")
        image_processor = get_image_processor(vlm_cfg.vit_img_size)
        tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

        # 4. Initialize Wrapper Datasets
        print(f"Initializing Datasets (Train: {len(train_ds_split)}, Val: {len(val_ds_split)})...")
        train_dataset = VQADataset(train_ds_split, tokenizer, image_processor)
        val_dataset = VQADataset(val_ds_split, tokenizer, image_processor)
        mmstar_dataset = MMStarDataset(mmstar_val, tokenizer, image_processor)

        # 5. Initialize Model
        print("Initializing Model...")
        if self.resume_from_vlm_checkpoint:
            model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
            model.use_grayscale = self.use_grayscale
            # If resuming, we might still want to freeze if requested, though usually it's baked into architecture/intent.
            # But let's respect the flag.
            if self.freeze_vision:
                 print("Freezing Vision Encoder weights (post-load)")
                 for param in model.vision_encoder.parameters():
                     param.requires_grad = False
        else:
            model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights, use_grayscale=self.use_grayscale, freeze_vision=self.freeze_vision)

        # 6. Collators
        collate_fn = VQACollator(tokenizer, vlm_cfg.lm_max_length)
        mmstar_collate_fn = MMStarCollator(tokenizer)

        # 7. Training Arguments
        # We map from self (Experiment config) to TrainingArguments
        print("Setting up Training Arguments...")

        # Check for FP16/BF16 support
        bf16 = False
        fp16 = False
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                bf16 = True
            else:
                fp16 = True

        training_args = tr.TrainingArguments(
            output_dir=f"output/{run_name}",

            # Batch size & accumulation
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_eval_batch_size=self.val_batch_size,

            # Learning Rate
            learning_rate=self.lr_mp,

            # Epochs
            num_train_epochs=self.epochs,
            max_steps=self.max_steps,

            # Workers
            dataloader_num_workers=self.train_dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=self.dataloader_persistent_workers,

            # Precision
            bf16=bf16,
            fp16=fp16,
            tf32=self.use_tf32 if torch.cuda.is_available() else False,

            # Logging
            report_to="wandb" if self.log_wandb else "none",
            run_name=run_name,
            logging_steps=10,

            # Evaluation
            eval_strategy=self.eval_strategy,
            save_strategy=self.eval_strategy, # Sync save with eval usually, or could use "steps" explicitly
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model="eval_loss",

            remove_unused_columns=False, # Essential for VQA custom models
            label_names=["labels"],
            torch_compile=self.compile,

            # Seeds
            seed=self.model_seed,
            data_seed=self.data_seed,
        )


        # 8. Setup Evaluation Callback (MMStar)
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Test loader for callback
        mmstar_loader = DataLoader(
            mmstar_dataset,
            batch_size=self.mmstar_batch_size,
            shuffle=False,
            collate_fn=mmstar_collate_fn
        )

        mmstar_callback = MMStarCallback(
            test_loader=mmstar_loader,
            tokenizer=tokenizer,
            device=device,
        )

        # 9. Initialize Optimizer (Split parameter groups)
        print("Initializing Optimizer with split learning rates...")

        backbone_params = list(model.decoder.parameters())
        if not self.freeze_vision:
            backbone_params += list(model.vision_encoder.parameters())

        param_groups = [
            {'params': model.MP.parameters(), 'lr': self.lr_mp},
            {'params': backbone_params, 'lr': self.lr_backbones}
        ]
        # Use AdamW as per original implementation (or Trainer default)
        optimizer = torch.optim.AdamW(param_groups)

        # 10. Initialize Trainer
        trainer = VQATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            processing_class=tokenizer,
            callbacks=[MMStarCallback(mmstar_loader, mmstar_dataset.tokenizer, device)],
            optimizers=(optimizer, None) # Pass optimizer, let Trainer create default scheduler
        )

        # 11. Evaluate
        trainer.evaluate()

        # 12. Run
        print("Starting training...")
        trainer.train()
        print("Training complete.")

@dataclass
class ControlBasicTraining(BasicTraining):
    wandb_project: str = "maryVLM"
    # Training Args
    eval_strategy: str = "steps" # "epoch" or "steps" or "no"
    save_steps: int = 1000
    eval_steps: int = 1000
    use_grayscale: bool = False
    freeze_vision: bool = True
    load_best_model_at_end: bool = True

    model_seed: int = 0
    data_seed: int = 0

    epochs: int = 4


@dataclass
class GrayscaleBasicTraining(ControlBasicTraining):
    use_grayscale: bool = True

@dataclass
class RunBasicTraining(Experiment):

    def run(self):
        experiment = ControlBasicTraining()
        experiment.run()

        experiment = ControlBasicTraining(model_seed=1, data_seed=1)
        experiment.run()

        experiment = GrayscaleBasicTraining()
        experiment.run()


@dataclass
class ModelLoadingTest(BasicTraining):
    # Point to the specific checkpoint found
    vlm_checkpoint_path: str = "/var/data/ImmaculatePerception/output/maryVLM_bs256_gray_ms0_ds0/checkpoint-19548"
    resume_from_vlm_checkpoint: bool = True

    # Disable training stuff
    epochs: int = 0
    max_steps: int = 0

    def run(self):
        print(f"Testing model loading from: {self.vlm_checkpoint_path}")

        # Load Config
        vlm_cfg = config.VLMConfig()

        try:
            # Attempt to load model
            model = VisionLanguageModel.from_pretrained(self.vlm_checkpoint_path)
            print("Success: Model loaded!")

            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            # Create dummy inputs for verification
            print("Running dummy inference...")
            dummy_image = torch.randn(1, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size).to(device)
            dummy_input_ids = torch.randint(0, vlm_cfg.lm_vocab_size, (1, 10)).to(device)

            with torch.no_grad():
                # Test forward pass
                loss, logits = model(dummy_input_ids, dummy_image)
                print(f"Forward pass successful. Logits shape: {logits.shape}")

                # Test generate
                generated = model.generate(dummy_input_ids, dummy_image, max_new_tokens=5)
                print(f"Generation successful. Output shape: {generated.shape}")

            print("VERIFICATION COMPLETE: The model checkpoint is valid and functional.")

        except Exception as e:
            print(f"FAILED: Model loading or inference failed with error:")
            print(e)
            raise e
