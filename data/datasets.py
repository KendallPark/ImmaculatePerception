import torch
from PIL import Image
from torch.utils.data import Dataset

import models.config as cfg


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle image (it's a list)
        image_data = item['images']
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Process text (also a list)
        text_data = item['texts']
        if isinstance(text_data, list) and len(text_data) > 0:
            text = text_data[0]
        else:
            text = text_data

        question = text['user']
        # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        answer = text['assistant'] + self.tokenizer.eos_token

        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item['image']

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        question = item['question']
        answer = item['answer'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation

        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


class ColorShapeDataset(Dataset):
    """Synthetic shape/color dataset for VLM fine-tuning"""
    def __init__(self, dataset, tokenizer, image_processor, prompt_style='caption'):
        """
        Args:
            dataset: HF dataset from color_shape.py
            tokenizer: Tokenizer for text
            image_processor: Image processor
            prompt_style: 'caption' (empty prompt), 'color' ("The color is"),
                         'shape' ("The shape is"), 'both' ("The color is X and the shape is Y")
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_prompt_text(prompt_style):
        """Return the prompt text for a given style."""
        if prompt_style == 'caption':
            return ""
        elif prompt_style == 'color':
            return "The color is"
        elif prompt_style == 'shape':
            return "The shape is"
        elif prompt_style == 'both':
            return "Describe this shape:"
        else:
            return ""

    def __getitem__(self, idx):
        # Lazy import to avoid circular dependency
        import models.config as cfg
        import numpy as np # Needed for random choice

        item = self.dataset[idx]

        image = item['image']

        # Process image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Get caption from dataset
        caption = item['caption']

        # Determine prompt style
        current_style = self.prompt_style
        if self.prompt_style == 'random':
            # Mix strategies:
            # 40% clean caption (empty prompt)
            # 20% "The color is"
            # 20% "The shape is"
            # 20% "Describe this shape:"
            options = ['caption', 'color', 'shape', 'both']
            probs = [0.4, 0.2, 0.2, 0.2]
            current_style = np.random.choice(options, p=probs)

        # Format prompt based on style
        formatted_text = self.get_prompt_text(current_style)

        # If we use a specific prompt, we might want the target to be just the answer, not the full caption?
        # The current dataset structure implies 'answer' is the full caption.
        # But if prompt is "The color is", the answer should probably just be "red" or "red square"?
        # Actually, let's keep it simple: The model learns to complete the sentence.
        # If prompt is "The color is", and caption is "a red square", the completion "a red square" is awkward.
        # "The color is a red square" -> grammatically okay-ish but weird.
        # Ideally:
        # Style 'color' -> Prompt "The color is" -> Target "red"
        # Style 'shape' -> Prompt "The shape is" -> Target "square"
        # Style 'caption' -> Prompt "" -> Target "a red square"

        # Let's adjust target based on style too!

        answer_text = caption # Default
        if current_style == 'color':
            # Extract color from item
            answer_text = item['color']
        elif current_style == 'shape':
            answer_text = item['shape']
        elif current_style == 'both':
            answer_text = caption
        elif current_style == 'caption':
            answer_text = caption

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer_text + self.tokenizer.eos_token
        }
