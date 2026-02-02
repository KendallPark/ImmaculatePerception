"""
Synthetic Shape/Color Dataset for Vision-Language Fine-Tuning

Generates simple geometric shapes in various colors for supervised fine-tuning.
Validation set uses the exact 8×8 combinations from analysis (64 examples).
Training set uses expanded colors/shapes, excluding validation combinations.
"""

import io
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, load_from_disk
import os

# =========================================================
# Constants: Validation Set (from analysis)
# =========================================================

VAL_COLORS = {
    'black': (0, 0, 0),
    'blue': (0, 0, 255),
    'cyan': (0, 255, 255),
    'gray': (128, 128, 128),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255),
    'red': (255, 0, 0),
    'yellow': (255, 255, 0),
}

VAL_SHAPES = ['circle', 'cross', 'diamond', 'hexagon', 'pentagon', 'square', 'star', 'triangle']

# =========================================================
# Constants: Extended Training Set
# =========================================================

TRAIN_COLORS = {
    **VAL_COLORS,  # Include validation colors
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'brown': (165, 42, 42),
    'pink': (255, 192, 203),
    'white': (255, 255, 255),
    'lime': (0, 255, 0),
    'navy': (0, 0, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'teal': (0, 128, 128),
    'coral': (255, 127, 80),
    'gold': (255, 215, 0),
    'indigo': (75, 0, 130),
    # 'khaki': (240, 230, 140), # Removed for generalization
    # 'lavender': (230, 230, 250),
    # 'mint': (189, 252, 201),
    # 'peach': (255, 218, 185),
    'salmon': (250, 128, 114),
    'silver': (192, 192, 192),
    'turquoise': (64, 224, 208),
    'crimson': (220, 20, 60),
    # 'beige': (245, 245, 220),
    # 'azure': (240, 255, 255),
    'chocolate': (210, 105, 30),
    # 'ivory': (255, 255, 240),
    'plum': (221, 160, 221),
    # 'tan': (210, 180, 140),
    # 'wheat': (245, 222, 179),
    'orchid': (218, 112, 214),
    'sienna': (160, 82, 45),
}

TRAIN_SHAPES = VAL_SHAPES + [
    'octagon',
    'oval',
    'rectangle',
    'trapezoid',
    'rhombus',
    'crescent',
    'heart',
    'parallelogram',
    'arrow',
    'cylinder',
    'semicircle',
    'heptagon',
    'kite',
    'ring',
]

# =========================================================
# Shape Drawing Functions
# =========================================================

def draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: Tuple[int, int, int],
               img_size: int = 224, center: Tuple[int, int] = None, size: int = None):
    """
    Draw a shape on the given ImageDraw object.

    Args:
        draw: PIL ImageDraw object
        shape: Name of shape to draw
        color: RGB tuple
        img_size: Image size (assumes square)
        center: (x, y) center point, defaults to image center
        size: Radius/half-width of shape, defaults to img_size // 3
    """
    if center is None:
        center = (img_size // 2, img_size // 2)
    if size is None:
        size = img_size // 3

    cx, cy = center

    if shape == 'circle':
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)

    elif shape == 'square':
        draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)

    elif shape == 'triangle':
        points = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size)
        ]
        draw.polygon(points, fill=color)

    elif shape == 'pentagon':
        points = []
        for i in range(5):
            angle = 2 * np.pi * i / 5 - np.pi / 2
            x = cx + size * np.cos(angle)
            y = cy + size * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    elif shape == 'hexagon':
        points = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            x = cx + size * np.cos(angle)
            y = cy + size * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    elif shape == 'diamond':
        points = [
            (cx, cy - size),
            (cx + size, cy),
            (cx, cy + size),
            (cx - size, cy)
        ]
        draw.polygon(points, fill=color)

    elif shape == 'cross':
        w = size // 3
        draw.rectangle([cx - w, cy - size, cx + w, cy + size], fill=color)
        draw.rectangle([cx - size, cy - w, cx + size, cy + w], fill=color)

    elif shape == 'star':
        points = []
        for i in range(10):
            angle = 2 * np.pi * i / 10 - np.pi / 2
            r = size if i % 2 == 0 else size // 2
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    elif shape == 'octagon':
        points = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            x = cx + size * np.cos(angle)
            y = cy + size * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    elif shape == 'oval':
        draw.ellipse([cx - size, cy - size//2, cx + size, cy + size//2], fill=color)

    elif shape == 'rectangle':
        draw.rectangle([cx - size, cy - size//2, cx + size, cy + size//2], fill=color)

    elif shape == 'trapezoid':
        w_top = size // 2
        w_bottom = size
        points = [
            (cx - w_top, cy - size//2),
            (cx + w_top, cy - size//2),
            (cx + w_bottom, cy + size//2),
            (cx - w_bottom, cy + size//2)
        ]
        draw.polygon(points, fill=color)

    elif shape == 'rhombus':
        points = [
            (cx, cy - size),
            (cx + size//2, cy),
            (cx, cy + size),
            (cx - size//2, cy)
        ]
        draw.polygon(points, fill=color)

    elif shape == 'crescent':
        # Draw two circles to create crescent
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        offset = size // 3
        bg_color = (255, 255, 255)  # White background
        draw.ellipse([cx - size + offset, cy - size, cx + size + offset, cy + size], fill=bg_color)

    elif shape == 'heart':
        # Simple heart shape using circles and triangle
        r = size // 2
        draw.ellipse([cx - size, cy - size//2, cx, cy + size//2], fill=color)
        draw.ellipse([cx, cy - size//2, cx + size, cy + size//2], fill=color)
        points = [
            (cx - size, cy),
            (cx, cy + size),
            (cx + size, cy)
        ]
        draw.polygon(points, fill=color)

    elif shape == 'parallelogram':
        skew = size // 2
        points = [
            (cx - size + skew, cy - size//2),
            (cx + size + skew, cy - size//2),
            (cx + size - skew, cy + size//2),
            (cx - size - skew, cy + size//2)
        ]
        draw.polygon(points, fill=color)

    elif shape == 'arrow':
        # Arrow pointing right
        stem_w = size // 2
        stem_len = size
        head_len = size

        points = [
            (cx - size, cy - stem_w//2), # top-left of stem
            (cx, cy - stem_w//2),         # top-right of stem (base of head)
            (cx, cy - size),              # top tip of head
            (cx + size, cy),              # point of arrow
            (cx, cy + size),              # bottom tip of head
            (cx, cy + stem_w//2),         # bottom-right of stem
            (cx - size, cy + stem_w//2)   # bottom-left of stem
        ]
        draw.polygon(points, fill=color)

    elif shape == 'cylinder':
        # Approximated 2D cylinder
        # Draw body rectangle
        draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        # Draw top and bottom ellipses
        draw.ellipse([cx - size, cy - size - size//4, cx + size, cy - size + size//4], fill=color)
        draw.ellipse([cx - size, cy + size - size//4, cx + size, cy + size + size//4], fill=color)

    elif shape == 'semicircle':
        # Draw full circle then mask bottom half with white rectangle
        # Actually easier to use pieslice or chord
        draw.pieslice([cx - size, cy - size, cx + size, cy + size], start=180, end=0, fill=color)

    elif shape == 'heptagon':
        points = []
        for i in range(7):
            angle = 2 * np.pi * i / 7 - np.pi / 2
            x = cx + size * np.cos(angle)
            y = cy + size * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)

    elif shape == 'kite':
        points = [
            (cx, cy - size),          # Top
            (cx + size//2, cy - size//4), # Right
            (cx, cy + size),          # Bottom
            (cx - size//2, cy - size//4)  # Left
        ]
        draw.polygon(points, fill=color)

    elif shape == 'ring':
        # Draw larger circle in color, smaller circle in white
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        inner_size = size // 2
        bg_color = (255, 255, 255)
        draw.ellipse([cx - inner_size, cy - inner_size, cx + inner_size, cy + inner_size], fill=bg_color)

# =========================================================
# Image Generation
# =========================================================

def generate_shape_image(color_name: str, color_rgb: Tuple[int, int, int],
                         shape: str, img_size: int = 224,
                         rotation: float = 0.0, scale: float = 1.0,
                         offset: Tuple[int, int] = (0, 0),
                         noise_std: float = 0.0) -> Image.Image:
    """
    Generate a single shape image with augmentations.

    Args:
        rotation: Degrees to rotate
        scale: Scaling factor ( <1.0 smaller, >1.0 larger)
        offset: (dx, dy) in pixels
        noise_std: Standard deviation for Gaussian noise (0-255 scale)
    """
    # 1. Create a larger canvas to avoid clipping during rotation
    canvas_size = int(img_size * 1.5)
    img = Image.new('RGB', (canvas_size, canvas_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 2. Draw shape in center of large canvas
    # Apply scaling to the base size
    base_size = int((img_size // 3) * scale)
    center = (canvas_size // 2 + int(offset[0]), canvas_size // 2 + int(offset[1]))

    draw_shape(draw, shape, color_rgb, canvas_size, center=center, size=base_size)

    # 3. Rotate
    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, fillcolor=(255,255,255))

    # 4. Crop back to target size
    left = (canvas_size - img_size) // 2
    top = (canvas_size - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    # 5. Perspective Warping (New)
    # Apply random 4-point perspective transform
    # We define source points (corners) and destination points (perturbed corners)
    warp_scale = 0.2
    if warp_scale > 0:
        width, height = img.size
        # Source points: Top-Left, Bottom-Left, Bottom-Right, Top-Right
        src_points = [(0, 0), (0, height), (width, height), (width, 0)]

        def get_distortion(max_offset):
             return np.random.uniform(-max_offset, max_offset)

        max_dist = width * warp_scale

        # Destination points
        # Iterate and add jitter to each corner
        dst_points = []
        for x, y in src_points:
             dst_points.append((x + get_distortion(max_dist), y + get_distortion(max_dist)))

        # Calculate coefficients for perspective transform
        try:
             # PIL requires coefficients for reverse mapping (destination -> source)
             # x = (ax + by + c) / (gx + hy + 1)
             # y = (dx + ey + f) / (gx + hy + 1)
             # We can use find_coeffs helper or numpy

             def find_coeffs(pa, pb):
                matrix = []
                for p1, p2 in zip(pa, pb):
                    matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
                    matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

                A = np.matrix(matrix, dtype=float)
                B = np.array(pb).reshape(8)

                res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
                return np.array(res).reshape(8)

             # Note: transform uses inverse mapping, so we map DST -> SRC to fill DST pixels
             # We want to map our Rect (SRC) INTO the Distorted (DST) shape...
             # Actually, if we want the output image to look distorted, we are mapping FROM the distorted space BACK to the source texture.
             # So we use find_coeffs(dst, src)

             coeffs = find_coeffs(dst_points, src_points)

             img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255, 255, 255))
        except Exception:
             # Fallback if matrix singular
             pass

    # 6. Add Noise
    if noise_std > 0:
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, noise_std, np_img.shape)
        np_img = np_img + noise
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

    return img

# =========================================================
# Dataset Generation
# =========================================================

def is_validation_example(color: str, shape: str) -> bool:
    """Check if this color-shape combo is in the validation set."""
    return color in VAL_COLORS and shape in VAL_SHAPES

def generate_caption(color: str, shape: str, template_idx: int = 0) -> str:
    """Generate caption with multiple template variations."""
    templates = [
        f"a {color} {shape}",
        f"the {color} {shape}",
        f"{color} {shape}",
        f"a {shape} that is {color}",
        f"the shape is {color} and {shape}",
    ]
    return templates[template_idx % len(templates)]

def create_color_shape_dataset(split='train', num_examples_per_combo=1,
                                 img_size=224, caption_templates=5, seed=42) -> Dataset:
    """
    Create synthetic shape/color dataset.

    Args:
        split: 'train' or 'validation'
        num_examples_per_combo: How many examples per color-shape combination
        img_size: Image size (square)
        caption_templates: Number of caption template variations per combo
        seed: Random seed for reproducibility

    Returns:
        HuggingFace Dataset with columns: image, color, shape, caption
    """
    # Set seed for this generation
    # We add split/num_examples/img_size to seed to ensure different settings produce consistent but distinct streams if needed,
    # but simplest is just to seed np.
    np.random.seed(seed)

    examples = []

    if split == 'validation':
        # Validation: only VAL_COLORS × VAL_SHAPES (64 examples)
        colors = VAL_COLORS
        shapes = VAL_SHAPES
    else:
        # Training: all combinations EXCEPT validation set
        colors = TRAIN_COLORS
        shapes = TRAIN_SHAPES

    for color_name, color_rgb in colors.items():
        for shape in shapes:
            # Skip validation examples in training set
            if split == 'train' and is_validation_example(color_name, shape):
                continue

            # Generate multiple examples per combination
            for i in range(num_examples_per_combo):

                # Determine augmentations
                # Determine augmentations
                if split == 'train':
                    # Mix in clean examples to anchor the model to canonical shapes/orientations
                    # Every 5th example is clean (20%), ensuring we always have some upright examples per combo
                    if i % 5 == 0:
                         rotation = 0.0
                         scale = 1.0
                         offset = (0, 0)
                         noise_std = 0.0
                    else:
                        # Random augmentations
                        rotation = np.random.uniform(0, 360)
                        scale = np.random.uniform(0.7, 1.3)
                        # Jitter relative to size (e.g., +/- 10%)
                        jitter_max = img_size * 0.1
                        offset = (np.random.uniform(-jitter_max, jitter_max),
                                  np.random.uniform(-jitter_max, jitter_max))
                        noise_std = np.random.uniform(0, 50.0) # Light to heavy noise
                else:
                    # No validation augmentation
                    rotation = 0.0
                    scale = 1.0
                    offset = (0, 0)
                    noise_std = 0.0

                img = generate_shape_image(color_name, color_rgb, shape, img_size,
                                           rotation=rotation, scale=scale,
                                           offset=offset, noise_std=noise_std)

                # Vary caption templates
                template_idx = i % caption_templates
                caption = generate_caption(color_name, shape, template_idx)

                # Convert PIL to bytes for HF dataset
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                examples.append({
                    'image': {'bytes': img_bytes.read()},
                    'color': color_name,
                    'shape': shape,
                    'caption': caption,
                })

    # Create HF dataset
    features = Features({
        'image': HFImage(),
        'color': Value('string'),
        'shape': Value('string'),
        'caption': Value('string'),
    })

    dataset = Dataset.from_list(examples, features=features)

    return dataset

def create_color_shape_dataset_dict(num_train_per_combo=5, num_val_per_combo=1,
                                     img_size=224, seed=42, cache_dir="data/cache/color_shape_dataset") -> DatasetDict:
    """
    Create train/validation split as DatasetDict, with caching.

    Args:
        num_train_per_combo: Examples per training combination
        num_val_per_combo: Examples per validation combination
        seed: Random seed
        cache_dir: Directory to store cached dataset

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    # Create a unique cache path based on parameters to avoid stale cache
    cache_name = f"seed{seed}_train{num_train_per_combo}_val{num_val_per_combo}_size{img_size}"
    full_cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(full_cache_path):
        print(f"Loading cached dataset from {full_cache_path}...")
        try:
            return DatasetDict.load_from_disk(full_cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print(f"Generating new dataset (seed={seed})...")
    train_ds = create_color_shape_dataset(
        split='train',
        num_examples_per_combo=num_train_per_combo,
        img_size=img_size,
        seed=seed
    )

    # Validation should also be seeded to be deterministic, though it has no random augs currently
    val_ds = create_color_shape_dataset(
        split='validation',
        num_examples_per_combo=num_val_per_combo,
        img_size=img_size,
        seed=seed
    )

    ds_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
    })

    print(f"Saving dataset to cache: {full_cache_path}")
    os.makedirs(os.path.dirname(full_cache_path), exist_ok=True)
    ds_dict.save_to_disk(full_cache_path)

    return ds_dict

# =========================================================
# Main (for testing)
# =========================================================

if __name__ == '__main__':
    print("Creating color-shape dataset...")

    # Match the ShapeColorSFT experiment settings
    ds_dict = create_color_shape_dataset_dict(num_train_per_combo=100, num_val_per_combo=1)

    print(f"Train: {len(ds_dict['train'])} examples")
    print(f"Validation: {len(ds_dict['validation'])} examples")

    # Show first example
    example = ds_dict['train'][0]
    print(f"\nExample: {example['color']} {example['shape']}")
    print(f"Caption: {example['caption']}")

    # Verify no validation examples in training
    train_combos = set((ex['color'], ex['shape']) for ex in ds_dict['train'])
    val_combos = set((ex['color'], ex['shape']) for ex in ds_dict['validation'])

    overlap = train_combos & val_combos
    if overlap:
        print(f"\nWARNING: Overlap found! {overlap}")
    else:
        print("\n✓ No overlap between train/val sets")
