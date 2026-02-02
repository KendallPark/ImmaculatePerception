import sys
import os
import re
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path to allow importing models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import (
    generate_stimuli, extract_activations, extract_text_activations,
    compute_mds, linear_cka, compute_rdm, representational_consistency, compute_procrustes,
    COLORS, SHAPES, generate_color_only_stimuli
)
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run VLM Analysis: MDS Plots, Metrics, and Text Generation")
    parser.add_argument("--plots", action="store_true", help="Generate MDS plots and metric comparisons")
    parser.add_argument("--text", action="store_true", help="Generate text output table")
    parser.add_argument("--all", action="store_true", help="Run all analyses (plots + text)")
    parser.add_argument("--stimuli-type", type=str, default="shapes", choices=["shapes", "colors"], help="Stimuli type for text generation (shapes or colors)")

    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.plots or args.text or args.all):
        args.all = True

    if args.all:
        args.plots = True
        args.text = True

    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure plots directory exists
    os.makedirs('analysis/plots', exist_ok=True)
    os.makedirs('analysis/results', exist_ok=True)

    # 1. Stimuli Generation (Common)
    print("Loading validation dataset...")
    # Use the official validation set source
    from data.color_shape import create_color_shape_dataset
    val_ds = create_color_shape_dataset(split='validation', num_examples_per_combo=1)

    stimuli = []
    for item in val_ds:
        stimuli.append({
            'image': item['image'],
            'color': item['color'],
            'shape': item['shape'],
            'label': f"{item['color']} {item['shape']}" # Simple label for plots
        })
    print(f"Loaded {len(stimuli)} validation stimuli.")

    # 2. Define Models
    configs = [
        {
            "name": r"MaryVLM$_{gray,0}$ (gray)",
            "repo_id": "output/sft_gray0_gray/checkpoint-291",
            "gray_input": True
        },
        {
            "name": r"MaryVLM$_{gray,0}$ (rgb)",
            "repo_id": "output/sft_gray0_rgb/checkpoint-291",
            "gray_input": False
        },
        {
            "name": r"MaryVLM$_{gray,0}^{cont}$ (rgb)",
            "repo_id": "output/sft_gray0_cont_rgb/checkpoint-291",
            "gray_input": False
        },
        {
            "name": r"MaryVLM$_{rgb,0}$ (rgb)",
            # "repo_id": "output/maryVLM_bs256_rgb_ms0_ds0/checkpoint-26064",
            "repo_id": "output/sft_rgb0/checkpoint-291",
            "gray_input": False
        },
        {
            "name": r"MaryVLM$_{rgb,1}$ (rgb)",
            "repo_id": "output/sft_rgb1/checkpoint-291",
            "gray_input": False
        }
    ]

    # --- PLOTS & METRICS SECTION ---
    if args.plots:
        print("\n" + "="*40)
        print("Running Plots & Metrics Analysis")
        print("="*40)

        # 1b. Text Stimuli & Tokenizer (Only needed for text activations)
        print("Setting up tokenizer for activation extraction...")
        activation_tokenizer = get_tokenizer('HuggingFaceTB/cosmo2-tokenizer')
        text_stimuli = [f"a {s['label']}" for s in stimuli]

        # 3. Extract Activations
        all_activations = {}
        all_text_activations = {}

        for cfg in configs:
            print(f"Processing: {cfg['name']}...")
            try:
                model = VisionLanguageModel.from_pretrained(cfg['repo_id'])

                # Extract Visual Activations
                acts = extract_activations(model, stimuli, device=device, use_grayscale_input=cfg['gray_input'])
                all_activations[cfg['name']] = acts

                # Extract Text Activations
                text_acts = extract_text_activations(model, activation_tokenizer, text_stimuli, device=device)
                all_text_activations[cfg['name']] = text_acts

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Failed to process {cfg['name']}: {e}")
                import traceback
                traceback.print_exc()

        # 4. MDS Visualization
        print("Generating MDS plots...")
        first_model_name = configs[0]['name']

        def layer_sort_key(n):
            if n == 'Logits': return 999
            if 'ViT' in n: return 100 + int(n.split('.')[-1])
            if 'Modality' in n: return 200
            if 'LLM' in n: return 300 + int(n.split('.')[-1])
            return 0

        if first_model_name in all_activations:
            all_layers = sorted(list(all_activations[first_model_name].keys()), key=layer_sort_key)

            colors_map = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y', 'cyan': 'c', 'magenta': 'm', 'white': 'k', 'gray': 'gray'}
            shapes_map = {
                'circle': 'o', 'square': 's', 'triangle': '^', 'pentagon': 'p',
                'hexagon': 'h', 'diamond': 'D', 'cross': 'P', 'star': '*'
            }

            for i, layer_name in enumerate(tqdm(all_layers, desc="Generating Plots")):
                plot_mds_grid(all_activations, all_text_activations, layer_name, stimuli, colors_map, shapes_map, index=i)

            # 5. Metric Comparison
            print("Generating Metric comparisons...")
            comparisons = [
                (r"MaryVLM$_{gray,0}$ (rgb)", r"MaryVLM$_{gray,0}$ (gray)"),
                (r"MaryVLM$_{gray,0}$ (rgb)", r"MaryVLM$_{gray,0}^{cont}$ (rgb)"),
                (r"MaryVLM$_{gray,0}^{cont}$ (rgb)", r"MaryVLM$_{rgb,0}$ (rgb)"),
                (r"MaryVLM$_{rgb,0}$ (rgb)", r"MaryVLM$_{rgb,1}$ (rgb)")
            ]
            plot_all_metric_comparisons(all_activations, comparisons)
        else:
            print("Skipping plots because activation extraction failed for first model.")

    # --- TEXT GENERATION SECTION ---
    if args.text:
        print("\n" + "="*40)
        print("Running Text Generation Analysis")
        print("="*40)

        # Decide stimuli for text generation
        if args.stimuli_type == "colors":
            gen_stimuli = generate_color_only_stimuli()
            print(f"Using {len(gen_stimuli)} color-only stimuli")
            prompt_style = "color" # Helper to define prompt logic
        else:
            gen_stimuli = stimuli # Use the shapes we generated earlier
            print(f"Using {len(gen_stimuli)} shape stimuli")
            prompt_style = "caption" # Default SFT style (empty prompt)

        generate_text_table(configs, gen_stimuli, device, prompt_style=prompt_style)

def generate_text_table(configs, stimuli, device, prompt_style="caption"):
    """
    Generates text for all stimuli using all models and saves a Markdown table.

    Args:
        prompt_style: "caption" (empty), "color" (The color is), or "shape" (The shape is)
    """
    results = {} # {model_name: [outputs]}

    # Load tokenizer for generation (SmolLM2-135M)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')

    # Load Image Processor using our helper
    from data.processors import get_image_processor
    # Use standard config name or path
    # We can fetch it from our VLMConfig default if we want
    import models.config as cfg
    image_processor = get_image_processor(cfg.VLMConfig.vit_img_size)


    # Determine prompt text based on style
    # Matching logic via shared method in data/datasets.py:ColorShapeDataset
    from data.datasets import ColorShapeDataset
    prompt_text = ColorShapeDataset.get_prompt_text(prompt_style)

    print(f"Using prompt: '{prompt_text}'")

    for cfg in configs:
        print(f"Generating for: {cfg['name']}...")
        try:
            model = VisionLanguageModel.from_pretrained(cfg['repo_id'])
            model.to(device)
            model.eval()

            outputs = []

            # Prepare Input IDs
            if prompt_text:
                prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
            else:
                # If prompt is empty, we still need input_ids for generate or just let it start?
                # VLM.generate usually expects input_ids.
                # If empty, we typically pass BOS token if available.
                if tokenizer.bos_token_id is not None:
                     prompt_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
                else:
                     # Fallback to empty tensor? HuggingFace generate handles it?
                     # Usually we need at least one token.
                     # Let's try passing None if model.generate handles it,
                     # or just a dummy token if we want unconditional generation?
                     # Wait, `ColorShapeDataset` returns formatted_text="".
                     # If we pass "" to tokenizer.encode(), we get empty list?
                     ids = tokenizer.encode("", add_special_tokens=False)
                     if not ids:
                         # Most models need a start token.
                         # Let's use BOS.
                         prompt_ids = torch.tensor([[tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id]], device=device)

            # Ensure 2D [B, L]
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids.unsqueeze(0)

            print(f"Prompt IDs: {prompt_ids[0].tolist()}")

            for i, stim in enumerate(tqdm(stimuli, desc="Generating")):
                img = stim['image']
                if cfg['gray_input']:
                     img = img.convert('L').convert('RGB')

                # Preprocess image
                # Use the correct image processor logic matching training
                pixel_values = image_processor(img)
                # Ensure it's a tensor and on device
                # image_processor might return a tensor or a list/dict depending on return_tensors
                # If we use our get_image_processor wrapper from data/processors, let's verify what it returns.
                # Actually, wait, let's just use the processor from the model config name or standard siglip
                # But to be safe, let's load it properly.

                # In data/datasets.py: processed_image = self.image_processor(image) which returns a tensor.
                # Let's import get_image_processor at top level and use it.

                img_tensor = pixel_values.unsqueeze(0).to(device)


                with torch.no_grad():
                    # generate(input_ids, image, ...)
                    gen_ids = model.generate(prompt_ids, img_tensor, max_new_tokens=10)
                    if i < 3:
                        print(f"Generated IDs: {gen_ids[0].tolist()}")
                    outputs.append(gen_ids.cpu())

            results[cfg['name']] = outputs

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"generation failed for {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Decoding and Table Creation
    print("Decoding and saving table...")

    # Create filenames based on prompt style
    filename_suffix = f"_{prompt_style}" if prompt_style != "caption" else ""

    # Markdown Table
    table = "| Stimulus | " + " | ".join([cfg['name'] for cfg in configs]) + " |\n"
    table += "|---|" + "|".join(["---" for _ in configs]) + "|\n"

    csv_rows = []
    csv_headers = ['Stimulus'] + [cfg['name'] for cfg in configs]
    csv_rows.append(csv_headers)

    num_stimuli = len(stimuli)
    for i in range(num_stimuli):
        label = stimuli[i].get('label', f"Item {i}")
        # Build Markdown Row
        row = f"| {label} |"

        # Build CSV Row
        csv_row = [label]

        for cfg in configs:
            if cfg['name'] in results:
                ids = results[cfg['name']][i][0]

                # Careful not to decode the prompt text itself if it's included in output
                # Standard HF generate returns [prompt + generated]
                # We should slice off prompt length if we want pure generation
                # But here we just decode everything and user can see context

                text = tokenizer.decode(ids, skip_special_tokens=True).strip()

                # If prompt was "The color is", remove it for cleaner table?
                if prompt_text and text.startswith(prompt_text):
                    text = text[len(prompt_text):].strip()

                if i < 3:
                     print(f"Decoded ({cfg['name']}): '{text}'")

                # Markdown escape
                md_text = text.replace("|", "\\|").replace("\n", " ")
                row += f" {md_text} |"
                csv_row.append(text)
            else:
                row += " N/A |"
                csv_row.append("N/A")

        table += row + "\n"
        csv_rows.append(csv_row)

    md_path = f'analysis/results/generation_table{filename_suffix}.md'
    csv_path = f'analysis/results/generation_table{filename_suffix}.csv'

    with open(md_path, 'w') as f:
        f.write(f"# Text Generation Results ({prompt_style})\n\n")
        f.write(table)
    print(f"Saved table to {md_path}")

    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Saved CSV to {csv_path}")

def plot_mds_grid(all_acts, all_text_acts, layer_name, stimuli, colors_map, shapes_map, index=None):
    plot_data = []
    for name, acts in all_acts.items():
        if layer_name in acts:
            act_data = acts[layer_name]
            mds_coords = compute_mds({layer_name: act_data})[layer_name]
            plot_data.append((name, mds_coords, False))

    if not plot_data:
        print(f"Layer {layer_name} not found.")
        return

    fig, axes = plt.subplots(1, len(plot_data), figsize=(5 * len(plot_data), 5))
    if len(plot_data) == 1: axes = [axes]

    for ax, (model_name, coords, has_text) in zip(axes, plot_data):
        num_stimuli = len(stimuli)
        img_coords = coords[:num_stimuli] if has_text else coords

        for i, item in enumerate(stimuli):
            c = colors_map.get(item.get('color', 'gray'), 'k')
            m = shapes_map.get(item.get('shape', 'circle'), 'o')
            ax.scatter(img_coords[i, 0], img_coords[i, 1], c=c, marker=m, s=100, alpha=0.7, edgecolors='k')

        ax.set_title(f"{model_name}\n{layer_name}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if index is not None:
        save_path = f"analysis/plots/{index:02d}_{layer_name}.png"
    else:
        save_path = f"analysis/plots/{layer_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved MDS plot to {save_path}")

def plot_all_metric_comparisons(all_acts, comparisons):
    results = []

    for (model_a, model_b) in comparisons:
        if model_a not in all_acts or model_b not in all_acts:
            continue

        acts_a = all_acts[model_a]
        acts_b = all_acts[model_b]
        layers = [l for l in acts_a.keys() if l in acts_b]

        cka_scores = []
        const_scores = []
        proc_scores = []

        for layer in tqdm(layers, desc=f"Compare {model_a} vs {model_b}", leave=False):
            X = acts_a[layer].float()
            Y = acts_b[layer].float()
            cka_scores.append(linear_cka(X, Y))
            rdm1 = compute_rdm(X, metric='correlation')
            rdm2 = compute_rdm(Y, metric='correlation')
            const_scores.append(representational_consistency(rdm1, rdm2))
            proc_scores.append(compute_procrustes(X, Y))

        results.append({
            'label': f"{model_a} vs {model_b}",
            'cka': cka_scores,
            'const': const_scores,
            'proc': proc_scores,
            'layers': layers
        })

    if not results:
        print("No comparisons generated.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    common_layers = results[0]['layers']
    x = range(len(common_layers))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for i, res in enumerate(results):
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]
        lbl = res['label']
        axes[0].plot(x, res['cka'], marker=m, color=c, label=lbl, alpha=0.8)
        axes[1].plot(x, res['const'], marker=m, color=c, label=lbl, alpha=0.8)
        axes[2].plot(x, res['proc'], marker=m, color=c, label=lbl, alpha=0.8)

    axes[0].set_ylabel("CKA Similarity")
    axes[0].set_title("CKA Similarity")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize='small')

    axes[1].set_ylabel("Rep. Consistency (Spearman)")
    axes[1].set_title("Representational Consistency")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    axes[2].set_ylabel("Procrustes Disparity (Lower is better)")
    axes[2].set_title("Procrustes Distance")
    axes[2].grid(True, alpha=0.3)

    plt.xticks(x, common_layers, rotation=90)
    plt.tight_layout()

    save_path = "analysis/plots/combined_metrics_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined metrics plot to {save_path}")

if __name__ == "__main__":
    main()
