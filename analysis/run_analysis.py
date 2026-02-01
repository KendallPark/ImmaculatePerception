import sys
import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path to allow importing models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import (
    generate_stimuli, extract_activations, extract_text_activations,
    compute_mds, linear_cka, compute_rdm, representational_consistency, compute_procrustes,
    COLORS, SHAPES
)
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer

def main():
    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure plots directory exists
    os.makedirs('analysis/plots', exist_ok=True)

    # 1. Stimuli Generation
    print("Generating stimuli...")
    stimuli = generate_stimuli()
    print(f"Generated {len(stimuli)} stimuli.")

    # 1b. Text Stimuli & Tokenizer
    print("Setting up tokenizer...")
    # Hardcoded or from config? Using default from basic_training setup
    tokenizer = get_tokenizer('HuggingFaceTB/cosmo2-tokenizer')
    text_stimuli = [f"a {s['label']}" for s in stimuli]
    print(f"Generated {len(text_stimuli)} text captions.")

    # 2. Define Models
    # Using local checkpoints found in output/
    configs = [
        {
            "name": r"MaryVLM$_{gray,0}$ (gray)",
            "repo_id": "output/maryVLM_bs256_gray_ms0_ds0/checkpoint-26064",
            "gray_input": True
        },
        {
            "name": r"MaryVLM$_{gray,0}$ (rgb)",
            "repo_id": "output/maryVLM_bs256_gray_ms0_ds0/checkpoint-26064",
            "gray_input": False
        },
        {
            "name": r"MaryVLM$_{gray,0}^{cont}$ (rgb)",
            "repo_id": "output/maryVLM_bs256_gray_ms0_ds0_22000_cont_rgb_1_lr_mp0.001_lr_backbone0.005/checkpoint-5000",
            "gray_input": False
        },
        {
            "name": r"MaryVLM$_{rgb,0}$ (rgb)",
            "repo_id": "output/maryVLM_bs256_rgb_ms0_ds0/checkpoint-26064",
            "gray_input": False
        },
        {
            "name": r"MaryVLM$_{rgb,1}$ (rgb)",
            "repo_id": "output/maryVLM_bs256_rgb_ms1_ds1/checkpoint-26064",
            "gray_input": False
        }
    ]

    # 3. Extract Activations
    all_activations = {}
    all_text_activations = {}

    for cfg in configs:
        print(f"Processing: {cfg['name']}...")
        try:
            # Load Model
            # Note: The model loading logic in VisionLanguageModel.from_pretrained handles local paths
            model = VisionLanguageModel.from_pretrained(cfg['repo_id'])

            # Extract
            acts = extract_activations(model, stimuli, device=device, use_grayscale_input=cfg['gray_input'])
            all_activations[cfg['name']] = acts

            # Extract Text (LLM layers only)
            text_acts = extract_text_activations(model, tokenizer, text_stimuli, device=device)
            all_text_activations[cfg['name']] = text_acts

            # Cleanup to save VRAM
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed to process {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("Extraction Complete.")

    # 4. MDS Visualization
    print("Generating MDS plots...")

    # Get all available layers from the first model
    first_model_name = configs[0]['name']
    # Sort layers to put Logits at the end
    def layer_sort_key(n):
        if n == 'Logits': return 999
        if 'ViT' in n: return 100 + int(n.split('.')[-1])
        if 'Modality' in n: return 200
        if 'LLM' in n: return 300 + int(n.split('.')[-1])
        return 0

    all_layers = sorted(list(all_activations[first_model_name].keys()), key=layer_sort_key)
    print(f"Found {len(all_layers)} layers across models (including Logits).")

    colors_map = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y', 'cyan': 'c', 'magenta': 'm', 'white': 'k', 'gray': 'gray'}
    # Updated shapes map for new diverse shapes
    shapes_map = {
        'circle': 'o',
        'square': 's',
        'triangle': '^',
        'pentagon': 'p',
        'hexagon': 'h', # or 'H'
        'diamond': 'D',
        'cross': 'P',
        'star': '*'
    }

    for i, layer_name in enumerate(tqdm(all_layers, desc="Generating Plots")):
        plot_mds_grid(all_activations, all_text_activations, layer_name, stimuli, colors_map, shapes_map, index=i)

    # 5. Metric Comparison (CKA, Consistency, Procrustes)
    print("Generating Metric comparisons...")

    # Define all comparisons we want to plot together
    comparisons = [
        (r"MaryVLM$_{gray,0}$ (rgb)", r"MaryVLM$_{gray,0}$ (gray)"),
        (r"MaryVLM$_{gray,0}$ (rgb)", r"MaryVLM$_{gray,0}^{cont}$ (rgb)"),
        (r"MaryVLM$_{gray,0}^{cont}$ (rgb)", r"MaryVLM$_{rgb,0}$ (rgb)"),
        (r"MaryVLM$_{rgb,0}$ (rgb)", r"MaryVLM$_{rgb,1}$ (rgb)")
    ]

    plot_all_metric_comparisons(all_activations, comparisons)

    # 6. Text Generation Table
    print("Generating text output table...")
    generate_text_table(configs, stimuli, device)

def generate_text_table(configs, stimuli, device, use_color_only=True):
    """
    Generates text for all stimuli using all models and saves a Markdown table.

    Args:
        use_color_only: If True, use solid color images instead of shapes
    """
    # Override stimuli with color-only if requested
    if use_color_only:
        from analysis.utils import generate_color_only_stimuli
        stimuli = generate_color_only_stimuli()
        print(f"Using {len(stimuli)} color-only stimuli for text generation")
    results = {} # {model_name: [outputs]}

    # We need to reload models because we deleted them in the main loop to save VRAM
    # This is slightly inefficient but safe.

    for cfg in configs:
        print(f"Generating for: {cfg['name']}...")
        try:
            model = VisionLanguageModel.from_pretrained(cfg['repo_id'])
            model.to(device)
            model.eval()

            outputs = []

            # Load tokenizer to encode prompt
            # Use the correct tokenizer for SmolLM2-135M
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')

            # Use color-focused prompt for color-only test
            prompt_text = "The color is" if use_color_only else "The shape is"

            # Use tokenizer to encode.
            # Assuming the tokenizer handles special tokens or we add them.
            # We will use encode() which usually adds BOS if configured.

            prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt')
            input_ids = prompt_ids.to(device)
            print(f"Prompt IDs: {input_ids[0].tolist()}")

            for i, stim in enumerate(tqdm(stimuli, desc="Generating")):
                img = stim['image']
                if cfg['gray_input']:
                     img = img.convert('L').convert('RGB')

                # Preprocess image (same as in extract_activations)
                # ToTensor: [H, W, C] -> [C, H, W] and normalize 0-1
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

                img_tensor = img_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    # Generate
                    # We utilize the model.generate method we saw in VisionLanguageModel
                    # generate(self, input_ids, image, attention_mask=None, max_new_tokens=5)
                    gen_ids = model.generate(input_ids, img_tensor, max_new_tokens=10)
                    if i < 3: # Debug print for first few
                        print(f"Generated IDs: {gen_ids[0].tolist()}")

                    # Store IDs for now, need decoding
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
    # Load tokenizer directly to ensure correctness
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')

    # Header
    table = "| Stimulus | " + " | ".join([cfg['name'] for cfg in configs]) + " |\n"
    table += "|---|" + "|".join(["---" for _ in configs]) + "|\n"

    num_stimuli = len(stimuli)
    for i in range(num_stimuli):
        row = f"| {stimuli[i]['label']} |"
        for cfg in configs:
            if cfg['name'] in results:
                ids = results[cfg['name']][i][0] # [1, len] -> [len]
                # Decode
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                if i < 3:
                     print(f"Decoded text ({cfg['name']}): '{text}' (Raw IDs: {ids.tolist()})")

                # Escape pipes
                text = text.replace("|", "\\|").replace("\n", " ")
                row += f" {text} |"
            else:
                row += " N/A |"
        table += row + "\n"

    os.makedirs('analysis/results', exist_ok=True)
    with open('analysis/results/generation_table.md', 'w') as f:
        f.write("# Text Generation Results\n\n")
        f.write(table)

    print("Saved generation table to analysis/results/generation_table.md")

    # Save as CSV
    import csv
    with open('analysis/results/generation_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['Stimulus'] + [cfg['name'] for cfg in configs]
        writer.writerow(headers)

        for i in range(num_stimuli):
            row_data = [stimuli[i]['label']]
            for cfg in configs:
                if cfg['name'] in results:
                    ids = results[cfg['name']][i][0]
                    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    row_data.append(text)
                else:
                    row_data.append("N/A")
            writer.writerow(row_data)

    print("Saved generation table to analysis/results/generation_table.csv")

def plot_mds_grid(all_acts, all_text_acts, layer_name, stimuli, colors_map, shapes_map, index=None):
    plot_data = []
    for name, acts in all_acts.items():
        if layer_name in acts:
            act_data = acts[layer_name]

            # Revert to Image-Only plotting as requested
            # We ignore text activations for the plot even if they exist
            mds_coords = compute_mds({layer_name: act_data})[layer_name]
            plot_data.append((name, mds_coords, False))

    if not plot_data:
        print(f"Layer {layer_name} not found.")
        return

    fig, axes = plt.subplots(1, len(plot_data), figsize=(5 * len(plot_data), 5))
    if len(plot_data) == 1: axes = [axes]

    for ax, (model_name, coords, has_text) in zip(axes, plot_data):
        num_stimuli = len(stimuli)

        # Plot Images (Filled)
        # If has_text, coords contains [Images, Text]. Image coords are first N.
        img_coords = coords[:num_stimuli] if has_text else coords

        for i, item in enumerate(stimuli):
            c = colors_map.get(item['color'], 'k')
            m = shapes_map.get(item['shape'], 'o')
            ax.scatter(img_coords[i, 0], img_coords[i, 1], c=c, marker=m, s=100, alpha=0.7, edgecolors='k')

        # Plot Text (Empty with outline)
        if has_text:
            text_coords = coords[num_stimuli:]
            for i, item in enumerate(stimuli):
                c = colors_map.get(item['color'], 'k')
                m = shapes_map.get(item['shape'], 'o')
                # Empty face, colored edge
                ax.scatter(text_coords[i, 0], text_coords[i, 1], facecolors='none', edgecolors=c, marker=m, s=100, linewidths=2)

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
    """
    Plots CKA, Consistency, and Procrustes for multiple pairs of models on the same figure.
    comparisons: list of tuples (model_a_name, model_b_name)
    """

    # Store results to plot later
    results = [] # List of {'label': str, 'cka': [], 'const': [], 'proc': [], 'layers': []}

    # Use the first comparison to determine common layers (assuming all have same layers for simplicity,
    # but we will check intersection for each pair)
    # Ideally we want them all on the same x-axis, so we should probably find the intersection of ALL models involved?
    # Or just assume standard layers. Let's compute for each pair.

    for (model_a, model_b) in comparisons:
        if model_a not in all_acts or model_b not in all_acts:
            print(f"Skipping {model_a} vs {model_b}: missing activations.")
            continue

        acts_a = all_acts[model_a]
        acts_b = all_acts[model_b]

        # Intersection of layers, sorted/preserved order
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

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    # Use the layers from the first result as x-axis (assuming they match, which they should for same arch)
    common_layers = results[0]['layers']
    x = range(len(common_layers))

    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for i, res in enumerate(results):
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]
        lbl = res['label']

        # CKA
        axes[0].plot(x, res['cka'], marker=m, color=c, label=lbl, alpha=0.8)

        # Consistency
        axes[1].plot(x, res['const'], marker=m, color=c, label=lbl, alpha=0.8)

        # Procrustes
        axes[2].plot(x, res['proc'], marker=m, color=c, label=lbl, alpha=0.8)

    # Styling
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

def compare_metrics(model_a_name, model_b_name, all_acts):
    if model_a_name not in all_acts or model_b_name not in all_acts:
        print(f"Skipping comparisons: {model_a_name} or {model_b_name} not in activations.")
        return

    activations_a = all_acts[model_a_name]
    activations_b = all_acts[model_b_name]

    # Get common layers
    layers = [l for l in activations_a.keys() if l in activations_b]

    cka_scores = []
    consistency_scores = []
    procrustes_scores = []

    for layer in tqdm(layers, desc=f"Comparing {model_a_name} vs {model_b_name}", leave=False):
        # Flattened activations
        X = activations_a[layer].float()
        Y = activations_b[layer].float()

        # 1. CKA
        cka = linear_cka(X, Y)
        cka_scores.append(cka)

        # 2. Representational Consistency
        rdm1 = compute_rdm(X, metric='correlation')
        rdm2 = compute_rdm(Y, metric='correlation')
        const = representational_consistency(rdm1, rdm2)
        consistency_scores.append(const)

        # 3. Procrustes
        proc = compute_procrustes(X, Y)
        procrustes_scores.append(proc)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Plot CKA
    axes[0].plot(cka_scores, marker='o', color='tab:blue')
    axes[0].set_ylabel("CKA Similarity")
    axes[0].set_title(f"CKA Similarity: {model_a_name} vs {model_b_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Plot Consistency
    axes[1].plot(consistency_scores, marker='s', color='tab:orange')
    axes[1].set_ylabel("Rep. Consistency (Spearman)")
    axes[1].set_title(f"Representational Consistency: {model_a_name} vs {model_b_name}")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    # Plot Procrustes
    axes[2].plot(procrustes_scores, marker='^', color='tab:green')
    axes[2].set_ylabel("Procrustes Disparity (Lower is better)")
    axes[2].set_title(f"Procrustes Distance: {model_a_name} vs {model_b_name}")
    axes[2].grid(True, alpha=0.3)

    # X-Axis
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.tight_layout()

    # Sanitize names for filenames
    def sanitize(n):
        return re.sub(r'[^a-zA-Z0-9]', '_', n).strip('_')

    safe_name_a = sanitize(model_a_name)
    safe_name_b = sanitize(model_b_name)
    save_path = f"analysis/plots/metrics_{safe_name_a}_vs_{safe_name_b}.png"

    plt.savefig(save_path)
    plt.close()
    print(f"Saved Metrics plot to {save_path}")

if __name__ == "__main__":
    main()
