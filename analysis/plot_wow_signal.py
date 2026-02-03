import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def parse_layer_order(layer_name):
    """
    Returns a tuple sort key for layer names.
    Order:
    1. ViT.Block.N (0-11)
    2. ModalityProjector
    3. LLM.Block.N (0-29)
    """
    if layer_name == "ModalityProjector":
        return (1, 0)

    match_vit = re.match(r"ViT\.Block\.(\d+)", layer_name)
    if match_vit:
        return (0, int(match_vit.group(1)))

    match_llm = re.match(r"LLM\.Block\.(\d+)", layer_name)
    if match_llm:
        return (2, int(match_llm.group(1)))

    # Fallback for anything else (shouldn't happen with this data)
    return (3, layer_name)

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "results", "wow_signal_table.csv")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, "wow_signal_layers.png")

    # Load data
    df = pd.read_csv(csv_path)

    # Sort data by layer order
    df["sort_key"] = df["Layer"].apply(parse_layer_order)
    df = df.sort_values("sort_key")

    # Create plot with improved readability
    # Use a wider aspect ratio but keeping it compact
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    # Create bar plot with error bars
    x = np.arange(len(df))
    plt.bar(x, df["Mean_Wow"], yerr=df["Std"], capsize=0, alpha=0.9, color="#4C72B0", width=0.8)

    # Customizing axes with LARGER FONTS
    # select a subset of ticks to avoid clutter
    # Show: ViT start/end, Projector, LLM start/mid/end

    tick_indices = []
    tick_labels = []

    # Simple logic: Show every 5th tick, but ensure specific landmarks are kept
    for i, row in enumerate(df.itertuples()):
        show_tick = False
        layer = row.Layer

        # Always show ModalityProjector
        if "ModalityProjector" in layer:
            show_tick = True

        # Show first and last of blocks
        elif "Block.0" in layer and ("ViT" in layer or "LLM" in layer):
             show_tick = True
        elif "Block.11" in layer and "ViT" in layer: # End of ViT
             show_tick = True
        elif "Block.29" in layer and "LLM" in layer: # End of LLM
             show_tick = True

        # Show some intermediate LLM ticks for context (e.g. 10, 20)
        elif "Block.10" in layer or "Block.20" in layer:
             show_tick = True

        if show_tick:
            tick_indices.append(i)
            # Simplify label
            clean_label = layer.replace("ViT.Block.", "ViT-").replace("LLM.Block.", "LLM-").replace("ModalityProjector", "Projector")
            tick_labels.append(clean_label)

    plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)

    plt.xlabel("Layer", fontsize=14, fontweight="bold")
    plt.ylabel("Wow Signal ($D_M$ diff)", fontsize=14, fontweight="bold")
    plt.title("Wow Signal across Model Hierarchy", fontsize=16, fontweight="bold")

    # Add spans to distinguish sections
    # Find indices for ViT, Projector, LLM
    vit_indices = [i for i, k in enumerate(df["sort_key"]) if k[0] == 0]
    proj_indices = [i for i, k in enumerate(df["sort_key"]) if k[0] == 1]
    llm_indices = [i for i, k in enumerate(df["sort_key"]) if k[0] == 2]

    if vit_indices:
        plt.axvspan(min(vit_indices)-0.5, max(vit_indices)+0.5, color='green', alpha=0.1, label='Vision')
    if proj_indices:
        plt.axvspan(min(proj_indices)-0.5, max(proj_indices)+0.5, color='orange', alpha=0.1, label='Projector')
    if llm_indices:
        plt.axvspan(min(llm_indices)-0.5, max(llm_indices)+0.5, color='blue', alpha=0.1, label='LLM')

    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
