"""
Standalone script to generate a compact metrics comparison plot from cached data.
This avoids re-running the full analysis pipeline.
"""
import json
import matplotlib.pyplot as plt
import os

def main():
    # Load cached metrics data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "results", "metrics_comparison.json")

    if not os.path.exists(json_path):
        print(f"Error: Cached data not found at {json_path}")
        print("Please run 'python analysis/run_analysis.py --plots' first to generate the data.")
        return

    with open(json_path, 'r') as f:
        results = f.read()
        # The JSON contains numpy arrays serialized as lists, need to handle that
        # But for plotting we just need the lists
        results = json.loads(results)

    if not results:
        print("No results found in cached data.")
        return

    # Setup figure for roughly half-page (single column in 2-col layout)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    common_layers = results[0]['layers']
    x = range(len(common_layers))

    # Use distinct style
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, res in enumerate(results):
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]
        lbl = res['label']

        # Thicker lines, larger markers
        axes[0].plot(x, res['cka'], marker=m, markersize=6, linewidth=2, color=c, label=lbl, alpha=0.8)
        axes[1].plot(x, res['const'], marker=m, markersize=6, linewidth=2, color=c, label=lbl, alpha=0.8)
        axes[2].plot(x, res['proc'], marker=m, markersize=6, linewidth=2, color=c, label=lbl, alpha=0.8)

    # Styling function
    def style_ax(ax, title, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=10)

    style_ax(axes[0], "CKA Similarity", "CKA")
    axes[0].set_ylim(0, 1.1)

    style_ax(axes[1], "Representational Consistency", "Spearman ρ")
    axes[1].set_ylim(0, 1.1)

    style_ax(axes[2], "Procrustes Distance", "Disparity")

    # Sparse X-Ticks Logic
    tick_indices = []
    tick_labels = []
    for i, layer in enumerate(common_layers):
        show_tick = False
        if "ModalityProjector" in layer: show_tick = True
        elif "ViT.Block.0" in layer or "ViT.Block.11" in layer: show_tick = True
        elif "LLM.Block.0" in layer or "LLM.Block.29" in layer: show_tick = True
        elif "LLM.Block.15" in layer: show_tick = True

        if show_tick:
            tick_indices.append(i)
            clean_label = layer.replace("ViT.Block.", "ViT-").replace("LLM.Block.", "LLM-").replace("ModalityProjector", "Proj")
            tick_labels.append(clean_label)

    plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=11, fontweight='bold')
    plt.xlabel("Layer", fontsize=12, fontweight='bold')

    # Global Legend at Bottom - VERTICAL STACK (ncol=1)
    handles, labels = axes[0].get_legend_handles_labels()
    # Adjust layout to make room for bottom legend
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=1, fontsize=11, frameon=False)

    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "combined_metrics_comparison_small.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved small combined metrics plot to {save_path}")

if __name__ == "__main__":
    main()
