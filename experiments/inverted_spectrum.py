"""
InvertedSpectrumExperiment: Tests Subjective Specificity via Procrustes Analysis

This experiment tests whether functionally equivalent agents can have structurally
different internal representations - a computational analog of the philosophical
"Inverted Spectrum" thought experiment.

From the paper:
    We trained two identical MaryVLM agents (M_A and M_B) with different random seeds.
    High functional equivalence (identical verbal outputs) combined with high Procrustes
    disparity (misaligned internal vectors) would support the hypothesis that subjective
    experience is structurally real but implementation-dependent.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import wandb

from experiments.experiment import Experiment, with_wandb
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer
from analysis.utils import (
    generate_color_only_stimuli,
    extract_activations,
    compute_procrustes,
    linear_cka,
    COLORS,
)


@dataclass
class InvertedSpectrumExperiment(Experiment):
    """
    Tests Subjective Specificity via Procrustes analysis between two models.

    This experiment:
    1. Loads two checkpoints trained with different random seeds
    2. Verifies functional equivalence (both trained successfully)
    3. Generates color stimuli (solid color patches)
    4. Extracts embeddings and computes color concept centroids
    5. Runs Procrustes analysis to measure structural alignment
    6. Reports: disparity (low = aligned, high = "inverted spectrum")
    """

    # Model A checkpoint (e.g., seed=0)
    model_a_checkpoint: str = "output/maryVLM_bs256_rgb_ms0_ds0/checkpoint-22000"
    model_a_name: str = "Model_A (seed=0)"

    # Model B checkpoint (e.g., seed=1)
    model_b_checkpoint: str = "output/maryVLM_bs256_rgb_ms1_ds1/checkpoint-22000"
    model_b_name: str = "Model_B (seed=1)"

    # Analysis configuration
    layer_to_analyze: str = "ModalityProjector"
    img_size: int = 224

    # Logging
    wandb_project: str = "maryVLM_analysis"
    wandb_entity: str = "llm-lg"
    log_wandb: bool = True

    @property
    def run_name(self) -> str:
        return f"inverted_spectrum_{self.layer_to_analyze}"

    def _load_model(self, checkpoint_path: str, device: torch.device) -> VisionLanguageModel:
        """Load a model from checkpoint."""
        print(f"Loading model from: {checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()
        return model

    def _compute_color_centroids(
        self,
        activations: Dict[str, torch.Tensor],
        stimuli: List[Dict],
        layer: str
    ) -> Dict[str, np.ndarray]:
        """
        Compute centroid embedding for each color.
        Since each color appears once in color_only_stimuli,
        the centroid is just the embedding itself.
        """
        embeddings = activations[layer]

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        centroids = {}
        for i, stim in enumerate(stimuli):
            color = stim['color']
            centroids[color] = embeddings[i]

        return centroids

    def _centroids_to_matrix(self, centroids: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert centroids dict to ordered matrix for Procrustes."""
        # Sort by color name for consistent ordering
        colors = sorted(centroids.keys())
        return np.stack([centroids[c] for c in colors])

    @with_wandb
    def run(self):
        print("=" * 60)
        print("InvertedSpectrumExperiment: Testing Subjective Specificity")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 1. Load Both Models
        print(f"\n--- Loading Models ---")
        model_a = self._load_model(self.model_a_checkpoint, device)
        model_b = self._load_model(self.model_b_checkpoint, device)

        # 2. Generate Color Stimuli
        print(f"\n--- Generating Color Stimuli ---")
        stimuli = generate_color_only_stimuli(img_size=self.img_size)
        print(f"Generated {len(stimuli)} color stimuli: {[s['color'] for s in stimuli]}")

        # 3. Extract Activations
        print(f"\n--- Extracting Activations ---")
        print(f"Extracting from {self.model_a_name}...")
        acts_a = extract_activations(model_a, stimuli, device=str(device))

        print(f"Extracting from {self.model_b_name}...")
        acts_b = extract_activations(model_b, stimuli, device=str(device))

        # 4. Validate layer exists
        if self.layer_to_analyze not in acts_a:
            available = list(acts_a.keys())
            raise ValueError(
                f"Layer '{self.layer_to_analyze}' not found. Available: {available}"
            )

        # 5. Compute Color Centroids
        print(f"\n--- Computing Color Centroids ---")
        centroids_a = self._compute_color_centroids(acts_a, stimuli, self.layer_to_analyze)
        centroids_b = self._compute_color_centroids(acts_b, stimuli, self.layer_to_analyze)

        # Convert to matrices
        matrix_a = self._centroids_to_matrix(centroids_a)
        matrix_b = self._centroids_to_matrix(centroids_b)

        print(f"Centroid matrices shape: {matrix_a.shape}")

        # 6. Compute Procrustes Disparity
        print(f"\n--- Computing Procrustes Disparity ---")
        disparity = compute_procrustes(matrix_a, matrix_b)

        # 7. Compute CKA for comparison
        print(f"\n--- Computing Linear CKA ---")
        cka_score = linear_cka(matrix_a, matrix_b)

        # 8. Report Results
        print("\n" + "=" * 60)
        print("RESULTS: Inverted Spectrum Analysis")
        print("=" * 60)
        print(f"Layer analyzed: {self.layer_to_analyze}")
        print(f"Number of colors: {len(stimuli)}")
        print(f"\n{self.model_a_name}: {self.model_a_checkpoint}")
        print(f"{self.model_b_name}: {self.model_b_checkpoint}")
        print(f"\nProcrustes Disparity: {disparity:.6f}")
        print(f"  (0 = perfectly aligned, higher = more misaligned)")
        print(f"\nLinear CKA: {cka_score:.6f}")
        print(f"  (1 = identical representations, 0 = orthogonal)")

        # Interpretation
        print("\n--- Interpretation ---")
        if disparity > 0.1 and cka_score < 0.9:
            print("✓ INVERTED SPECTRUM DETECTED: Models show significant")
            print("  structural disparity despite (presumed) functional equivalence.")
            print("  This supports the hypothesis that subjective experience is")
            print("  structurally real but implementation-dependent.")
        elif disparity < 0.05:
            print("✗ LOW DISPARITY: Models have highly aligned representations.")
            print("  This suggests similar 'subjective' encoding of colors.")
        else:
            print("~ MODERATE DISPARITY: Some structural differences detected.")
            print("  Further investigation recommended.")

        # 9. Per-color analysis
        print("\n--- Per-Color Centroid Distances ---")
        print(f"{'Color':<15} {'L2 Distance':>15}")
        print("-" * 30)

        colors = sorted(centroids_a.keys())
        per_color_distances = {}
        for color in colors:
            dist = np.linalg.norm(centroids_a[color] - centroids_b[color])
            per_color_distances[color] = dist
            print(f"{color:<15} {dist:>15.4f}")

        # 10. Log to WandB
        if wandb.run is not None:
            wandb.log({
                "inverted_spectrum/procrustes_disparity": disparity,
                "inverted_spectrum/linear_cka": cka_score,
                "inverted_spectrum/layer": self.layer_to_analyze,
                "inverted_spectrum/n_colors": len(stimuli),
            })

            for color, dist in per_color_distances.items():
                wandb.log({
                    f"inverted_spectrum/per_color/{color}": dist,
                })

        # Return results
        return {
            "procrustes_disparity": disparity,
            "linear_cka": cka_score,
            "per_color_distances": per_color_distances,
            "layer": self.layer_to_analyze,
        }


@dataclass
class InvertedSpectrumAllLayers(Experiment):
    """
    Runs InvertedSpectrumExperiment across all layers to find where
    representations diverge most between differently-seeded models.
    """

    model_a_checkpoint: str = "output/maryVLM_bs256_rgb_ms0_ds0/checkpoint-22000"
    model_b_checkpoint: str = "output/maryVLM_bs256_rgb_ms1_ds1/checkpoint-22000"
    wandb_project: str = "maryVLM_analysis"
    log_wandb: bool = False

    def run(self):
        print("=" * 60)
        print("InvertedSpectrumAllLayers: Layer-wise Representation Divergence")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        model_a = VisionLanguageModel.from_pretrained(self.model_a_checkpoint)
        model_b = VisionLanguageModel.from_pretrained(self.model_b_checkpoint)
        model_a.to(device).eval()
        model_b.to(device).eval()

        # Generate stimuli
        stimuli = generate_color_only_stimuli()

        # Extract all activations
        print("\nExtracting activations...")
        acts_a = extract_activations(model_a, stimuli, str(device))
        acts_b = extract_activations(model_b, stimuli, str(device))

        results = []

        for layer_name in acts_a.keys():
            if layer_name == "Logits":
                continue

            emb_a = acts_a[layer_name]
            emb_b = acts_b[layer_name]

            if isinstance(emb_a, torch.Tensor):
                emb_a = emb_a.cpu().numpy()
            if isinstance(emb_b, torch.Tensor):
                emb_b = emb_b.cpu().numpy()

            disparity = compute_procrustes(emb_a, emb_b)
            cka = linear_cka(emb_a, emb_b)

            results.append({
                "layer": layer_name,
                "procrustes": disparity,
                "cka": cka,
            })

        # Sort by Procrustes disparity (descending = most divergent first)
        results.sort(key=lambda x: x["procrustes"] if not np.isnan(x["procrustes"]) else -1, reverse=True)

        print("\n" + "=" * 60)
        print("RESULTS: Representation Divergence by Layer")
        print("=" * 60)
        print(f"{'Layer':<25} {'Procrustes':>12} {'CKA':>10}")
        print("-" * 50)

        for r in results:
            proc_str = f"{r['procrustes']:.6f}" if not np.isnan(r['procrustes']) else "N/A"
            print(f"{r['layer']:<25} {proc_str:>12} {r['cka']:>10.4f}")

        return results
