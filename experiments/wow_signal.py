"""
WowSignalExperiment: Measures the "Wow" signal (Mismatch Negativity in Latent Space)

This experiment operationalizes the neural signature of novelty as Mahalanobis Distance.
It tests the Impenetrable Representation Hypothesis by measuring the "subjective shock"
when a grayscale-trained model encounters chromatic stimuli.

From the paper:
    S = D_M(z_c) - D_M(z_g)

A significant positive S indicates that "Redness" is not treated as just another feature,
but as a Violation of Expectation (VoE) regarding the fundamental format of the input.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats
import wandb

from experiments.experiment import Experiment, with_wandb
from models.vision_language_model import VisionLanguageModel
from analysis.utils import (
    generate_stimuli,
    extract_activations,
    fit_gaussian,
    compute_wow_signal,
)


@dataclass
class WowSignalExperiment(Experiment):
    """
    Measures the "Wow" Signal: novelty detection via Mahalanobis distance.

    This experiment:
    1. Loads a grayscale-trained checkpoint
    2. Generates synthetic color/shape stimuli
    3. Extracts Modality Projector embeddings for both grayscale and RGB versions
    4. Fits a Gaussian on achromatic embeddings
    5. Computes D_M for both conditions
    6. Reports the Wow Signal: S = D_M(chromatic) - D_M(achromatic)
    """

    # Required: Path to grayscale-trained checkpoint
    vlm_checkpoint_path: str = "output/maryVLM_bs256_gray_ms0_ds0/checkpoint-22000"

    # Analysis configuration
    layer_to_analyze: str = "ModalityProjector"  # Could also use specific LLM blocks
    num_colors: int = 8
    num_shapes: int = 8
    img_size: int = 224

    # Logging
    wandb_project: str = "maryVLM_analysis"
    wandb_entity: str = "llm-lg"
    log_wandb: bool = True

    @property
    def run_name(self) -> str:
        return f"wow_signal_{self.layer_to_analyze}"

    @with_wandb
    def run(self):
        print("=" * 60)
        print("WowSignalExperiment: Measuring Mismatch Negativity")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 1. Load Model
        print(f"\nLoading grayscale-trained model from: {self.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(self.vlm_checkpoint_path)
        model.to(device)
        model.eval()

        # 2. Generate Stimuli
        print(f"\nGenerating {self.num_colors}x{self.num_shapes} synthetic stimuli...")
        stimuli = generate_stimuli(
            num_colors=self.num_colors,
            num_shapes=self.num_shapes,
            img_size=self.img_size
        )
        print(f"Generated {len(stimuli)} stimuli")

        # 3. Extract Achromatic (Grayscale) Embeddings
        print("\nExtracting ACHROMATIC embeddings (grayscale input)...")
        achromatic_acts = extract_activations(
            model, stimuli, device=str(device), use_grayscale_input=True
        )

        # 4. Extract Chromatic (RGB) Embeddings
        print("\nExtracting CHROMATIC embeddings (RGB input)...")
        chromatic_acts = extract_activations(
            model, stimuli, device=str(device), use_grayscale_input=False
        )

        # 5. Get embeddings for the target layer
        if self.layer_to_analyze not in achromatic_acts:
            available = list(achromatic_acts.keys())
            raise ValueError(
                f"Layer '{self.layer_to_analyze}' not found. Available: {available}"
            )

        z_achromatic = achromatic_acts[self.layer_to_analyze]  # [N, D]
        z_chromatic = chromatic_acts[self.layer_to_analyze]    # [N, D]

        print(f"\nAnalyzing layer: {self.layer_to_analyze}")
        print(f"Embedding shape: {z_achromatic.shape}")

        # 6. Fit Gaussian on achromatic distribution
        print("\nFitting Gaussian on achromatic embeddings...")
        mean_achromatic, cov_inv = fit_gaussian(z_achromatic)

        # 7. Compute Wow Signal
        print("\nComputing Wow Signal: S = D_M(z_chromatic) - D_M(z_achromatic)")
        wow_signals, mean_wow, std_wow = compute_wow_signal(
            z_chromatic, z_achromatic, mean_achromatic, cov_inv
        )

        # 8. Statistical Test
        # One-sample t-test: Is the mean Wow signal significantly > 0?
        t_stat, p_value = stats.ttest_1samp(wow_signals, 0)
        # Use one-tailed p-value (we expect positive Wow signal)
        p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        # 9. Report Results
        print("\n" + "=" * 60)
        print("RESULTS: Wow Signal (Mismatch Negativity)")
        print("=" * 60)
        print(f"Layer analyzed: {self.layer_to_analyze}")
        print(f"Number of stimuli: {len(stimuli)}")
        print(f"Mean Wow Signal: {mean_wow:.4f} ± {std_wow:.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value (one-tailed): {p_value_one_tailed:.6f}")

        significance = "***" if p_value_one_tailed < 0.001 else (
            "**" if p_value_one_tailed < 0.01 else (
                "*" if p_value_one_tailed < 0.05 else "n.s."
            )
        )
        print(f"Significance: {significance}")

        # Interpretation
        if mean_wow > 0 and p_value_one_tailed < 0.05:
            print("\n✓ HYPOTHESIS SUPPORTED: Chromatic stimuli trigger significantly")
            print("  higher Mahalanobis distance than achromatic controls.")
            print("  This is consistent with a 'Violation of Expectation' (VoE).")
        else:
            print("\n✗ HYPOTHESIS NOT SUPPORTED: No significant difference detected.")

        # 10. Log to WandB
        if wandb.run is not None:
            wandb.log({
                "wow_signal/mean": mean_wow,
                "wow_signal/std": std_wow,
                "wow_signal/t_statistic": t_stat,
                "wow_signal/p_value": p_value_one_tailed,
                "wow_signal/layer": self.layer_to_analyze,
                "wow_signal/n_stimuli": len(stimuli),
            })

            # Log per-stimulus results
            for i, (stim, wow) in enumerate(zip(stimuli, wow_signals)):
                wandb.log({
                    f"wow_signal/per_stimulus/{stim['label']}": wow,
                })

        # Return results for programmatic access
        return {
            "mean_wow": mean_wow,
            "std_wow": std_wow,
            "t_statistic": t_stat,
            "p_value": p_value_one_tailed,
            "wow_signals": wow_signals,
            "significance": significance,
        }


@dataclass
class WowSignalAllLayers(Experiment):
    """
    Runs WowSignalExperiment across all model layers to find where
    the "Wow" signal is strongest (analogous to finding the locus of
    qualia emergence in the processing hierarchy).
    """

    vlm_checkpoint_path: str = "output/maryVLM_bs256_gray_ms0_ds0/checkpoint-22000"
    wandb_project: str = "maryVLM_analysis"
    log_wandb: bool = False  # We'll log aggregate results

    def run(self):
        print("=" * 60)
        print("WowSignalAllLayers: Finding Peak Mismatch Negativity")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model once
        model = VisionLanguageModel.from_pretrained(self.vlm_checkpoint_path)
        model.to(device)
        model.eval()

        # Generate stimuli once
        stimuli = generate_stimuli()

        # Extract all activations
        print("\nExtracting activations for all layers...")
        achromatic_acts = extract_activations(model, stimuli, str(device), use_grayscale_input=True)
        chromatic_acts = extract_activations(model, stimuli, str(device), use_grayscale_input=False)

        results = []

        for layer_name in achromatic_acts.keys():
            if layer_name == "Logits":
                continue  # Skip logits layer

            z_a = achromatic_acts[layer_name]
            z_c = chromatic_acts[layer_name]

            mean_a, cov_inv = fit_gaussian(z_a)
            wow_signals, mean_wow, std_wow = compute_wow_signal(z_c, z_a, mean_a, cov_inv)

            t_stat, p_value = stats.ttest_1samp(wow_signals, 0)
            p_one = p_value / 2 if t_stat > 0 else 1 - p_value / 2

            results.append({
                "layer": layer_name,
                "mean_wow": mean_wow,
                "std_wow": std_wow,
                "t_stat": t_stat,
                "p_value": p_one,
            })

        # Sort by mean Wow signal (descending)
        results.sort(key=lambda x: x["mean_wow"], reverse=True)

        print("\n" + "=" * 60)
        print("RESULTS: Wow Signal by Layer (sorted by magnitude)")
        print("=" * 60)
        print(f"{'Layer':<25} {'Mean Wow':>10} {'Std':>10} {'p-value':>12} {'Sig':>5}")
        print("-" * 60)

        for r in results:
            sig = "***" if r["p_value"] < 0.001 else (
                "**" if r["p_value"] < 0.01 else (
                    "*" if r["p_value"] < 0.05 else ""
                )
            )
            print(f"{r['layer']:<25} {r['mean_wow']:>10.4f} {r['std_wow']:>10.4f} {r['p_value']:>12.6f} {sig:>5}")

        return results
