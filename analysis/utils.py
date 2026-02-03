import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
from tqdm.auto import tqdm
from functools import partial
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

from models.vision_language_model import VisionLanguageModel

# Import validation set constants from shared dataset module
from data.color_shape import VAL_COLORS, VAL_SHAPES, draw_shape

# Keep local reference for backward compatibility
COLORS = VAL_COLORS
SHAPES = VAL_SHAPES


# =========================================================
# 1. Stimuli Generation
# =========================================================

def generate_stimuli(num_colors=8, num_shapes=8, img_size=224):
    """
    Generates 64 synthetic images (8 colors x 8 shapes).
    Returns a list of dictionaries: {'image': PIL.Image, 'color': str, 'shape': str}
    """
    # Sort keys to ensure deterministic order
    color_keys = sorted(list(COLORS.keys()))
    shape_keys = sorted(SHAPES)

    stimuli = []

    for c_name in color_keys:
        for s_name in shape_keys:
            # Create image
            img = Image.new('RGB', (img_size, img_size), color=(255, 255, 255)) # White background
            draw = ImageDraw.Draw(img)

            c_val = COLORS[c_name]

            # Draw shape logic (simplified)
            cx, cy = img_size // 2, img_size // 2
            r = img_size // 3 # Radius/Size

            if s_name == 'circle':
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=c_val)
            elif s_name == 'square':
                draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=c_val)
            elif s_name == 'triangle':
                # Equilateral triangle pointing up
                points = [(cx, cy-r), (cx-r*0.866, cy+r*0.5), (cx+r*0.866, cy+r*0.5)]
                draw.polygon(points, fill=c_val)
            elif s_name == 'pentagon':
                 # 5 points, pointing up
                 # Start from -pi/2 (top)
                 angle = np.linspace(-np.pi/2, 2*np.pi - np.pi/2, 6)[:-1]
                 points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angle]
                 draw.polygon(points, fill=c_val)
            elif s_name == 'diamond':
                points = [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)]
                draw.polygon(points, fill=c_val)
            elif s_name == 'cross':
                w = r // 2
                draw.rectangle([cx-w, cy-r, cx+w, cy+r], fill=c_val)
                draw.rectangle([cx-r, cy-w, cx+r, cy+w], fill=c_val)
            elif s_name == 'x':
                # Simplified X using lines with thickness or polygon
                w = r // 3
                # Not perfect X but distinct
                points1 = [(cx-r, cy-r+w), (cx-r+w, cy-r), (cx+r, cy+r-w), (cx+r-w, cy+r)]
                points2 = [(cx+r, cy-r+w), (cx+r-w, cy-r), (cx-r, cy+r-w), (cx-r+w, cy+r)]
                draw.polygon(points1, fill=c_val)
                draw.polygon(points2, fill=c_val)
            elif s_name == 'star':
                # Simple 4-point star approximation
                draw.polygon([(cx, cy-r), (cx+r//3, cy), (cx, cy+r), (cx-r//3, cy)], fill=c_val)
                draw.polygon([(cx-r, cy), (cx, cy-r//3), (cx+r, cy), (cx, cy+r//3)], fill=c_val)
            elif s_name == 'hexagon':
                # 6 points
                angle = np.linspace(0, 2*np.pi, 7)[:-1]
                points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angle]
                draw.polygon(points, fill=c_val)

            stimuli.append({
                'image': img,
                'color': c_name,
                'shape': s_name,
                'label': f"{c_name} {s_name}"
            })

    return stimuli

def generate_color_only_stimuli(img_size=224):
    """
    Generates solid color images (no shapes) for testing color recognition.
    Returns a list of dictionaries: {'image': PIL.Image, 'color': str, 'label': str}
    """
    color_keys = sorted(list(COLORS.keys()))
    stimuli = []

    for c_name in color_keys:
        # Create solid color image
        img = Image.new('RGB', (img_size, img_size), color=COLORS[c_name])

        stimuli.append({
            'image': img,
            'color': c_name,
            'label': c_name
        })

    return stimuli

# =========================================================
# 2. Model Loading & Activation Extraction
# =========================================================

class ActivationHook:
    def __init__(self):
        self.activations = []

    def __call__(self, module, input, output):
        # We want the output.
        # For ViT/LLM blocks, output is usually a tensor [Batch, Seq, Dim]
        # or a tuple. If tuple, usually first element is hidden states.

        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output

        # Move to CPU to save GPU memory
        self.activations.append(act.detach().cpu())

    def clear(self):
        self.activations = []

def get_layer_names(model):
    """
    Returns a list of readable layer names we want to hook.
    """
    layers = []

    # ViT Blocks (~12)
    for i in range(len(model.vision_encoder.blocks)):
        layers.append(f"ViT.Block.{i}")

    # Modality Projector (1)
    layers.append("ModalityProjector")

    # LLM Blocks (~30)
    for i in range(len(model.decoder.blocks)):
        layers.append(f"LLM.Block.{i}")

    return layers

def extract_activations(model, stimuli, device='cuda', use_grayscale_input=False):
    """
    Runs stimuli through the model and extracts averaged activations for each layer.
    Returns: Dict[layer_name] -> Tensor of shape (num_stimuli, hidden_dim)
    """
    model.eval()
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.to(device)

    # Hooks
    hooks = {}
    handles = []

    # Register hooks
    # ViT
    for i, block in enumerate(model.vision_encoder.blocks):
        name = f"ViT.Block.{i}"
        hooks[name] = ActivationHook()
        handles.append(block.register_forward_hook(hooks[name]))

    # MP
    hooks["ModalityProjector"] = ActivationHook()
    handles.append(model.MP.register_forward_hook(hooks["ModalityProjector"]))

    # LLM
    for i, block in enumerate(model.decoder.blocks):
        name = f"LLM.Block.{i}"
        hooks[name] = ActivationHook()
        handles.append(block.register_forward_hook(hooks[name]))

    # Run Inference
    layer_outputs = {name: [] for name in hooks}

    # Dummy input_ids
    input_ids = torch.tensor([[1]]).to(device) # Shape [1, 1]

    with torch.no_grad():
        for item in tqdm(stimuli, desc="Extracting Activations"):
            img = item['image']

            if use_grayscale_input:
                img = img.convert('L').convert('RGB')

            # Preprocess image
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            img_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, H, W]

            # Forward
            model(input_ids, img_tensor)

            # Collect and Aggregate
            for name, hook in hooks.items():
                act = hook.activations[-1]
                # Average pooling over sequence dimension
                pool_act = act.mean(dim=1).squeeze(0) # [Dim]
                layer_outputs[name].append(pool_act)

            # Clear hooks for next batch
            for hook in hooks.values():
                hook.clear()

    # Stack
    final_outputs = {}
    for name, act_list in layer_outputs.items():
        # [NumStimuli, Dim]
        final_outputs[name] = torch.stack(act_list)

    # Process last block output to get Logits
    # (assuming the last hook captured the final hidden state)
    # The last block in SmolLM2 (135M) is likely 'model.layers.29' or similar.
    # We'll just take the lat captured "LLM.Block" and apply the head.

    last_llm_layer = None
    for k in sorted(final_outputs.keys()):
        if "LLM.Block" in k:
            last_llm_layer = k

    if last_llm_layer:
        final_hidden = final_outputs[last_llm_layer] # [N, D]
        # We need to apply model.decoder.head. BUT the hook captured flattened [N, D].
        # model.decoder.head expects [..., D].

        with torch.no_grad():
            # Check if we need to apply normalization first?
            # Usually 'head' is just Linear. But usually there is a LayerNorm before the head (norm).
            # LanguageModel forward: x = self.norm(x); x = self.head(x) (if use_tokens)
            # The hook on 'model.outputs.layers[-1]' captures BEFORE final norm (usually).
            # Let's check LanguageModel code structure or just hook 'norm' output too?
            # Creating a transient forward pass through norm and head is safer.

            # Actually, to be precise, let's just run a forward pass invocation for the head
            # assuming 'final_hidden' is the output of the last TransformerBlock.
            # In Llama/SmolLM, the architecture is: layers -> norm -> head.

            # We will try to apply model.decoder.norm then model.decoder.head
            # Accessing internal modules might be tricky if names vary, but 'norm' is standard in our LanguageModel.

            logits = final_hidden.to(device)
            if hasattr(model.decoder, 'norm'):
                logits = model.decoder.norm(logits)

            if hasattr(model.decoder, 'head'):
                logits = model.decoder.head(logits)

            final_outputs['Logits'] = logits.cpu()

    # Cleanup
    for h in handles:
        h.remove()

    return final_outputs

def extract_text_activations(model, tokenizer, texts, device='cuda'):
    """
    Extracts activations for text-only inputs from LLM layers.
    texts: List of strings e.g. ["a red circle", ...]
    """
    model.eval()
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
    model.to(device)

    # Hooks for LLM only
    hooks = {}
    handles = []
    # Identify LLM blocks
    # Note: We assume model.decoder has blocks.
    for i, block in enumerate(model.decoder.blocks):
        name = f"LLM.Block.{i}"
        hooks[name] = ActivationHook()
        handles.append(block.register_forward_hook(hooks[name]))

    layer_outputs = {name: [] for name in hooks}

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting Text Activations"):
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt').to(device)
            # Embed
            token_embd = model.decoder.token_embedding(inputs.input_ids)

            # Pass through Decoder
            # model.decoder is a LanguageModel which usually returns logits
            # and takes (inputs_embeds, attention_mask)
            model.decoder(token_embd, attention_mask=inputs.attention_mask)

            # Collect
            for name, hook in hooks.items():
                act = hook.activations[-1]
                # Average pooling over sequence dimension
                pool_act = act.mean(dim=1).squeeze(0)
                layer_outputs[name].append(pool_act)

            # Clear
            for hook in hooks.values():
                hook.clear()

    # Stack results
    final_outputs = {name: torch.stack(act_list) for name, act_list in layer_outputs.items()}

    for h in handles: h.remove()

    return final_outputs

# =========================================================
# 3. Analysis: CKA & MDS
# =========================================================

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def linear_cka(X, Y):
    """
    Computes Linear CKA between X and Y.
    X, Y: Tensor or Numpy array [N, D]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(X, np.ndarray): X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y)

    X = X.to(device)
    Y = Y.to(device)

    # Linear Kernel K = XX^T
    GramX = torch.matmul(X, X.t())
    GramY = torch.matmul(Y, Y.t())

    GramX = centering(GramX)
    GramY = centering(GramY)

    scaled_hsic = torch.sum(GramX * GramY)
    norm_x = torch.sqrt(torch.sum(GramX * GramX))
    norm_y = torch.sqrt(torch.sum(GramY * GramY))

    return (scaled_hsic / (norm_x * norm_y)).item()

def compute_mds(activations_dict, n_components=2):
    """
    Computes MDS for each layer using sklearn.
    activations_dict: {layer_name: Tensor [N, Dim]}
    Returns: {layer_name: Tensor [N, n_components]}
    """
    results = {}
    mds = MDS(n_components=n_components, dissimilarity='euclidean', random_state=42, normalized_stress='auto')

    for name, acts in activations_dict.items():
        # acts is tensor, convert to numpy
        X = acts.numpy()
        res = mds.fit_transform(X)
        results[name] = res

    return results

def compute_rdm(X, metric='correlation'):
    """
    Computes the Representational Dissimilarity Matrix (RDM).
    X: [N_samples, N_features]
    metric: distance metric (default: correlation)
    Returns: RDM [N_samples, N_samples]
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    # pdist returns condensed distance matrix (upper triangle)
    dists = pdist(X, metric=metric)
    # squareform converts to square matrix
    rdm = squareform(dists)
    return rdm

def representational_consistency(rdm1, rdm2):
    """
    Computes Representational Consistency as the Spearman correlation
    between the upper triangles of two RDMs.
    """
    # Get upper triangle indices
    n = rdm1.shape[0]
    triu_indices = np.triu_indices(n, k=1)

    vec1 = rdm1[triu_indices]
    vec2 = rdm2[triu_indices]

    corr, _ = spearmanr(vec1, vec2)
    return corr

def compute_procrustes(X, Y):
    """
    Computes Procrustes Disparity between two point clouds.
    X, Y: [N, D]
    Returns: disparity (lower is more similar)
    """
    if isinstance(X, torch.Tensor): X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor): Y = Y.cpu().numpy()

    # Check dimensions to avoid LAPACK overflow (e.g. Logits ~50k dim)
    # If D is very large, Procrustes (which computes SVD of DxD matrix) is prohibitively expensive/impossible.
    if X.shape[1] > 10000:
        # Fallback: Compute on PCA-reduced data? or just return NaN
        # Calculating Procrustes on 50k dims for 64 points is not very meaningful geometrically anyway without reduction.
        # Let's return NaN (or 1.0) and warn.
        # Or, we could reduce to N-1 dims. Procrustes is about aligning the point clouds.
        return np.nan

    # Procrustes aligns Y to X (translation, scaling, rotation)
    # mtx1, mtx2, disparity = procrustes(data1, data2)
    # returns disparity: sum of squared errors between standardized data
    try:
        _, _, disparity = procrustes(X, Y)
        return disparity
    except Exception as e:
        print(f"Procrustes failed: {e}")
        return np.nan


# =========================================================
# 4. Mahalanobis Distance & "Wow" Signal
# =========================================================

def compute_mahalanobis(z, mean, cov_inv):
    """
    Computes Mahalanobis distance for a single vector or batch.

    Args:
        z: [D] or [N, D] - embedding vector(s)
        mean: [D] - mean of the reference distribution
        cov_inv: [D, D] - inverse covariance matrix

    Returns:
        Mahalanobis distance (scalar or [N] array)
    """
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(cov_inv, torch.Tensor):
        cov_inv = cov_inv.cpu().numpy()

    # Handle single vector
    if z.ndim == 1:
        diff = z - mean
        return np.sqrt(np.dot(np.dot(diff, cov_inv), diff))

    # Handle batch
    diff = z - mean  # [N, D]
    # D_M^2 = (z - mu)^T @ Sigma^-1 @ (z - mu)
    left = np.dot(diff, cov_inv)  # [N, D]
    mahal_sq = np.sum(left * diff, axis=1)  # [N]
    return np.sqrt(mahal_sq)


def fit_gaussian(embeddings):
    """
    Fit a multivariate Gaussian to embeddings.

    Args:
        embeddings: [N, D] tensor or array

    Returns:
        mean: [D] array
        cov_inv: [D, D] inverse covariance matrix
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    mean = np.mean(embeddings, axis=0)
    # Use regularized covariance to avoid singularity
    cov = np.cov(embeddings, rowvar=False)
    # Add small regularization for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    return mean, cov_inv


def compute_wow_signal(z_chromatic, z_achromatic, mean_achromatic, cov_inv):
    """
    Computes the "Wow" Signal as defined in the paper.

    The Wow Signal measures the novelty of chromatic input relative to
    achromatic baseline: S = D_M(z_c) - D_M(z_g)

    A positive S indicates the chromatic input is treated as a
    "Violation of Expectation" (VoE) by the model.

    Args:
        z_chromatic: [N, D] - chromatic (RGB) embeddings
        z_achromatic: [N, D] - achromatic (grayscale) embeddings
        mean_achromatic: [D] - mean of achromatic distribution
        cov_inv: [D, D] - inverse covariance of achromatic distribution

    Returns:
        wow_signals: [N] array of per-stimulus Wow signals
        mean_wow: scalar mean Wow signal
        std_wow: scalar std of Wow signals
    """
    d_m_chromatic = compute_mahalanobis(z_chromatic, mean_achromatic, cov_inv)
    d_m_achromatic = compute_mahalanobis(z_achromatic, mean_achromatic, cov_inv)

    wow_signals = d_m_chromatic - d_m_achromatic

    return wow_signals, np.mean(wow_signals), np.std(wow_signals)
