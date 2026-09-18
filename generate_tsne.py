import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from lib.networks_hup import EMCADNet
from utils.dataloader_hup import _load_image_any, _load_mask_any, generate_hupanno

def setup_matplotlib_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.autolayout": True
    })

def load_data_with_labels(image_path, mask_path):
    img_pil = _load_image_any(image_path)
    img = np.array(img_pil)
    img = cv2.resize(img, (352, 352))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    mask_pil = _load_mask_any(mask_path)
    mask = np.array(mask_pil)
    mask = cv2.resize(mask, (352, 352))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_np = (mask > 127).astype(np.float32)
    
    # Generate ground truth multi-class labels for t-SNE
    _, _, _, _, _, _, _, y_c = generate_hupanno(mask_np, K=4)
    return img, mask, y_c

def generate_tsne_visualization(model, img, y_c, save_dir):
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        # Get embeddings from the train mode output
        # preds, cls_logits, embeddings, conf_map
        _, _, embeddings, _ = model(img_tensor, mode='train')
    
    emb = embeddings[0].cpu().numpy() # Shape: (C, H, W)
    C, H, W = emb.shape
    emb_flat = emb.reshape(C, -1).T # Shape: (H*W, C)
    y_c_flat = y_c.reshape(-1)      # Shape: (H*W,)
    
    # Class mappings based on HUPAnno:
    # 0: Background
    # 2: LRP Uncertain / Boundary / Middle
    # 3: Lesion (Foreground)
    
    bg_idx = np.where(y_c_flat == 0)[0]
    mid_idx = np.where(y_c_flat == 2)[0]
    fg_idx = np.where(y_c_flat == 3)[0]
    
    # Randomly sample points to keep t-SNE tractable (e.g., 500 per class)
    n_samples = 500
    
    if len(bg_idx) > n_samples: bg_idx = np.random.choice(bg_idx, n_samples, replace=False)
    if len(mid_idx) > n_samples: mid_idx = np.random.choice(mid_idx, n_samples, replace=False)
    if len(fg_idx) > n_samples: fg_idx = np.random.choice(fg_idx, n_samples, replace=False)
    
    # Prepare data for Plot A (Lesion vs Background)
    idx_a = np.concatenate([fg_idx, bg_idx])
    X_a = emb_flat[idx_a]
    y_a = y_c_flat[idx_a]
    
    # Prepare data for Plot B (Lesion vs Middle vs Background)
    idx_b = np.concatenate([fg_idx, mid_idx, bg_idx])
    X_b = emb_flat[idx_b]
    y_b = y_c_flat[idx_b]
    
    print("Running t-SNE for Plot A (Lesion vs Background)...")
    tsne_a = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_a_2d = tsne_a.fit_transform(X_a)
    
    print("Running t-SNE for Plot B (including Boundary)...")
    tsne_b = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_b_2d = tsne_b.fit_transform(X_b)
    
    # PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Custom color palette matching the uploaded paper
    color_lesion = '#66c2a5'     # Teal/Green
    color_bg = '#fc8d62'         # Orange/Coral
    color_mid = '#8da0cb'        # Purple/Blue
    
    # Plot A
    sns.scatterplot(
        x=X_a_2d[:, 0], y=X_a_2d[:, 1],
        hue=y_a,
        palette={3: color_lesion, 0: color_bg},
        edgecolor="white", s=50, ax=axes[0], alpha=0.9
    )
    axes[0].set_title("(a) Feature Embeddings (Lesion vs Background)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, ['Background', 'Lesion'], loc='upper left')
    
    # Plot B
    sns.scatterplot(
        x=X_b_2d[:, 0], y=X_b_2d[:, 1],
        hue=y_b,
        palette={3: color_lesion, 2: color_mid, 0: color_bg},
        edgecolor="white", s=50, ax=axes[1], alpha=0.9
    )
    axes[1].set_title("(b) Feature Embeddings (including Boundary)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, ['Background', 'Boundary (Middle)', 'Lesion'], loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tsne_visualizations.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved t-SNE visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained .pth model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to a sample test image')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the sample ground truth mask')
    parser.add_argument('--save_dir', type=str, default='./tmi_visualizations')
    opt = parser.parse_args()
    
    os.makedirs(opt.save_dir, exist_ok=True)
    setup_matplotlib_style()
    
    img, mask, y_c = load_data_with_labels(opt.image_path, opt.mask_path)
    
    model = EMCADNet(use_morph=True, morph_sample_k=4, pretrain=False)
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=False)
    model.eval()
    
    generate_tsne_visualization(model, img, y_c, opt.save_dir)

if __name__ == '__main__':
    main()
