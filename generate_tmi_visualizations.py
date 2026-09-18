import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter

from lib.networks_hup import EMCADNet
from utils.dataloader_hup import _load_image_any, _load_mask_any

# IEEE TMI Professional Style Settings
def setup_matplotlib_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.autolayout": True
    })

def create_dummy_data():
    img = np.zeros((352, 352, 3), dtype=np.uint8)
    cv2.circle(img, (176, 176), 80, (150, 100, 200), -1)
    mask = np.zeros((352, 352), dtype=np.uint8)
    cv2.circle(mask, (176, 176), 80, 255, -1)
    # Add some noise to contour
    for i in range(352):
        for j in range(352):
            if 70 < np.sqrt((i-176)**2 + (j-176)**2) < 90:
                if np.random.rand() > 0.5:
                    mask[i, j] = 255
    return img, mask

def load_data(image_path, mask_path):
    if image_path and os.path.exists(image_path) and mask_path and os.path.exists(mask_path):
        img_pil = _load_image_any(image_path)
        img = np.array(img_pil)
        img = cv2.resize(img, (352, 352))
        # RGB to BGR for internal cv2 usage consistency if needed, but img_pil is RGB
        # Our plotting uses cv2.cvtColor(img, cv2.COLOR_BGR2RGB), so we should provide BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        mask_pil = _load_mask_any(mask_path)
        mask = np.array(mask_pil)
        mask = cv2.resize(mask, (352, 352))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        print("Paths not found or not provided, using dummy data for demonstration.")
        img, mask = create_dummy_data()
    return img, mask

def generate_qualitative_comparison(img, mask, pred_mask, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("HUPAnno Prediction")
    axes[2].axis('off')
    
    plt.savefig(os.path.join(save_dir, 'qualitative_comparison.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved qualitative_comparison.pdf")

def generate_lrp_visualization(img, mask, save_dir):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return
    c = max(contours, key=cv2.contourArea)
    c = c.squeeze()
    
    # Calculate geometric curvature heuristically
    step = 5
    curvatures = []
    points = []
    for i in range(len(c)):
        p1 = c[i - step] if i - step >= 0 else c[i - step + len(c)]
        p2 = c[i]
        p3 = c[(i + step) % len(c)]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            curvatures.append(0)
        else:
            cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
            kappa = 1 - max(min(cosine_angle, 1.0), -1.0)
            curvatures.append(kappa)
        points.append(p2)
        
    curvatures = savgol_filter(curvatures, 11, 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot 1: Curvature Heatmap on Contour
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    sc = axes[0].scatter([p[0] for p in points], [p[1] for p in points], 
                         c=curvatures, cmap='jet', s=10, zorder=2)
    axes[0].set_title("Boundary Curvature Heatmap")
    axes[0].axis('off')
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot 2: Selected LRPs
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1].plot([p[0] for p in points], [p[1] for p in points], color='white', linewidth=1, zorder=1)
    
    top_indices = np.argsort(curvatures)[-2:] # Top 2 points
    for idx in top_indices:
        px, py = points[idx]
        rect = patches.Rectangle((px - 20, py - 20), 40, 40, linewidth=2, edgecolor='red', facecolor='none', zorder=3)
        axes[1].add_patch(rect)
    
    axes[1].set_title("Selected Local Refinement Patches")
    axes[1].axis('off')
    
    plt.savefig(os.path.join(save_dir, 'lrp_selection.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved lrp_selection.pdf")

def generate_morphwrapper_quiver(model, img_tensor, save_dir):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
        
    # Register hook to the MorphWrapper's estimator (mfe)
    try:
        hook_handle = model.decoder.morph1.mfe.register_forward_hook(get_activation('morph1'))
    except Exception as e:
        print("Could not attach hook to morph1. Model architecture might differ.")
        return
        
    with torch.no_grad():
        _ = model(img_tensor, mode='test')
        
    hook_handle.remove()
    
    if 'morph1' not in activation:
        return
        
    # activation['morph1'] shape: [B, 3K, H, W]
    O = activation['morph1'][0].cpu().numpy()
    
    # Extract first head's offsets (channels 0 and 1)
    u_x = O[0, :, :]
    u_y = O[1, :, :]
    
    step = max(1, u_x.shape[0] // 30) # downsample for quiver plot visibility
    
    Y, X = np.mgrid[0:u_x.shape[0]:step, 0:u_x.shape[1]:step]
    U = u_x[0:u_x.shape[0]:step, 0:u_x.shape[1]:step]
    V = u_y[0:u_y.shape[0]:step, 0:u_y.shape[1]:step]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(u_x**2 + u_y**2, cmap='viridis', alpha=0.5) # Background heatmap of offset magnitude
    ax.quiver(X, Y, U, V, color='red', scale=50, headwidth=3)
    ax.set_title("MorphWrapper Spatial Offset Vector Field")
    ax.axis('off')
    
    plt.savefig(os.path.join(save_dir, 'morphwrapper_quiver.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved morphwrapper_quiver.pdf")

def generate_difficulty_entropy(model, img_tensor, save_dir):
    with torch.no_grad():
        preds, cls_logits, embeddings, conf_map = model(img_tensor, mode='train')
    
    difficulty = conf_map[0, 0].cpu().numpy()
    
    # Calculate predictive entropy from 4-class logits
    probs = F.softmax(cls_logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = axes[0].imshow(difficulty, cmap='magma')
    axes[0].set_title("Heuristic Difficulty Proxy Map")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(entropy, cmap='inferno')
    axes[1].set_title("Predictive Entropy Map")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.savefig(os.path.join(save_dir, 'difficulty_and_entropy.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved difficulty_and_entropy.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='Path to trained .pth model')
    parser.add_argument('--image_path', type=str, default='', help='Path to a sample test image')
    parser.add_argument('--mask_path', type=str, default='', help='Path to the sample ground truth mask')
    parser.add_argument('--save_dir', type=str, default='./tmi_visualizations', help='Directory to save PDFs')
    opt = parser.parse_args()
    
    os.makedirs(opt.save_dir, exist_ok=True)
    setup_matplotlib_style()
    
    img, mask = load_data(opt.image_path, opt.mask_path)
    
    # Initialize Model
    model = EMCADNet(use_morph=True, morph_sample_k=4, pretrain=False)
    if opt.model_path and os.path.exists(opt.model_path):
        model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=False)
        print(f"Loaded model from {opt.model_path}")
    else:
        print("Model path not provided or not found. Using initialized weights for visualization.")
    
    model.eval()
    
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    
    # Generate Output Mask
    with torch.no_grad():
        preds = model(img_tensor, mode='test')
        pred_mask = preds[-1].sigmoid().squeeze().cpu().numpy()
        pred_mask_binary = (pred_mask >= 0.5).astype(np.uint8) * 255
    
    # Generate Visualizations
    generate_qualitative_comparison(img, mask, pred_mask_binary, opt.save_dir)
    generate_lrp_visualization(img, mask, opt.save_dir)
    generate_morphwrapper_quiver(model, img_tensor, opt.save_dir)
    generate_difficulty_entropy(model, img_tensor, opt.save_dir)
    
    print(f"\nAll visualizations generated successfully in '{opt.save_dir}'")

if __name__ == '__main__':
    main()
