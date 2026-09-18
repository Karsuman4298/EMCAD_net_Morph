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
from utils.dataloader_hup import _load_image_any, _load_mask_any, _pair_images_masks

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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        mask_pil = _load_mask_any(mask_path)
        mask = np.array(mask_pil)
        mask = cv2.resize(mask, (352, 352))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        print("Paths not found or not provided, using dummy data.")
        img, mask = create_dummy_data()
    return img, mask

def get_lrp_data(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], []
    c = max(contours, key=cv2.contourArea)
    c = c.squeeze()
    
    step = 5
    curvatures = []
    points = []
    if len(c.shape) < 2: return [], []
    
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
    return points, curvatures

def generate_comprehensive_grid(data_list, save_dir, filename="comprehensive_grid.pdf"):
    # Columns: 1. Image, 2. Ground Truth, 3. HUPAnno Pred, 4. Difficulty Map, 5. MorphWrapper Offset
    num_rows = len(data_list)
    num_cols = 5
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    for r, data in enumerate(data_list):
        # 1. Input Image
        axes[r, 0].imshow(cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB))
        if r == 0: axes[r, 0].set_title("Input Image", fontweight='bold')
        axes[r, 0].axis('off')
        
        # 2. Ground Truth
        axes[r, 1].imshow(data['mask'], cmap='gray')
        if r == 0: axes[r, 1].set_title("Ground Truth", fontweight='bold')
        axes[r, 1].axis('off')
        
        # 3. HUPAnno Prediction
        axes[r, 2].imshow(data['pred'], cmap='gray')
        if r == 0: axes[r, 2].set_title("HUPAnno Prediction", fontweight='bold')
        axes[r, 2].axis('off')
        
        # 4. Difficulty Map
        im4 = axes[r, 3].imshow(data['difficulty'], cmap='magma')
        if r == 0: axes[r, 3].set_title("Difficulty Map", fontweight='bold')
        axes[r, 3].axis('off')
        
        # 5. MorphWrapper Quiver
        u_x, u_y = data['u_x'], data['u_y']
        if u_x is not None:
            step = max(1, u_x.shape[0] // 30)
            Y, X = np.mgrid[0:u_x.shape[0]:step, 0:u_x.shape[1]:step]
            U = u_x[0:u_x.shape[0]:step, 0:u_x.shape[1]:step]
            V = u_y[0:u_y.shape[0]:step, 0:u_y.shape[1]:step]
            
            axes[r, 4].imshow(u_x**2 + u_y**2, cmap='viridis', alpha=0.5)
            axes[r, 4].quiver(X, Y, U, V, color='red', scale=50, headwidth=3)
        if r == 0: axes[r, 4].set_title("MorphWrapper Field", fontweight='bold')
        axes[r, 4].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def generate_lrp_grid(data_list, save_dir, filename="lrp_grid.pdf"):
    # Columns: 1. Input Image, 2. Curvature Heatmap, 3. Selected LRPs
    num_rows = len(data_list)
    num_cols = 3
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    for r, data in enumerate(data_list):
        img_rgb = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
        
        # 1. Input Image
        axes[r, 0].imshow(img_rgb)
        if r == 0: axes[r, 0].set_title("Input Image", fontweight='bold')
        axes[r, 0].axis('off')
        
        # 2. Curvature Heatmap
        points, curvatures = data['points'], data['curvatures']
        axes[r, 1].imshow(img_rgb)
        if points:
            sc = axes[r, 1].scatter([p[0] for p in points], [p[1] for p in points], 
                                     c=curvatures, cmap='jet', s=10, zorder=2)
        if r == 0: axes[r, 1].set_title("Boundary Curvature", fontweight='bold')
        axes[r, 1].axis('off')
        
        # 3. Selected LRPs
        axes[r, 2].imshow(img_rgb)
        if points:
            axes[r, 2].plot([p[0] for p in points], [p[1] for p in points], color='white', linewidth=1, zorder=1)
            top_indices = np.argsort(curvatures)[-2:]
            for idx in top_indices:
                px, py = points[idx]
                rect = patches.Rectangle((px - 20, py - 20), 40, 40, linewidth=2, edgecolor='red', facecolor='none', zorder=3)
                axes[r, 2].add_patch(rect)
        if r == 0: axes[r, 2].set_title("Selected Local Patches", fontweight='bold')
        axes[r, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def process_examples(model, image_paths, mask_paths, save_dir):
    data_list = []
    
    for img_p, msk_p in zip(image_paths, mask_paths):
        img, mask = load_data(img_p, msk_p)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
        
        # Hook for morph
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
            
        hook_handle = None
        try:
            hook_handle = model.decoder.morph1.mfe.register_forward_hook(get_activation('morph1'))
        except Exception:
            pass
            
        with torch.no_grad():
            preds, cls_logits, embeddings, conf_map = model(img_tensor, mode='train')
            preds_test = model(img_tensor, mode='test')
            
        if hook_handle: hook_handle.remove()
        
        pred_mask = preds_test[-1].sigmoid().squeeze().cpu().numpy()
        pred_mask_binary = (pred_mask >= 0.5).astype(np.uint8) * 255
        
        difficulty = conf_map[0, 0].cpu().numpy()
        probs = F.softmax(cls_logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)[0].cpu().numpy()
        
        u_x, u_y = None, None
        if 'morph1' in activation:
            O = activation['morph1'][0].cpu().numpy()
            u_x = O[0, :, :]
            u_y = O[1, :, :]
            
        points, curvatures = get_lrp_data(mask)
        
        data_list.append({
            'img': img,
            'mask': mask,
            'pred': pred_mask_binary,
            'difficulty': difficulty,
            'entropy': entropy,
            'u_x': u_x,
            'u_y': u_y,
            'points': points,
            'curvatures': curvatures
        })
        
    generate_comprehensive_grid(data_list, save_dir, filename="grid_all_visualizations.pdf")
    generate_lrp_grid(data_list, save_dir, filename="grid_lrp_selection.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='Path to trained .pth model')
    parser.add_argument('--image_path', type=str, default='', help='Path to test image OR directory of images')
    parser.add_argument('--mask_path', type=str, default='', help='Path to GT mask OR directory of masks')
    parser.add_argument('--save_dir', type=str, default='./tmi_visualizations', help='Directory to save PDFs')
    parser.add_argument('--num_examples', type=int, default=3, help='Number of rows (examples) in the grid')
    opt = parser.parse_args()
    
    os.makedirs(opt.save_dir, exist_ok=True)
    setup_matplotlib_style()
    
    model = EMCADNet(use_morph=True, morph_sample_k=4, pretrain=False)
    if opt.model_path and os.path.exists(opt.model_path):
        model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=False)
        print(f"Loaded model from {opt.model_path}")
    
    model.eval()

    img_paths, msk_paths = [], []
    if opt.image_path and os.path.isdir(opt.image_path) and opt.mask_path and os.path.isdir(opt.mask_path):
        try:
            pairs = _pair_images_masks(opt.image_path, opt.mask_path)[:opt.num_examples]
            img_paths = [p[0] for p in pairs]
            msk_paths = [p[1] for p in pairs]
        except Exception as e:
            print(f"Error pairing images: {e}")
    else:
        img_paths, msk_paths = [opt.image_path], [opt.mask_path]
        
    print(f"\nGenerating {len(img_paths)}x grid for paper...")
    process_examples(model, img_paths, msk_paths, opt.save_dir)
    print(f"\nAll grids generated successfully in '{opt.save_dir}'")

if __name__ == '__main__':
    main()
