import json

def fix_notebook():
    with open('infer.ipynb', 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = "".join(cell['source'])
            if 'def _pair_images_masks' in source:
                # We found the target cell.
                new_source = []
                for line in cell['source']:
                    if "img_files = sorted([f for f in os.listdir(image_root) if f.lower().endswith('.npy')])" in line:
                        line = line.replace("'.npy'", "('.npy', '.jpg', '.png', '.jpeg')")
                    elif "msk_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith('.npy')])" in line:
                        line = line.replace("'.npy'", "('.npy', '.jpg', '.png', '.jpeg')")
                    elif "msk_map = {_stem(p): p for p in msk_paths}" in line:
                        line = line.replace("_stem(p)", "_stem(p).replace('_segmentation', '')")
                    elif "arr = np.load(path, allow_pickle=True)" in line:
                        # We need to replace these blocks in both load functions
                        pass # We will do a full string replacement for the functions instead to make it clean
                
                # Actually let's just replace the whole functions
                # It's easier and less error prone to just parse the strings
                pass

if __name__ == "__main__":
    with open('infer.ipynb', 'r') as f:
        content = f.read()
    
    # fix pair images masks
    content = content.replace(
        "img_files = sorted([f for f in os.listdir(image_root) if f.lower().endswith('.npy')])",
        "img_files = sorted([f for f in os.listdir(image_root) if f.lower().endswith(('.npy', '.jpg', '.png', '.jpeg'))])"
    )
    content = content.replace(
        "msk_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith('.npy')])",
        "msk_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(('.npy', '.jpg', '.png', '.jpeg'))])"
    )
    content = content.replace(
        "msk_map = {_stem(p): p for p in msk_paths}",
        "msk_map = {_stem(p).replace('_segmentation', ''): p for p in msk_paths}"
    )
    
    # load_image_matched
    old_img_load = '''    arr = np.load(path, allow_pickle=True)\\n",'''
    new_img_load = '''    if path.lower().endswith('.npy'):\\n",
    "        arr = np.load(path, allow_pickle=True)\\n",
    "    else:\\n",
    "        arr = cv2.imread(path)\\n",
    "        if arr is not None:\\n",
    "            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)\\n",
    "        else:\\n",
    "            raise ValueError(f\\"Could not load image: {path}\\")\\n",'''
    content = content.replace(old_img_load, new_img_load, 1) # Only first occurrence (load_image_matched)
    
    # load_mask_matched
    old_mask_load = '''    arr = np.load(path, allow_pickle=True)\\n",'''
    new_mask_load = '''    if path.lower().endswith('.npy'):\\n",
    "        arr = np.load(path, allow_pickle=True)\\n",
    "    else:\\n",
    "        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)\\n",
    "        if arr is None:\\n",
    "            raise ValueError(f\\"Could not load mask: {path}\\")\\n",'''
    content = content.replace(old_mask_load, new_mask_load, 1) # Only second occurrence (load_mask_matched)
    
    with open('infer.ipynb', 'w') as f:
        f.write(content)
