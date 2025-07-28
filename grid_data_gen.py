# import numpy as np
# import random
# from PIL import Image
# import os
# from tqdm import tqdm

# seed = 42
# np.random.seed(seed)
# random.seed(seed)

# output_root = "./data/grid_dataset"
# os.makedirs(output_root, exist_ok=True)

# grid_sizes = list(range(1, 65))
# images_per_class = 1600
# img_size = 128

# def random_color():
#     return tuple(np.random.randint(0, 256, size=3))

# def generate_grid_image(grid_n):
#     cell_size = img_size // grid_n
#     img = Image.new("RGB", (img_size, img_size))
#     for i in range(grid_n):
#         for j in range(grid_n):
#             color = random_color()
#             block = Image.new("RGB", (cell_size, cell_size), color)
#             img.paste(block, (j * cell_size, i * cell_size))
#     return img

# for grid_n in grid_sizes:
#     class_dir = os.path.join(output_root, f"grid_{grid_n}x{grid_n}")
#     os.makedirs(class_dir, exist_ok=True)
#     for idx in tqdm(range(images_per_class), desc=f"{grid_n}x{grid_n}"):
#         img = generate_grid_image(grid_n)
#         img.save(os.path.join(class_dir, f"{idx:06d}.png"))

# print("Dataset generation completed.")

import os, re
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

IMG_SIZE = 128  # 원본 한 변 길이

def _fix_one(args):
    in_path, out_path, grid_n, img_size = args
    img = Image.open(in_path).convert("RGB")
    w, h = img.size
    cell = img_size // grid_n
    crop_w = cell * grid_n
    crop_h = cell * grid_n
    if crop_w == w and crop_h == h:
        img.save(out_path)
        return
    img_cropped = img.crop((0, 0, crop_w, crop_h))
    img_fixed = img_cropped.resize((img_size, img_size), resample=Image.NEAREST)
    img_fixed.save(out_path)

def fix_dataset(input_root, output_root, img_size=IMG_SIZE, workers=None):
    if workers is None:
        workers = max(1, cpu_count() - 1)
    os.makedirs(output_root, exist_ok=True)
    tasks = []
    dir_pat = re.compile(r"grid_(\d+)x\1$")
    for root, dirs, files in os.walk(input_root):
        m = dir_pat.search(os.path.basename(root))
        if not m:
            continue
        n = int(m.group(1))
        rel_dir = os.path.relpath(root, input_root)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        for fn in files:
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            in_path = os.path.join(root, fn)
            out_path = os.path.join(out_dir, fn)
            tasks.append((in_path, out_path, n, img_size))
    if not tasks:
        return 0
    with Pool(processes=workers) as p:
        for _ in tqdm(p.imap_unordered(_fix_one, tasks), total=len(tasks)):
            pass
    return len(tasks)

fixed_cnt = fix_dataset(
    input_root="./data/gridset",      # 원본
    output_root="./data/gridset_fixed"  # 결과
)
print("fixed images:", fixed_cnt)