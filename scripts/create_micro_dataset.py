import os
import shutil
import random
from pathlib import Path
import yaml

def create_micro_dataset(src_dir="data/dataset_final", dst_dir="data/dataset_micro", n_train=100, n_val=20):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if dst_path.exists():
        shutil.rmtree(dst_path)

    for split in ["train", "val"]:
        (dst_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        src_images_dir = src_path / "images" / split
        if not src_images_dir.exists():
            continue

        images = list(src_images_dir.glob("*.tif")) + list(src_images_dir.glob("*.jpg")) + list(src_images_dir.glob("*.png"))
        random.shuffle(images)

        n_samples = n_train if split == "train" else n_val
        selected_images = images[:n_samples]

        for img_path in selected_images:
            shutil.copy(img_path, dst_path / "images" / split / img_path.name)
            
            label_path = src_path / "labels" / split / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, dst_path / "labels" / split / label_path.name)

    yaml_path = src_path / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        data["path"] = str(Path(dst_dir).absolute())
        
        with open(dst_path / "dataset.yaml", "w") as f:
            yaml.dump(data, f)
            
    print(f"Created micro-dataset at {dst_dir} with {n_train} train and {n_val} val images.")

if __name__ == "__main__":
    create_micro_dataset()
