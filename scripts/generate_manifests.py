import os
import datetime

def generate_manifests():
    base_dir = r"c:\Users\Udit\Desktop\deepfake1"
    data_dir = os.path.join(base_dir, "data", "kaggle_realfake", "real_vs_fake", "real-vs-fake")
    manifests_dir = os.path.join(base_dir, "manifests")
    
    os.makedirs(manifests_dir, exist_ok=True)
    
    splits = ["train", "valid", "test"]
    split_names = {"train": "train", "valid": "val", "test": "test"}
    
    counts = {}
    stylegan_extra = []
    
    for split in splits:
        split_name = split_names[split]
        counts[split_name] = {"real": 0, "fake": 0}
        manifest_path = os.path.join(manifests_dir, f"kaggle_{split_name}.tsv")
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            for label_name, label_val in [("real", 0), ("fake", 1)]:
                dir_path = os.path.join(data_dir, split, label_name)
                if not os.path.exists(dir_path):
                    continue
                    
                files = []
                for fname in os.listdir(dir_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        files.append(fname)
                        if fname.lower().startswith("stylegan") and split_name == "train" and label_name == "fake":
                            stylegan_extra.append(fname)
                            
                files.sort()
                
                for fname in files:
                    rel_path = f"{split}/{label_name}/{fname}"
                    f.write(f"{rel_path}\t{label_val}\n")
                    counts[split_name][label_name] += 1
                    
    readme_path = os.path.join(manifests_dir, "MANIFEST_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Kaggle Dataset Manifests\n\n")
        f.write(f"**Date Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## File Counts\n\n")
        for split in ["train", "val", "test"]:
            f.write(f"### {split.capitalize()}\n")
            f.write(f"- Real: {counts[split]['real']}\n")
            f.write(f"- Fake: {counts[split]['fake']}\n")
            f.write(f"- Total: {counts[split]['real'] + counts[split]['fake']}\n\n")
            
        f.write("## StyleGAN Extra Files\n\n")
        if stylegan_extra:
            f.write(f"Found {len(stylegan_extra)} stylegan_* files in train/fake:\n")
            for fname in stylegan_extra:
                f.write(f"- {fname}\n")
        else:
            f.write("No extra stylegan_* files found.\n")
            
        f.write("\n## How to Regenerate\n")
        f.write("Run the following command from the project root:\n")
        f.write("```bash\npython scripts/generate_manifests.py\n```\n")

if __name__ == "__main__":
    generate_manifests()
