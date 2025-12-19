import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

def split_dataset_step(base_path, step_name):
    step_dir = os.path.join(base_path, step_name)
    src_images_dir = os.path.join(step_dir, "images")
    src_labels_dir = os.path.join(step_dir, "labels")

    if not os.path.exists(src_images_dir):
        print(f"[{step_name}] 'images' 폴더를 찾을 수 없습니다: {src_images_dir}")
        return

    if os.path.exists(os.path.join(step_dir, "train")):
        print(f"[{step_name}] 이미 'train' 폴더가 존재하여 작업을 건너뜁니다.")
        return

    print(f"--- {step_name} 데이터 분할 시작 (8:1:1, Stratified) ---")

    files_by_class = defaultdict(list)
    valid_exts = {'.pt'}

    for root, dirs, files in os.walk(src_images_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, src_images_dir)
                
                class_key = os.path.dirname(rel_path)
                files_by_class[class_key].append(rel_path)

    if not files_by_class:
        print(f"[{step_name}] 분할할 이미지 파일이 없습니다.")
        return

    random.seed(42)

    splits = {"train": [], "val": [], "test": []}
    total_files = 0

    for class_key, files in files_by_class.items():
        random.shuffle(files)
        
        n_total = len(files)
        total_files += n_total
        
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        # 나머지는 test
        
        splits["train"].extend(files[:n_train])
        splits["val"].extend(files[n_train : n_train + n_val])
        splits["test"].extend(files[n_train + n_val :])

    print(f"[{step_name}] 총 {total_files}개 (Stratified) -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    for split, files in splits.items():
        for rel_path in tqdm(files, desc=f"[{step_name}] Moving {split}"):
            src_img = os.path.join(src_images_dir, rel_path)
            dst_img = os.path.join(step_dir, split, "images", rel_path)
            
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.move(src_img, dst_img)

            file_dir = os.path.dirname(rel_path)
            file_stem = os.path.splitext(os.path.basename(rel_path))[0]
            
            curr_label_dir = os.path.join(src_labels_dir, file_dir)
            
            if os.path.exists(curr_label_dir):
                for f in os.listdir(curr_label_dir):
                    if f.lower().endswith('.json') and os.path.splitext(f)[0] == file_stem:
                        src_label = os.path.join(curr_label_dir, f)
                        dst_label = os.path.join(step_dir, split, "labels", file_dir, f)
                        
                        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
                        shutil.move(src_label, dst_label)
                        break
    print(f"[{step_name}] 완료.")

if __name__ == "__main__":
    BASE_PATH = r"F:\dataset_final"
    
    for i in range(4):
        split_dataset_step(BASE_PATH, f"step{i}")
