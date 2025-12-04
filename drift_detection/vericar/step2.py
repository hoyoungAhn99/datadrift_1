import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl

# Import your modules
from dataloader import VehicleDataModule 
from model import VehiInfoRet

def extract_features(model, dataloader, device):
    """
    Extracts features and labels from a dataloader using the model.
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, all_labels, _ = batch
            images = images.to(device)
            
            features = model(images)
            
            features_list.append(features.cpu())
            labels_list.append(all_labels.cpu())

    if len(features_list) == 0:
        return torch.tensor([]), torch.tensor([])
        
    return torch.cat(features_list), torch.cat(labels_list)

def calculate_ood_metrics(id_features, ood_features, exemplar_features, k=1):
    """
    Calculates AUROC and FPR based on the paper's criteria.
    ID = Step0 + Step1
    OOD = Step2
    """
    # Move exemplars to CPU
    gallery = exemplar_features.cpu()
    
    # Combined Queries
    X = torch.cat([id_features, ood_features], dim=0)
    
    # 1 = ID (Positive), 0 = OOD (Negative)
    y_true = np.concatenate([np.ones(len(id_features)), np.zeros(len(ood_features))])
    
    print(f"Calculating distances for {len(X)} queries against {len(gallery)} exemplars...")
    
    # Calculate Distances (Euclidean)
    dists = torch.cdist(X, gallery, p=2) 
    
    # Get k-Nearest Neighbor Distances
    topk_dists, _ = torch.topk(dists, k=k, dim=1, largest=False)
    
    # Mean distance to k nearest neighbors
    mean_dists = topk_dists.mean(dim=1).numpy()
    
    # --- Paper Implementation Logic ---
    
    # 1. Split distances into ID and OOD
    id_dists = mean_dists[:len(id_features)]
    ood_dists = mean_dists[len(id_features):]
    
    # 2. Determine Threshold for 95% Recall (TPR) on ID Samples
    # ID samples should have SMALL distance.
    threshold_95 = np.percentile(id_dists, 95)
    
    # 3. Calculate FPR on OOD samples at this threshold
    fp_count = np.sum(ood_dists < threshold_95)
    fpr_at_95 = fp_count / len(ood_dists)
    
    # 4. Calculate AUROC
    scores = -mean_dists 
    auroc = roc_auc_score(y_true, scores)
    
    print(f"\n[Analysis Results]")
    print(f"Threshold (at 95% ID Recall): {threshold_95:.4f}")
    print(f"Mean Dist ID : {id_dists.mean():.4f}")
    print(f"Mean Dist OOD: {ood_dists.mean():.4f}")

    return auroc, fpr_at_95, threshold_95

def sample_exemplars(features, labels, exemplar_k):
    """
    Samples 'exemplar_k' samples per class closest to the class mean.
    """
    unique_labels = torch.unique(labels)
    new_exemplar_features = []
    new_exemplar_labels = []

    print(f"Sampling {exemplar_k} exemplars per class...")

    for label in unique_labels:
        indices = (labels == label).nonzero(as_tuple=True)[0]
        class_features = features[indices]
        
        # Calculate Prototype (Mean)
        class_mean = class_features.mean(dim=0, keepdim=True)
        
        # Calculate distance to mean
        dists = torch.cdist(class_features, class_mean).squeeze()
        
        # Select top K closest
        k = min(len(indices), exemplar_k)
        _, topk_indices = torch.topk(dists, k=k, largest=False)
        
        selected_features = class_features[topk_indices]
        
        new_exemplar_features.append(selected_features)
        new_exemplar_labels.append(torch.full((k,), label, dtype=torch.long))

    return torch.cat(new_exemplar_features), torch.cat(new_exemplar_labels)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Load Model (with FIX for Unexpected Keys) ---
    ckpt_path = config['step2']['ckpt_path']
    print(f"Loading checkpoint from: {ckpt_path}")
    
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Check config path.")

    # [FIX 1] strict=False로 로드
    model = VehiInfoRet.load_from_checkpoint(ckpt_path, strict=False)
    
    # [FIX 2] 수동으로 state_dict에서 Exemplar 데이터를 꺼내서 주입
    raw_checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = raw_checkpoint['state_dict']
    
    if 'exemplar_features' in state_dict:
        print("Restoring exemplar features/labels from checkpoint manually...")
        
        # [CRITICAL FIX] 이미 속성이 존재하면 삭제하여 충돌 방지
        if hasattr(model, 'exemplar_features'):
            del model.exemplar_features
        if hasattr(model, 'exemplar_labels'):
            del model.exemplar_labels
            
        # register_buffer를 사용해 모델에 다시 등록
        model.register_buffer('exemplar_features', state_dict['exemplar_features'])
        model.register_buffer('exemplar_labels', state_dict['exemplar_labels'])
    else:
        # 만약 strict=False로 로드했는데 이미 모델 안에 들어와 있을 수도 있음 (PyTorch 버전/설정에 따라 다름)
        # 확인 후 없으면 에러 발생
        if not hasattr(model, 'exemplar_features'):
             raise RuntimeError("Checkpoint does not contain exemplar features! Please re-run step1.py.")

    model.to(device)
    model.eval()
    
    print(f"Loaded {len(model.exemplar_features)} exemplars.")

    # --- 2. Prepare Data (ID: Step 0 + Step 1, OOD: Step 2) ---
    
    # A. Step 0 Data (ID Part 1) - Test Set
    print("\nSetting up Step 0 Data (ID Part 1)...")
    dm_step0 = VehicleDataModule(
        json_paths=Path(config['data']['json_path']),
        image_path=Path(config['data']['image_path']),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        seed=42, 
        cache_filename="metadata_cache.pkl"
    )
    dm_step0.setup(stage='test')
    loader_step0 = dm_step0.test_dataloader()
    
    # B. Step 1 Data (ID Part 2) - All Data
    print("Setting up Step 1 Data (ID Part 2)...")
    dm_step1 = VehicleDataModule(
        json_paths=Path(config['step1']['json_path']),
        image_path=Path(config['step1']['image_path']),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        seed=42,
        cache_filename="metadata_cache_step1.pkl" 
    )
    dm_step1.setup(stage='fit')
    loader_step1 = dm_step1.train_dataloader() 
    
    # C. Step 2 Data (OOD) - All Data
    print("Setting up Step 2 Data (OOD)...")
    dm_step2 = VehicleDataModule(
        json_paths=Path(config['step2']['json_path']),
        image_path=Path(config['step2']['image_path']),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        seed=42,
        cache_filename="metadata_cache_step2.pkl"
    )
    dm_step2.setup(stage='fit')
    loader_step2 = dm_step2.train_dataloader()

    # --- 3. Feature Extraction ---
    print("\n[Extraction] Step 0 features...")
    feats_step0, _ = extract_features(model, loader_step0, device)
    
    print("[Extraction] Step 1 features...")
    feats_step1, _ = extract_features(model, loader_step1, device)
    
    print("[Extraction] Step 2 features...")
    feats_step2, labels_step2 = extract_features(model, loader_step2, device)

    # --- 4. Combine ID Features ---
    # ID = Step 0 (Test) + Step 1 (All)
    id_features = torch.cat([feats_step0, feats_step1], dim=0)
    ood_features = feats_step2
    
    print(f"\nTotal ID Samples: {len(id_features)} (Step0: {len(feats_step0)} + Step1: {len(feats_step1)})")
    print(f"Total OOD Samples: {len(ood_features)} (Step2)")

    # --- 5. OOD Detection Task ---
    ood_k = config['step2']['ood_k']
    current_exemplars = model.exemplar_features
    
    auroc, fpr95, threshold = calculate_ood_metrics(id_features, ood_features, current_exemplars, k=ood_k)
    
    print("="*40)
    print(f"Step 2 OOD Detection Results (k={ood_k})")
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f} (at TPR=95%, Threshold={threshold:.4f})")
    print("="*40)

    # --- 6. Update Exemplars with Step 2 (Incremental Learning) ---
    exemplar_k = config['training']['exemplar_k']
    
    # Sample new exemplars from Step 2
    step2_ex_features, step2_ex_labels = sample_exemplars(feats_step2, labels_step2, exemplar_k)
    
    print(f"Selected {len(step2_ex_features)} new exemplars from Step 2.")
    
    step2_ex_features = step2_ex_features.to(device)
    step2_ex_labels = step2_ex_labels.to(device)
    
    updated_features = torch.cat([model.exemplar_features, step2_ex_features], dim=0)
    updated_labels = torch.cat([model.exemplar_labels, step2_ex_labels], dim=0)
    
    # Assign (Update Model)
    model.exemplar_features = updated_features
    model.exemplar_labels = updated_labels
    
    # --- 7. Save New Checkpoint ---
    save_dir = Path("checkpoints_step2")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "step2_updated_model.ckpt"
    
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', logger=False)
    # 수동 연결
    model.trainer = trainer
    trainer.strategy.connect(model)
    trainer.save_checkpoint(save_path)
    
    print(f"Updated model saved to: {save_path}")

if __name__ == '__main__':
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it.")
        exit()
    
    main(config)