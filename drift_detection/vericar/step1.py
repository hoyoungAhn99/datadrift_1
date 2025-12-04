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

def calculate_ood_metrics(id_features, ood_features, exemplar_features, k=5):
    """
    Calculates AUROC and FPR based on the paper's criteria:
    "Threshold is set to achieve 95% recall of ID samples."
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
    
    # --- [New Logic] Paper Implementation ---
    
    # 1. Split distances into ID and OOD
    id_dists = mean_dists[:len(id_features)]
    ood_dists = mean_dists[len(id_features):]
    
    # 2. Determine Threshold for 95% Recall (TPR) on ID Samples
    # "Distance that achieves a 95% recall of ID samples"
    # ID samples should have SMALL distance.
    # We find the distance value that covers 95% of ID samples (95th percentile).
    threshold_95 = np.percentile(id_dists, 95)
    
    # 3. Calculate FPR on OOD samples at this threshold
    # False Positive: OOD Sample (Negative) that is predicted as ID (Positive)
    # Condition: OOD Distance < Threshold
    fp_count = np.sum(ood_dists < threshold_95)
    fpr_at_95 = fp_count / len(ood_dists)
    
    # 4. Calculate AUROC (Overall Performance)
    # Use -distance because lower distance = higher probability of ID
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
    
    # --- 1. Load Model ---
    ckpt_path = config['step1']['ckpt_path']
    print(f"Loading checkpoint from: {ckpt_path}")
    model = VehiInfoRet.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # --- Handle Missing Exemplars with Safe Registration ---
    if getattr(model, 'exemplar_features', None) is None:
        print("\n[Warning] Exemplars not found in checkpoint. Regenerating from Step 0 Train Data...")
        
        # Load Step 0 Training Data
        step0_dm = VehicleDataModule(
            json_paths=Path(config['data']['json_path']),
            image_path=Path(config['data']['image_path']),
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            seed=42, 
            cache_filename="metadata_cache.pkl"
        )
        step0_dm.setup(stage='fit')
        train_loader_step0 = step0_dm.train_dataloader()
        
        # Extract Features
        print("Extracting features from Step 0 Train Set...")
        train_feats, train_labels = extract_features(model, train_loader_step0, device)
        
        # Sample Exemplars
        ex_k = config['training']['exemplar_k']
        ex_feats, ex_labels = sample_exemplars(train_feats, train_labels, ex_k)
        
        # Clean up attribute if exists
        if hasattr(model, 'exemplar_features'):
            del model.exemplar_features
        if hasattr(model, 'exemplar_labels'):
            del model.exemplar_labels

        # Register buffers
        model.register_buffer('exemplar_features', ex_feats.to(device))
        model.register_buffer('exemplar_labels', ex_labels.to(device))
        
        print(f"Regenerated {len(ex_feats)} exemplars from Step 0.")
    else:
        print("Exemplars found in checkpoint.")

    # --- 2. Prepare Data for OOD Task ---
    
    print("\nSetting up Step 0 Data (Test Set)...")
    step0_dm_test = VehicleDataModule(
        json_paths=Path(config['data']['json_path']),
        image_path=Path(config['data']['image_path']),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        seed=42,
        cache_filename="metadata_cache.pkl"
    )
    step0_dm_test.setup(stage='test')
    test_loader_step0 = step0_dm_test.test_dataloader()
    
    print("Setting up Step 1 Data (OOD Set)...")
    step1_dm = VehicleDataModule(
        json_paths=Path(config['step1']['json_path']),
        image_path=Path(config['step1']['image_path']),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        seed=42,
        cache_filename="metadata_cache_step1.pkl"
    )
    step1_dm.setup(stage='fit') 
    loader_step1 = step1_dm.train_dataloader()

    # --- 3. Feature Extraction ---
    print("Extracting features for Step 0 Test (ID)...")
    id_features, id_labels = extract_features(model, test_loader_step0, device)
    
    print("Extracting features for Step 1 (OOD)...")
    ood_features, ood_labels = extract_features(model, loader_step1, device)

    # --- 4. OOD Detection Task ---
    ood_k = config['step1']['ood_k']
    current_exemplars = model.exemplar_features
    
    auroc, fpr95, threshold = calculate_ood_metrics(id_features, ood_features, current_exemplars, k=ood_k)
    
    print("="*40)
    print(f"OOD Detection Results (k={ood_k})")
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f} (at TPR=95%, Threshold={threshold:.4f})")
    print("="*40)

    # --- 5. Update Exemplars with Step 1 ---
    exemplar_k = config['training']['exemplar_k']
    
    # Sample new exemplars from Step 1
    step1_ex_features, step1_ex_labels = sample_exemplars(ood_features, ood_labels, exemplar_k)
    
    print(f"Selected {len(step1_ex_features)} new exemplars from Step 1.")
    
    step1_ex_features = step1_ex_features.to(device)
    step1_ex_labels = step1_ex_labels.to(device)
    
    updated_features = torch.cat([model.exemplar_features, step1_ex_features], dim=0)
    updated_labels = torch.cat([model.exemplar_labels, step1_ex_labels], dim=0)
    
    # Assign logic (avoid register_buffer reuse error)
    model.exemplar_features = updated_features
    model.exemplar_labels = updated_labels
    
    # --- 6. Save New Checkpoint ---
    save_dir = Path("checkpoints_step1")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "step1_updated_model.ckpt"
    
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', logger=False)
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