import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig
import torch.nn.functional as F
import pytorch_lightning as pl
import tqdm

from loss2 import HiMS_min_loss
from loss_ms import MS_loss

class VehiInfoRet(pl.LightningModule):
    def __init__(self, pretrained=True, model_name="openai/clip-vit-base-patch32", 
                 exemplar_k=15, knn=1, alpha=2.0, beta=50.0, lam=0.5, loss="HiMS_loss",
                 min_lr=2e-6, max_lr=2e-4, weight_decay=1e-4):
        super(VehiInfoRet, self).__init__()

        self.exemplar_k = exemplar_k
        self.knn = knn
        self.exemplar_features = None
        self.exemplar_labels = None
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.loss = loss
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay

        if pretrained:
            self.feature_extractor = CLIPVisionModel.from_pretrained(model_name)
        else:
            model_config = CLIPVisionConfig(model_name)
            self.feature_extractor = CLIPVisionModel(model_config)

        self.feature_extractor.train()
        self.fc = nn.Linear(768, 128, bias=False)
    
    def forward(self, images):
        outputs = self.feature_extractor(images)
        features = outputs.pooler_output
        low_features = self.fc(features)
        return F.normalize(low_features, p=2, dim=1)
    
    def training_step(self, batch, batch_idx):
        images, all_labels, hi_labels = batch  # hi_labels: [B, 4]

        batch_size = images.size(0)
        num_hi = hi_labels.size(1)

        features = self(images)  # [B, D]
        if self.loss == "HiMS_min":
            loss = HiMS_min_loss(features, hi_labels, batch_size, num_hi, self.alpha, self.beta, self.lam)
        elif self.loss == "MS":
            loss = MS_loss(features, hi_labels, self.alpha, self.beta, self.lam)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self):
        print("\nBuilding exemplar set for retrieval validation...")
        self.feature_extractor.eval()
        train_loader = self.trainer.datamodule.train_dataloader()
        
        features_by_class = {}

        # 1. Extract features for the entire training set
        with torch.no_grad():
            for batch in tqdm.tqdm(train_loader, desc="Building exemplar set"):
                images, all_labels, _ = batch
                images = images.to(self.device)
                features = self(images)

                for feature, label in zip(features, all_labels):
                    label_item = label.item()
                    if label_item not in features_by_class:
                        features_by_class[label_item] = []
                    features_by_class[label_item].append(feature)

        exemplar_features = []
        exemplar_labels = []

        # 2. For each class, select exemplars
        for class_idx, class_features_list in features_by_class.items():
            if len(class_features_list) == 0:
                continue
            
            class_features = torch.stack(class_features_list)
            # Use all samples if k is large, or select top-k closest to mean
            if self.exemplar_k >= len(class_features):
                 exemplar_features.append(class_features)
                 exemplar_labels.extend([class_idx] * len(class_features))
            else:
                mean_feature = torch.mean(class_features, dim=0)
                distances = torch.cdist(mean_feature.unsqueeze(0), class_features).squeeze(0)
                _, top_k_indices = torch.topk(distances, self.exemplar_k, largest=False)
                exemplar_features.append(class_features[top_k_indices])
                exemplar_labels.extend([class_idx] * self.exemplar_k)

        if exemplar_features:
            self.exemplar_features = torch.cat(exemplar_features, dim=0)
            self.exemplar_labels = torch.tensor(exemplar_labels, device=self.device, dtype=torch.long)
            print(f"Exemplar set built with {len(self.exemplar_features)} samples.")
        else:
            print("Could not build exemplar set.")

    def calc_retrieval_metrics(self, query_features, query_labels):
        """
        Calculates Precision@1 and mAP@R efficiently using vector operations.
        """
        if self.exemplar_features is None or self.exemplar_labels is None:
            return 0.0, 0.0

        # 1. Compute Distance Matrix [Batch_Size, Num_Exemplars]
        dists = torch.cdist(query_features, self.exemplar_features)

        # 2. Sort distances (Ascending: closest first)
        # We process the entire ranking for mAP
        sorted_indices = torch.argsort(dists, dim=1, descending=False)
        
        # 3. Get the labels of the ranked exemplars [Batch_Size, Num_Exemplars]
        ranked_labels = self.exemplar_labels[sorted_indices]
        
        # 4. Create Boolean Mask of True Matches [Batch_Size, Num_Exemplars]
        # query_labels: [B] -> [B, 1] for broadcasting
        matches = (ranked_labels == query_labels.unsqueeze(1))

        # --- Metric 1: Precision@1 ---
        # Check if the very first item (rank 0) is a match
        prec1 = matches[:, 0].float().mean()

        # --- Metric 2: mAP@R ---
        # R is the number of total positives in the gallery for each query.
        # Note: Since our exemplar set might be a subset of train, R is count in exemplar_set.
        num_positives = matches.sum(dim=1) # [B]
        
        # Avoid division by zero if a query has no positives in gallery
        has_positives = num_positives > 0
        
        aps = []
        # Vectorized Average Precision Calculation
        # cumsum gives the number of positives found up to rank k
        cumulative_matches = matches.cumsum(dim=1).float()
        
        # Rank vector: [1, 2, 3, ...]
        ranks = torch.arange(1, matches.size(1) + 1, device=self.device).float()
        
        # Precision at k: (number of matches up to k) / k
        prec_at_k = cumulative_matches / ranks
        
        # We only sum precisions where a relevant item was found (matches == True)
        relevant_precisions = prec_at_k * matches.float()
        
        # Sum of relevant precisions / Total positives (R)
        # We calculate this per sample
        sum_precisions = relevant_precisions.sum(dim=1)
        
        # Calculate AP, handling cases with 0 positives
        ap = torch.zeros_like(sum_precisions)
        ap[has_positives] = sum_precisions[has_positives] / num_positives[has_positives].float()
        
        map_r = ap.mean()

        return prec1, map_r

    def validation_step(self, batch, batch_idx):
        images, all_labels, hi_labels = batch 
        batch_size = images.size(0)
        num_hi = hi_labels.size(1)

        features = self(images)

        # --- Loss Calculation ---
        if self.loss == "HiMS_min":
            loss = HiMS_min_loss(features, hi_labels, batch_size, num_hi, self.alpha, self.beta, self.lam)
        elif self.loss == "MS":
            loss = MS_loss(features, hi_labels, self.alpha, self.beta, self.lam)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # --- Metrics Calculation (P@1, mAP@R) ---
        prec1, map_r = self.calc_retrieval_metrics(features, all_labels)
        
        # Log metrics
        self.log("val_prec1", prec1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_map_r", map_r, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, all_labels, hi_labels = batch
        batch_size = images.size(0)
        num_hi = hi_labels.size(1)

        features = self(images)

        # --- Loss Calculation ---
        loss = HiMS_min_loss(features, hi_labels, batch_size, num_hi, self.alpha, self.beta, self.lam)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # --- Metrics Calculation ---
        prec1, map_r = self.calc_retrieval_metrics(features, all_labels)
        
        self.log("test_prec1", prec1, on_step=False, on_epoch=True)
        self.log("test_map_r", map_r, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.min_lr, # base_lr (min)에서 시작
            weight_decay=self.weight_decay
        )
        
        try:
            train_loader = self.trainer.datamodule.train_dataloader()
            steps_per_epoch = len(train_loader)
        except:
            print("Warning: Cannot auto-detect steps_per_epoch in configure_optimizers. Using default 95.")
            steps_per_epoch = 95

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=2e-6,
            max_lr=2e-4,
            step_size_up=5 * steps_per_epoch,
            mode='triangular',
            cycle_momentum=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }