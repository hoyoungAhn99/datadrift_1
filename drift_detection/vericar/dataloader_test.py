import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
import json
import glob
import tqdm
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split


class VehicleDataset(Dataset):
    def __init__(self, metadata, image_path,
                 type_to_idx, make_to_idx, model_to_idx, year_to_idx, all_labels_to_idx):
        self.metadata = metadata
        self.image_path = image_path

        # string -> index
        self.type_to_idx = type_to_idx
        self.make_to_idx = make_to_idx
        self.model_to_idx = model_to_idx
        self.year_to_idx = year_to_idx
        self.all_labels_to_idx = all_labels_to_idx
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        pt_path = self.image_path / item['image_path']
        image = torch.load(pt_path)

        type_str  = item['type']
        make_str  = item['make']
        model_str = item['model']
        year_str  = item['year']
        all_str   = item['all_labels']

        type_label  = self.type_to_idx[type_str]
        make_label  = self.make_to_idx[make_str]
        model_label = self.model_to_idx[model_str]
        year_label  = self.year_to_idx[year_str]
        all_label   = self.all_labels_to_idx[all_str]

        hi_labels = torch.tensor(
            [type_label, make_label, model_label, year_label],
            dtype=torch.long
        )

        return image, all_label, hi_labels


class VehicleDataModule(pl.LightningDataModule):
    def __init__(self, json_paths, image_path, batch_size=64, num_workers=4):
        super().__init__()
        self.json_paths = json_paths
        self.image_path = image_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        cache_path = self.json_paths / "metadata_cache.pkl"

        if cache_path.exists():
            print(f"Loading metadata from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            self.metadata = cache_data['metadata']
            self.type_to_idx = cache_data['type_to_idx']
            self.make_to_idx = cache_data['make_to_idx']
            self.model_to_idx = cache_data['model_to_idx']
            self.year_to_idx = cache_data['year_to_idx']
            self.all_labels_to_idx = cache_data['all_labels_to_idx']
            self.idx_to_type = cache_data['idx_to_type']
            self.idx_to_make = cache_data['idx_to_make']
            self.idx_to_model = cache_data['idx_to_model']
            self.idx_to_year = cache_data['idx_to_year']
            self.idx_to_all_labels = cache_data['idx_to_all_labels']
        else:
            print("Cache not found. Scanning JSON files and creating new metadata...")
            json_files = glob.glob(str(self.json_paths / '**/*.json'), recursive=True)

            self.metadata = []
            type_labels = set()
            make_labels = set()
            model_labels = set()
            year_labels = set()
            all_labels = set()

            for json_path in tqdm.tqdm(json_files, desc="Processing JSONs"):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    attrs = data['car']['attributes']
                    car_type = attrs['type']
                    make = attrs['brand']
                    model = attrs['model']
                    year = attrs['year']

                    type_labels.add(car_type)
                    make_labels.add(make)
                    model_labels.add(model)
                    year_labels.add(year)
                    all_labels.add(f"{car_type}-{make}-{model}-{year}")

                    pt_path = Path(f"{car_type}/{make}/{model}/{year}/{Path(json_path).stem}").with_suffix('.pt')
                    self.metadata.append({
                        'image_path': str(pt_path),
                        'all_labels': f"{car_type}-{make}-{model}-{year}",
                        'type': car_type,
                        'make': make,
                        'model': model,
                        'year': year
                    })
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Warning: Skipping file {json_path} due to error: {e}")

            # string -> idx
            self.type_to_idx  = {label: idx for idx, label in enumerate(sorted(type_labels))}
            self.make_to_idx  = {label: idx for idx, label in enumerate(sorted(make_labels))}
            self.model_to_idx = {label: idx for idx, label in enumerate(sorted(model_labels))}
            self.year_to_idx  = {label: idx for idx, label in enumerate(sorted(year_labels))}
            self.all_labels_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

            # idx -> string
            self.idx_to_type  = {idx: label for label, idx in self.type_to_idx.items()}
            self.idx_to_make  = {idx: label for label, idx in self.make_to_idx.items()}
            self.idx_to_model = {idx: label for label, idx in self.model_to_idx.items()}
            self.idx_to_year  = {idx: label for label, idx in self.year_to_idx.items()}
            self.idx_to_all_labels = {idx: label for label, idx in self.all_labels_to_idx.items()}

            print(f"Saving metadata to cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'type_to_idx': self.type_to_idx,
                    'make_to_idx': self.make_to_idx,
                    'model_to_idx': self.model_to_idx,
                    'year_to_idx': self.year_to_idx,
                    'all_labels_to_idx': self.all_labels_to_idx,
                    'idx_to_type': self.idx_to_type,
                    'idx_to_make': self.idx_to_make,
                    'idx_to_model': self.idx_to_model,
                    'idx_to_year': self.idx_to_year,
                    'idx_to_all_labels': self.idx_to_all_labels
                }, f)
            
        print(f"Found {len(self.metadata)} images and {len(self.all_labels_to_idx)} unique all_labels classes.")

        # stratify by all_labels string
        all_labels_list = [m['all_labels'] for m in self.metadata]

        train_val_meta, self.test_meta = train_test_split(
            self.metadata,
            test_size=0.1,
            random_state=42,
            stratify=all_labels_list
        )
        train_val_labels = [m['all_labels'] for m in train_val_meta]

        self.train_meta, self.val_meta = train_test_split(
            train_val_meta,
            test_size=1/9,
            random_state=42,
            stratify=train_val_labels
        )

        if stage == 'fit' or stage is None:
            self.train_dataset = VehicleDataset(
                self.train_meta, self.image_path,
                self.type_to_idx, self.make_to_idx,
                self.model_to_idx, self.year_to_idx,
                self.all_labels_to_idx
            )
            self.val_dataset = VehicleDataset(
                self.val_meta, self.image_path,
                self.type_to_idx, self.make_to_idx,
                self.model_to_idx, self.year_to_idx,
                self.all_labels_to_idx
            )

        if stage == 'test' or stage is None:
            self.test_dataset = VehicleDataset(
                self.test_meta, self.image_path,
                self.type_to_idx, self.make_to_idx,
                self.model_to_idx, self.year_to_idx,
                self.all_labels_to_idx
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          persistent_workers=True)