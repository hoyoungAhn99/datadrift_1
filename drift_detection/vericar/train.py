import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import yaml
 
# Import custom modules
from model import VehiInfoRet
from dataloader import VehicleDataModule

def main(args):
    datamodule = VehicleDataModule(
        json_paths=Path(config['data']['json_path']),
        image_path=Path(config['data']['image_path']),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        seed=config['seed'],

    )

    model = VehiInfoRet(pretrained=config['training']['pretrained'],
                        model_name=config['training']['model_name'],
                        exemplar_k=config['training']['exemplar_k'],
                        knn=config['training']['knn'],
                        loss=config['training']['loss'],
                        min_lr=config['training']['min_lr'],
                        max_lr=config['training']['max_lr'],
                        weight_decay=config['training']['weight_decay'])

    logger = TensorBoardLogger(config['logging']['log_dir'], name=config['logging']['exp_name'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_prec1',
        dirpath=f"{config['logging']['log_dir']}/{config['logging']['exp_name']}",
        filename='best-model-{epoch:02d}-{val_prec1:.2f}',
        save_top_k=3,
        mode='max',
    )

    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0] if torch.cuda.is_available() else 'auto',
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    print("Starting model training...")
    trainer.fit(model, datamodule)
    print("Training finished.")

    print("Starting model testing with the best checkpoint...")
    trainer.test(datamodule, ckpt_path='best')
    print("Testing finished.")

    save_path = config['training']['save_path']
    save_dir = Path(save_path).parent
    exemplar_save_path = save_dir / "exemplar_set.pt"

    print(f"Saving the best feature extractor weights to {save_path}")
    best_model = VehiInfoRet.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.feature_extractor.state_dict(), save_path)

    if best_model.exemplar_features is not None and best_model.exemplar_labels is not None:
        print(f"Saving the corresponding exemplar set to {exemplar_save_path}")
        torch.save({
            'exemplar_features': best_model.exemplar_features,
            'exemplar_labels': best_model.exemplar_labels
        }, exemplar_save_path)
    else:
        print("Warning: Exemplar set not found in the best model. Skipping save.")
    print("Script finished successfully.")

if __name__ == '__main__':
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it.")
        exit()
    main(config)
