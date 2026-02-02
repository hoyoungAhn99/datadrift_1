import torch
import yaml
from pathlib import Path
import tqdm
from pytorch_lightning import Trainer, seed_everything

from model import VehiInfoRet


def main():
    ckpt_path = Path("F:\IPIU2026\logs\mining\vericar_experiment_seed1234\version_1-HiMSwei\checkpoints\best-loss-epoch=53-val_loss=0.6140.ckpt")
    model = VehiInfoRet.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()

    



if __name__ == "__main__":
    main()