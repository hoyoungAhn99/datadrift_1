# Experiment Config Layout

The configuration tree follows the order in which the NegZeroHOC experiments
were developed.

```text
configs/
  00_baselines/             Frozen CLIP child-only and manual-unknown baselines
  01_feature_probes/        Linear/MLP probes on cached CLIP image features
  02_idea3_cached_prompt/   Idea 3 prompt learning on cached features
  03_sparse_path/           Sparse-path loss and Global Path MAP experiments
  04_joint_vision_lora/     Raw-image true Vision LoRA + prompt joint training
  references/               External/reference experiment configs
```

The main current FGVC-Aircraft experiment is:

```powershell
python scripts\train_idea3_joint_vision_lora.py --config configs\04_joint_vision_lora\idea3_fgvc_aircraft_b16_joint_vision_lora.yaml
python scripts\infer_idea3_joint_vision_lora.py --config configs\04_joint_vision_lora\idea3_fgvc_aircraft_b16_joint_vision_lora.yaml
```

`base_config` paths are resolved relative to the config file containing them.

The `lora` configs under `03_sparse_path/` are feature-space low-rank adapters.
Only `04_joint_vision_lora/` modifies the actual CLIP vision transformer.

The joint trainer validates every configured interval and runs final inference
automatically. The separate inference command regenerates the metrics from the
compact positive-prompt and Vision-LoRA checkpoint without retraining.
