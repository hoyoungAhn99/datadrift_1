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
  05_idea4_negative_prompt/ Frozen positive/LoRA + parent-local unknown prompts
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

Idea 4 starts from the strongest Idea 3 positive Vision-LoRA checkpoint. It
freezes the CLIP base, Vision LoRA, and positive prompts, then trains only a
non-root parent-local unknown prompt with leave-child-out supervision. Its
primary inference compares ID leaves and local unknown terminal paths with
exact global-path MAP; local greedy inference is saved as an ablation.

`idea3_fgvc_aircraft_b16_joint_vision_lora_global_depth.yaml` is the GPU 1
parent-context ablation. It inherits the main joint experiment and changes only
the prompt ablation, GPU, experiment name, and output paths.

Experiment outputs use this layout:

```text
outputs/
  shared/features/<dataset>/<clip-model>/
  experiments/<experiment-name>/
    checkpoints/
    results/
    diagnostics/
```

CLIP feature caches are shared because several experiments reuse the same
frozen features. All trainable checkpoints and evaluation artifacts are grouped
under the experiment that produced them.
