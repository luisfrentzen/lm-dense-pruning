<h3>Structural Pruning of Large Language Models Based LLM-Pruner<h3>


## Introduction

#### Why LLM-Pruner
- [x] **Task-agnostic compression**: The compressed LLM should retain its original ability as a multi-task solver. 
- [x] **Less pruning calibration set**: In this work, we use only 10 samples from public dataset (bookcorpus) for each iteration to prune the LLM.  
- [x] **Efficient compression**: Around  3 minutes for pruning.

#### Updates:
* April 27, 2026: We implemented gemma3 for pruning mechanism based on LLMPruner

## Step-by-step Instructions  
    
### 1. Pruning

Modify the pruning configuration in the YAML file `./config/prune_config.yaml`. All configuration values will be included in the output directory name when the pruned model is saved.
```
# base model path
base_model: "../original_model/gemma-3-12b-it"

# output folder for pruned checkpoints
save_dir: "pruned_models"

# pruning setting
pruning_ratio: 0.1
pruner_type: "taylor"   # random | l1 | l2 | taylor

# enable exactly one mode
channel_wise: false
block_wise: true
layer_wise: false

# used only when layer_wise=true
# keep layers [0, layer-1]
layer: 24

# used only when block_wise=true
# end is exclusive
block_attention_layer_start: 0
block_attention_layer_end: 48
block_mlp_layer_start: 0
block_mlp_layer_end: 48

# pruning control
iterative_steps: 1
grouping_strategy: "sum"   # sum | mean | max | prod | first
global_pruning: false

# used only when pruner_type=taylor
taylor: "param_first"      # vectorize | param_second | param_first | param_mix
num_examples: 10

# generation / save setting
test_before_train: false
test_after_train: false
seed: 42
save_model: true

# generation setting
temperature: 1.0
top_p: 0.95
max_seq_len: 128
```

After updating the configuration file, run the pruning script to generate the pruned model.
```
python main_prune.py
```


### 2. Generation

#### How to load pruned/pre-trained models:
This project supports two types of saved pruned models: **uniformly pruned models** and **customized pruned models**. A uniformly pruned model is pruned consistently across all layers, from the first layer to the last layer. Because the layer dimensions remain consistent, this type of model can be loaded directly using Hugging Face's `.from_pretrained()` method. In contrast, a customized pruned model may prune only selected layers, and each layer may have a different remaining width. For example, some layers may retain a larger width, while other layers may be pruned more aggressively. Because the layer shapes are no longer uniform across the model, Hugging Face's standard `.from_pretrained()` method cannot reconstruct the architecture correctly. Therefore, customized pruned models require a custom loading procedure.

For customized pruned models, we save the model weights using `torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))`. We also save a `prune_config.json` file, which contains the layer-wise pruning information required to rebuild the customized model architecture. During loading, the custom loader first reads `prune_config.json`, reconstructs the pruned model architecture layer by layer, and then loads the saved weights from `pytorch_model.bin`. This allows customized pruned models with non-uniform layer shapes to be saved and loaded correctly.

### 3. Evaluation
For evaluating the performance of the pruned model, we follow [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the model:

The evaluation configuration can be modified in the YAML file `./config/eval_config.yaml`.
```
# directory that contains model folders to evaluate
root_dir: "./pruned_models_uniform"

# evaluation tasks for lm-eval
tasks:
  - "mmlu"
  - "arc_easy"
  - "arc_challenge"

# target device for single-device loading
device: "cuda:0"

# model dtype
dtype: "bfloat16"

# batch size for lm-eval
batch_size: "auto"

# upper limit used when batch_size="auto"
max_batch_size: 64

# output files
output_json: "./eval_out/all_results.json"
output_csv: "./eval_out/all_results.csv"

# enable multi-GPU loading path
use_multi_gpu: false

# number of GPUs to use
#   null   -> use all visible GPUs
gpus: null

# per-GPU memory cap for multi-GPU loading
max_memory_per_gpu: null

# CPU RAM cap for offloading
max_cpu_memory: null

# folder for HF offload files
offload_folder: "./offload"

```

```
python main_eval.py 
```

Results of LLM-Pruner:
| Pruning Ratio | #Param | Memory     | MMLU  | GSM8K | PIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | Average |
|---------------|--------|------------|-------|-------|-------|-----------|------------|-------|-------|---------|
| Gemma3-12B    | 12.19B | 22700.7MiB | 71.46 | 87.34 | 80.14 | 62.66     | 74.74      | 83.50 | 61.18 | 74.43   |
| Gemma3-11B    | 11.41B | 21500.0MiB | 65.99 | 84.84 | 77.64 | 57.34     | 70.64      | 80.81 | 54.78 | 70.29   |
| Gemma3-10B    | 10.63B | 19953.1MiB | 59.03 | 69.75 | 73.72 | 51.06     | 68.11      | 74.96 | 48.38 | 63.57   |


## References
```
@inproceedings{ma2023llmpruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Xinyin Ma and Gongfan Fang and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```
