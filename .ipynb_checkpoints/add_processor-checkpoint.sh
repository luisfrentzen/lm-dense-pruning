BASE="/mnt/foxbrain-omni-distillation/academia_sinica/pruning/out"
FILE="/path/to/your/file.txt"

for dir in "$BASE"/*google-gemma*/; do
  cp /mnt/Model-Weights/google/gemma-3-4b-it/preprocessor_config.json "$dir"
  cp /mnt/Model-Weights/google/gemma-3-4b-it/processor_config.json "$dir"
done