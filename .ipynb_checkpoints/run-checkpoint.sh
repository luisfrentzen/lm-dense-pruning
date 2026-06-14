models=(
    "/mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct"
    # "/mnt/Model-Weights/meta-llama/Llama-3.2-3B-Instruct"
    # "/mnt/Model-Weights/google/gemma-3-4b-it"
    # "/mnt/foxbrain-omni-distillation/academia_sinica/distill/out_seq/llama_Kalpaca"
    # "/mnt/foxbrain-omni-distillation/academia_sinica/distill/out_seq/llama_Kgsm8k"
    # "/mnt/foxbrain-omni-distillation/academia_sinica/distill/out_seq/llama_Kgsm8k+alpaca"
    
    # "/mnt/foxbrain-omni-distillation/academia_sinica/distill/out/gemma_Kalpaca"
    # "/mnt/foxbrain-omni-distillation/academia_sinica/distill/out/gemma_Kgsm8k"
    # "/mnt/foxbrain-omni-distillation/academia_sinica/distill/out/gemma_Kgsm8k+alpaca"
)

datasets=(
    # "bookcorpus+gsm8k"
    "bookcorpus"
    # "gsm8k"
)

for model in "${models[@]}"; do
    if [[ "$model" == *"llama"* ]]; then
        model_type="llama"
    else
        model_type="gemma3"
    fi
    
    for dataset in "${datasets[@]}"; do
        for ratio in 0.5 0.4 0.3 0.2 0.1; do
            uv run python main_prune.py base_model=$model \
                save_dir="/mnt/foxbrain-omni-distillation/academia_sinica/pruning/out_analysis" \
                model_type=$model_type \
                pruning_ratio=$ratio \
                dataset=$dataset
                
            # uv run python main_prune.py base_model=$model \
            #     model_type=$model_type \
            #     save_dir="/mnt/foxbrain-omni-distillation/academia_sinica/pruning/out_gold" \
            #     pruning_ratio=$ratio \
            #     dataset=$dataset \
            #     loss_type=gold \
            #     gold_beta=0.5 \
            #     gold_temperature=0.9
        done
    done
done