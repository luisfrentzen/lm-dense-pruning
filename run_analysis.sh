# python -m importance_analysis.run \
#       --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
#       --pruned ./out_analysis/Llama-3.1-8B-Instruct_P0.1bookcorpus \
#       --model_type llama --metric fisher --mobility \
#       --mapping  ./out_analysis/Llama-3.1-8B-Instruct_P0.1bookcorpus/structure_map.json \
#       --out analysis_out/Llama-3.1-8B-Instruct_P0.1bookcorpus_fisher

python -m importance_analysis.run \
      --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
      --pruned ./out_analysis/Llama-3.1-8B-Instruct_P0.2bookcorpus \
      --model_type llama --metric fisher --mobility \
      --mapping  ./out_analysis/Llama-3.1-8B-Instruct_P0.2bookcorpus/structure_map.json \
      --out analysis_out/Llama-3.1-8B-Instruct_P0.2bookcorpus_fisher

python -m importance_analysis.run \
      --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
      --pruned ./out_analysis/Llama-3.1-8B-Instruct_P0.3bookcorpus \
      --model_type llama --metric fisher --mobility \
      --mapping  ./out_analysis/Llama-3.1-8B-Instruct_P0.3bookcorpus/structure_map.json \
      --out analysis_out/Llama-3.1-8B-Instruct_P0.3bookcorpus_fisher
      
python -m importance_analysis.run \
      --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
      --pruned ./out_analysis/Llama-3.1-8B-Instruct_P0.4bookcorpus \
      --model_type llama --metric fisher --mobility \
      --mapping  ./out_analysis/Llama-3.1-8B-Instruct_P0.4bookcorpus/structure_map.json \
      --out analysis_out/Llama-3.1-8B-Instruct_P0.4bookcorpus_fisher

# python -m importance_analysis.run \
#       --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
#       --pruned ./out_analysis/Llama-3.1-8B-Instruct_P0.5bookcorpus \
#       --model_type llama --metric fisher --mobility \
#       --mapping  ./out_analysis/Llama-3.1-8B-Instruct_P0.5bookcorpus/structure_map.json \
#       --out analysis_out/Llama-3.1-8B-Instruct_P0.5bookcorpus_fisher