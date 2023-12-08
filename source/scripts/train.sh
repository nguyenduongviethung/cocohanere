declare -a model_names=("codebert"
                        "roberta"
                        "roberta-code"
                        "graphcodebert")
declare -a model_ids=("microsoft/codebert-base"
                       "roberta-base"
                       "microsoft/codebert-base-mlm"
                       "microsoft/graphcodebert-base")
declare -a tasks=("tune" "nncl")
declare -a langs=("ruby" "java" "python" "javascript" "php" "go")

function train() {
  for task in "${tasks[@]}"; do
    echo ${task}
    for lang in "${langs[@]}"; do
      echo ${lang}
      for((i=0;i<${#model_names[@]};i++)) do
        echo ${model_names[i]}
        accelerate launch main.py --task ${task} \
                    --lang ${lang} \
                    --encoder_name ${model_names[i]} \
                    --encoder_id ${model_ids[i]} \
                    --n_gpu 4 \
                    --num_workers 4 \
                    --eval_batch_size 128 \
                    --tune_batch_size 32 \
                    --tune_epoch 10 \
                    --nn_size 8096 \
                    --nn_k 10

        wait
      done
    done
  done
}

train;

