# Augmenting Contrastive Learning-based Code Search via Consistent Hard Negative Examples Retrieval

This repository contains the data and source code used in paper Augmenting Contrastive Learning-based Code Search via Consistent Hard Negative Examples Retrieval. 
Code search is the process of finding the most semantically similar code snippets in a codebase based on a given natural language query. The current state-of-the-art code search methods use contrastive learning pre-training on pre-trained models, which learn semantic representation by distinguishing positive and negative samples. However, these methods can only learn low-level semantic differences due to their randomly selected negative samples from memory banks and gradient-free negative sample embeddings. This limitation means that they can only represent basic semantic representations and cannot represent higher-order semantic differences between similar codes and queries. To address this issue, we propose a new contrastive learning code search model called cocohanere. This model is based on differentiable hard negative examples and uses a newly constructed memory bank that can use the K-Nearest-Neighbor (KNN) algorithm to select hard negative examples in each learning and obtain differentiable negative sample embeddings to calculate contrastive learning loss. We conducted extensive experiments to evaluate the effectiveness of cocohanere on the large code search benchmark CodeSearchNet (CSN) with six programming languages. The experimental results show that cocohanere outperforms 16 baseline approaches and exceeds CodeBERT, GraphCodeBERT, and UniXcoder by 9.81\%, 6.73\% and 5.99\% on average MRR scores. Additionally, cocohanere is effective for different programming languages and code pre-train models and performs robustly under different hyper-parameters. Furthermore, qualitative analysis shows that cocohanere can learn higher-order semantic differences.

## Repository Structure
1. `Supplementary.pdf` include some experimental material
2. `data` dir includes the dataset url and preprocess scripts
3. `source` includes all implementation of model and experiments
4. `log` dir includes experiment logs

## Requirements
```
torch
accelerate
deepspeed
transformers
tensorboardX
tensorboard
tree_sitter
```

## prepare dataset

1. put tree sitter parser in ./data/parser. See [GraphCodeBert](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch)
2. put CodeSearchNet dataset in ./dataset
3. set const `basedir` in utils.py

## Run

```shell
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
```

## Evaluation

