conda create -n code-search python=3.9
conda activate code-search
pip3 install --pre torch  --extra-index-url https://download.pytorch.org/whl/nightly/cu113
pip install accelerate
pip install deepspeed
pip install transformers
pip install tensorboardX
pip install tensorboard
pip install tree_sitter
accelerate config