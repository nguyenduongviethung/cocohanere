## 1. Download filter dataset
```
wget https://raw.githubusercontent.com/microsoft/CodeBERT/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip
cp -f data/preprocess.py dataset/preprocess.py
cp -f data/run.sh dataset/run.sh
```

## 3. Preprocess
```
cd dataset
./run.sh
```
