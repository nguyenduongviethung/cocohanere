## 1. Download CodeSearchNet dataset
```
mkdir dataset
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip
unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
```

## 2. Download filter dataset
```
wget https://raw.githubusercontent.com/microsoft/CodeBERT/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip
```

## 3. Preprocess
```
cd dataset
python preprocess.py
```
