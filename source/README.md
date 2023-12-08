# CoCoHaNeRe

## Install

### set environment

```shell
./script/init.sh
```

### prepare dataset

1. put tree sitter parser in ./data/parser. See [GraphCodeBert](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch)
2. put CodeSearchNet dataset in ./dataset
3. set const `basedir` in utils.py

## Run

```shell
./script/train.sh
```
