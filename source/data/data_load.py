import os
import json
import time
import numpy as np
import torch
import pickle
import multiprocessing
from utils import DATA_MODE, to_device
from data.extract import FeatureExtractor
from data.extract import GraphCodeBertFeatureExtractor, whole_word_mask
from torch.utils.data import Subset, DataLoader, Dataset


class CSNData(object):
    def __init__(self, js):
        self.func_name = js['func_name']
        self.language = js['language']
        self.url = js['url']
        self.code = ' '.join(js['code_tokens'])
        self.nl = ' '.join(js['docstring_tokens'])
        self.partition = js['partition']


def js2data(line):
    return CSNData(js=json.loads(line.strip()))


class CSNGraphData(object):
    def __init__(self, js):
        self.func_name = js['func_name']
        self.language = js['language']
        self.url = js['url']
        self.code = js['original_string']
        self.nl = ' '.join(js['docstring_tokens'])
        self.partition = js['partition']


def js2graphdata(line):
    return CSNGraphData(js=json.loads(line.strip()))


def read_csn_dataset(encoder_name, file_path, convert_fn, extractor, cpu=8):
    if encoder_name == 'graphcodebert':
        js_fn = js2graphdata
    else:
        js_fn = js2data

    with open(file_path) as f:
        data = []
        for l in f.readlines():
            d = js_fn(l)
            data.append((d, extractor,))
        pool = multiprocessing.Pool(cpu)
        res = pool.map(convert_fn, data)
        return res


class BasicDataset(Dataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        super(BasicDataset, self).__init__()
        self.json_file_path = json_file_path
        self.tokenizer = tokenizer
        self.args = args
        self.cache_file = json_file_path + args.encoder_name + '.pkl'
        # if os.path.exists(self.cache_file):
        #     self.examples = pickle.load(open(self.cache_file, 'rb'))
        # else:
        #     self.read_dataset()
        self.read_dataset()

        self.subloader = None
        self.subset = None

    def read_dataset(self):
        self.extractor = self.extractor_class(tokenizer=self.tokenizer, args=self.args)
        self.examples = read_csn_dataset(
            encoder_name=self.args.encoder_name,
            file_path=self.json_file_path,
            convert_fn=self.convert_func,
            extractor=self.extractor)
        # pickle.dump(self.examples, open(self.cache_file, 'wb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        res = self.examples[idx]
        return res


class CodeBertTrainDataset(BasicDataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        self.convert_func = js2cbt
        self.extractor_class = FeatureExtractor
        self.partition = 'train'
        super(CodeBertTrainDataset, self).__init__(json_file_path, tokenizer, args)

    def __getitem__(self, idx):
        example = self.examples[idx]
        if self.args.mask:
            return (
                torch.tensor([idx]),
                torch.from_numpy(
                    whole_word_mask(
                        tokenizer=self.tokenizer,
                        token_ids=example.nl_ids,
                        # mask_rate=mask_rate
                    )
                ),
                torch.from_numpy(
                    whole_word_mask(
                        tokenizer=self.tokenizer,
                        token_ids=example.code_ids
                    )
                )
            )
        else:
            return (torch.tensor([idx]),
                    torch.from_numpy(
                        example.nl_ids),
                    torch.from_numpy(
                        example.code_ids))

    def get_by_idx(self, indexs, device):

        if self.subloader is None:
            self.subset = Subset(self, indexs)
            self.subloader = DataLoader(self.subset,
                                        batch_size=len(indexs))
            res = next(iter(self.subloader))
        else:
            self.subset.indices = indexs
            res = next(iter(self.subloader))
        return res


class CodeBertTrainData(object):
    def __init__(self,
                 nl_ids,
                 code_ids
                 ):
        self.nl_ids = nl_ids
        self.code_ids = code_ids


def js2cbt(item):
    data, extractor = item
    _, nl_ids = extractor.gen_nl_feature(data, mode=DATA_MODE.TRAIN)
    _, code_ids = extractor.gen_code_feature(data, mode=DATA_MODE.TRAIN)

    return CodeBertTrainData(
        nl_ids=nl_ids,
        code_ids=code_ids)


class CodeBertQueryDataset(BasicDataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        self.convert_func = js2cbq
        self.extractor_class = FeatureExtractor
        self.partition = 'test'

        super(CodeBertQueryDataset, self).__init__(json_file_path,
                                                   tokenizer,
                                                   args)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return torch.from_numpy(example.nl_ids)


class CodeBertQueryData(object):
    def __init__(self,
                 url,
                 nl_ids
                 ):
        self.url = url
        self.nl_ids = nl_ids


def js2cbq(item):
    data, extractor = item
    _, nl_ids = extractor.gen_nl_feature(data, mode=DATA_MODE.QUERY)
    return CodeBertQueryData(
        url=data.url,
        nl_ids=nl_ids)


class CodeBertCodeDataset(BasicDataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        self.convert_func = js2cbc
        self.extractor_class = FeatureExtractor
        self.partition = 'test'

        super(CodeBertCodeDataset, self).__init__(json_file_path,
                                                  tokenizer,
                                                  args)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return torch.from_numpy(example.code_ids)


class CodeBertCodeData(object):
    def __init__(self,
                 url,
                 code_ids):
        self.url = url
        self.code_ids = code_ids


def js2cbc(item):
    data, extractor = item
    _, code_ids = extractor.gen_code_feature(data, mode=DATA_MODE.CODE)
    return CodeBertCodeData(
        url=data.url,
        code_ids=code_ids)


class GraphCodeBertQueryDataset(BasicDataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        self.convert_func = js2gcbq
        self.extractor_class = GraphCodeBertFeatureExtractor
        self.partition = 'test'

        super(GraphCodeBertQueryDataset, self).__init__(json_file_path,
                                                        tokenizer,
                                                        args)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return torch.from_numpy(example.nl_ids)


class GraphCodeBertQueryData(object):
    def __init__(self, url,
                 nl_ids,
                 ):
        self.url = url
        self.nl_ids = nl_ids


def js2gcbq(item):
    data, extractor = item
    _, nl_ids = extractor.gen_nl_feature(data, mode=DATA_MODE.QUERY)

    return GraphCodeBertQueryData(
        url=data.url,
        nl_ids=nl_ids
    )


class GraphCodeBertCodeDataset(BasicDataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        self.convert_func = js2gcbc
        self.extractor_class = GraphCodeBertFeatureExtractor
        self.partition = 'test'

        super(GraphCodeBertCodeDataset, self).__init__(json_file_path,
                                                       tokenizer,
                                                       args)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return (torch.from_numpy(example.code_ids),
                torch.from_numpy(example.attn_mask),
                torch.from_numpy(example.position_idx))


class GraphCodeBertCodeData(object):
    def __init__(self, url,
                 code_ids,
                 attn_mask,
                 position_idx,
                 ):
        self.url = url
        self.code_ids = code_ids
        self.attn_mask = attn_mask
        self.position_idx = position_idx


def js2gcbc(item):
    data, extractor = item
    code_ids, attn_mask, position_idx = extractor.gen_code_feature(data, mode=DATA_MODE.CODE)

    return GraphCodeBertCodeData(
        url=data.url,
        code_ids=code_ids,
        attn_mask=attn_mask,
        position_idx=position_idx,
    )


class GraphCodeBertTrainDataset(BasicDataset):
    def __init__(self,
                 json_file_path,
                 tokenizer,
                 args,
                 ):
        self.convert_func = js2gcbt
        self.extractor_class = GraphCodeBertFeatureExtractor
        self.partition = 'train'

        super(GraphCodeBertTrainDataset, self).__init__(json_file_path,
                                                        tokenizer,
                                                        args)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.args.mask:
            return (
                torch.tensor([idx]),
                torch.from_numpy(
                    whole_word_mask(
                        tokenizer=self.tokenizer,
                        token_ids=example.nl_ids
                    )
                ),
                torch.from_numpy(
                    whole_word_mask(
                        tokenizer=self.tokenizer,
                        token_ids=example.code_ids
                    )
                ),
                torch.from_numpy(example.attn_mask),
                torch.from_numpy(example.position_idx)
            )

        else:
            return (torch.tensor([idx]),
                    torch.from_numpy(example.nl_ids),
                    torch.from_numpy(example.code_ids),
                    torch.from_numpy(example.attn_mask),
                    torch.from_numpy(example.position_idx))

    def get_by_idx(self, indexs, device):
        def collate(batch):
            idxes = torch.stack([idx for idx, _, _, _, _ in batch])
            nl_ids = torch.stack([nl_id for _, nl_id, _, _, _ in batch])
            code_ids = torch.stack([code_id for _, _, code_id, _, _ in batch])
            posi_ids = torch.stack(to_device([posi_id for _, _, _, _, posi_id in batch], device=device))
            attn_masks = torch.stack(to_device([attn_mask for _, _, _, attn_mask, _ in batch], device=device))
            return [idxes, nl_ids, code_ids, attn_masks, posi_ids]

        if self.subloader is None:
            self.subset = Subset(self, indexs)
            self.subloader = DataLoader(self.subset,
                                        batch_size=len(indexs),
                                        collate_fn=collate
                                        )
            res = next(iter(self.subloader))
        else:
            self.subset.indices = indexs
            res = next(iter(self.subloader))
        return res


class GraphCodeBertTrainData(object):
    def __init__(self,
                 nl_ids,
                 code_ids,
                 attn_mask,
                 position_idx,
                 ):
        self.nl_ids = nl_ids
        self.code_ids = code_ids
        self.attn_mask = attn_mask
        self.position_idx = position_idx


def js2gcbt(item):
    data, extractor = item
    _, nl_ids = extractor.gen_nl_feature(data, mode=DATA_MODE.TRAIN)

    code_ids, attn_mask, position_idx = extractor.gen_code_feature(data, mode=DATA_MODE.TRAIN)

    return GraphCodeBertTrainData(
        nl_ids=nl_ids,
        code_ids=code_ids,
        attn_mask=attn_mask,
        position_idx=position_idx,
    )
