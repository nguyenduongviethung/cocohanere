import random
import numpy as np
from data.parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from data.parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token
from tree_sitter import Language, Parser
from utils import DATA_MODE, basedir


def whole_word_mask_fn(tokenizer, tokens, mask_rate):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == tokenizer.pad_token or id == tokenizer.sep_token:
            break
        elif id not in tokenizer.all_special_tokens:
            if len(cand_indexes) > 0 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

    random.shuffle(cand_indexes)
    num_to_predict = int(round(len(cand_indexes) * mask_rate))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue

        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)

    assert len(covered_indexes) == len(masked_lms)
    masked_tokens = [tokenizer.mask_token if i in covered_indexes else token for token in tokens]
    return masked_tokens


def whole_word_mask(tokenizer, token_ids, mask_rate=0.1):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    masked_tokens = whole_word_mask_fn(tokenizer=tokenizer, tokens=tokens, mask_rate=mask_rate)
    return np.array(tokenizer.convert_tokens_to_ids(masked_tokens))


class FeatureExtractor(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def gen_code_feature(self, data, mode=DATA_MODE.TRAIN):
        # code str
        code = data.code

        # tokenize code
        code_tokens = self.tokenizer.tokenize(code)
        if self.args.identi:
            code_tokens = self.tokenizer.tokenize(
                'Function Identifier: \n' + data.func_name + '\nLanguage: \n' + data.language + '\nFunction Definition:\n') + [
                              self.tokenizer.sep_token] + code_tokens
            # xxx sep token ?

        code_tokens = code_tokens[:self.args.code_length - 2]

        code_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]

        # convert to ids
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

        # padding tokens
        code_ids += [self.tokenizer.pad_token_id] * (self.args.code_length - len(code_ids))
        return code_tokens, np.array(code_ids)

    def gen_nl_feature(self, data, mode=DATA_MODE.TRAIN):
        # query string
        nl = data.nl

        # tokenize
        nl_tokens = self.tokenizer.tokenize(nl)
        if self.args.identi:
            nl_tokens = self.tokenizer.tokenize('Language: \n' + data.language +
                                                '\nQuery: \n') + nl_tokens
        nl_tokens = nl_tokens[:self.args.nl_length - 2]

        nl_tokens = [self.tokenizer.cls_token] + nl_tokens + [
            self.tokenizer.sep_token]
        nl_ids = self.tokenizer.convert_tokens_to_ids(nl_tokens)
        nl_ids += [self.tokenizer.pad_token_id] * (self.args.nl_length - len(nl_ids))
        return nl_tokens, np.array(nl_ids)


dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

parsers = {}

for lang in dfg_function:
    LANGUAGE = Language(basedir + '/data/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class GraphCodeBertFeatureExtractor(FeatureExtractor):
    def __init__(self, tokenizer, args):
        FeatureExtractor.__init__(self, tokenizer=tokenizer, args=args)

    def gen_code_feature(self, data, mode=DATA_MODE.TRAIN):
        code = data.code

        # tokenize
        parser = parsers[data.language]
        code_tokens, dfg = extract_dataflow(code, parser, data.language)
        code_tokens = [self.tokenizer.tokenize('@ ' + x)[1:]
                       if idx != 0 else self.tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]

        if self.args.identi:
            code_tokens = self.tokenizer.tokenize(
                'Function Identifier: \n' + data.func_name + '\nLanguage: \n' + data.language + '\nFunction Definition: \n') + [
                              self.tokenizer.sep_token] + code_tokens

        code_tokens = code_tokens[
                      :self.args.code_length + self.args.data_flow_length - 2 - min(len(dfg, ),
                                                                                    self.args.data_flow_length)]
        code_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

        position_idx = [i + self.tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg = dfg[:self.args.code_length + self.args.data_flow_length - len(code_tokens)]
        code_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        code_ids += [self.tokenizer.unk_token_id for x in dfg]
        padding_length = self.args.code_length + self.args.data_flow_length - len(code_ids)
        position_idx += [self.tokenizer.pad_token_id] * padding_length
        code_ids += [self.tokenizer.pad_token_id] * padding_length
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([self.tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in position_idx])
        max_length = sum([i != 1 for i in position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return np.array(code_ids), attn_mask, np.array(position_idx)
