from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

from source.model.wrapper import GraphCodeBertWrapper, CodeBertWrapper
from source.data.data_load import (CodeBertTrainDataset, CodeBertQueryDataset,
                           CodeBertCodeDataset, GraphCodeBertCodeDataset,
                           GraphCodeBertQueryDataset,
                           GraphCodeBertTrainDataset)
from source.utils import DATA_MODE


class Pack(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained(args.encoder_id)
        self.encoder_name = args.encoder_name
        self.dataset_dict = {}

    def get_dataset(self, file_path, mode):
        if self.dataset_dict.get(file_path) is None:
            dataset_class = None
            if self.encoder_name == 'graphcodebert':
                if mode == DATA_MODE.TRAIN:
                    dataset_class = GraphCodeBertTrainDataset
                elif mode == DATA_MODE.CODE:
                    dataset_class = GraphCodeBertCodeDataset
                elif mode == DATA_MODE.QUERY:
                    dataset_class = GraphCodeBertQueryDataset
            else:
                if mode == DATA_MODE.TRAIN:
                    dataset_class = CodeBertTrainDataset
                elif mode == DATA_MODE.QUERY:
                    dataset_class = CodeBertQueryDataset
                elif mode == DATA_MODE.CODE:
                    dataset_class = CodeBertCodeDataset
            self.dataset_dict[file_path] = dataset_class(json_file_path=file_path,
                                                         tokenizer=self.tokenizer,
                                                         args=self.args)

        return self.dataset_dict[file_path]


def get_encoder(encoder_name, encoder_id):
    config = RobertaConfig.from_pretrained(encoder_id)
    # config.attention_probs_dropout_prob = 0.1
    # config.hidden_dropout_prob = 0.1

    if encoder_name == 'graphcodebert':
        nl_encoder = GraphCodeBertWrapper(encoder=RobertaModel.from_pretrained(encoder_id, config=config))
        return nl_encoder, nl_encoder
    else:
        nl_encoder = CodeBertWrapper(encoder=RobertaModel.from_pretrained(encoder_id, config=config))
        return nl_encoder, nl_encoder
