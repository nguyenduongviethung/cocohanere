import torch
import torch.nn as nn
from utils import INPUT_MODE


class CodeBertWrapper(nn.Module):
    def __init__(self, encoder):
        super(CodeBertWrapper, self).__init__()
        self.encoder = encoder

    def forward(self, inputs=None, mode=INPUT_MODE.NL):
        if mode == INPUT_MODE.NL:
            outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        else:
            code_ids = inputs[0]
            outputs = self.encoder(code_ids, attention_mask=code_ids.ne(1))[1]
        return outputs


class GraphCodeBertWrapper(nn.Module):
    def __init__(self, encoder):
        super(GraphCodeBertWrapper, self).__init__()
        self.encoder = encoder

    def forward(self, inputs=None, mode=INPUT_MODE.NL):
        if mode == INPUT_MODE.NL:
            outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        else:
            code_ids, attn_mask, position_idx = inputs
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_ids)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]

            nodes_to_token_mask = nodes_to_token_mask.half()  # xxx
            inputs_embeddings = inputs_embeddings.half()

            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            outputs = self.encoder(inputs_embeds=inputs_embeddings,
                                   attention_mask=attn_mask,
                                   position_ids=position_idx)[1]
        return outputs
