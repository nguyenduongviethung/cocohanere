import torch.nn as nn
from model.loss import LossModule
from utils import INPUT_MODE, to_device


class Retriever(nn.Module):
    def __init__(self, nl_encoder, code_encoder, args):
        super(Retriever, self).__init__()
        self.nl_encoder = nl_encoder
        self.code_encoder = code_encoder
        self.args = args
        self.loss_module = LossModule()

    def get_vec(self, nl_inputs=None, code_inputs=None):
        nl_vec = self.nl_encoder(inputs=nl_inputs, mode=INPUT_MODE.NL) if nl_inputs is not None else None
        code_vec = self.nl_encoder(inputs=code_inputs, mode=INPUT_MODE.CODE) if code_inputs is not None else None
        return nl_vec, code_vec

    def forward(self,
                nl_inputs=None,
                code_inputs=None,
                nl_bank=None,
                code_bank=None,
                accelerator=None,
                dataset=None,
                indexs=None,
                ):
        nl_vec, code_vec = self.get_vec(nl_inputs=nl_inputs, code_inputs=code_inputs)

        if accelerator is None:
            return nl_vec, code_vec
        else:
            return self.nn_loss(
                nl_vec=nl_vec,
                code_vec=code_vec,
                nl_bank=nl_bank,
                code_bank=code_bank,
                accelerator=accelerator,
                dataset=dataset,
                indexs=indexs
            )

    def nn_loss(self,
                nl_vec,
                code_vec,
                nl_bank,
                code_bank,
                accelerator,
                dataset,
                indexs,
                ):
        # compute basic loss
        loss_norm = self.loss_module.clip_loss(nl_vec=nl_vec, code_vec=code_vec)

        # save batch vec and indexs
        all_indexs = accelerator.gather(indexs)
        nl_bank.put(accelerator.gather(nl_vec), all_indexs)
        code_bank.put(accelerator.gather(code_vec), all_indexs)

        # get batch shape, current device, code input amount
        bs, dim = nl_vec.shape
        device = nl_vec.device

        # select batch input to get nn loss xxx
        # if 0 < self.args.alpha < 1:
        #     select_len = int(bs * self.args.alpha)
        #     code_vec = code_vec[:select_len]
        #     nl_vec = nl_vec[:select_len]

        # get nl k neighbor index and inputs
        if self.args.knn == 'bimodal':
            nl_neighbor_indexs, nl_neighbor_indexs_real = nl_bank.get_index(code_vec)
            nl_neighbor_inputs = dataset.get_by_idx(indexs=nl_neighbor_indexs, device=device)[1].to(device)
            # get code k neighbor index and inputs
            code_neighbor_indexs, code_neighbor_indexs_real = code_bank.get_index(nl_vec)
            code_neighbor_inputs = to_device(dataset.get_by_idx(indexs=code_neighbor_indexs, device=device)[2:],
                                             device)
        elif self.args.knn == 'unimodal':
            nl_neighbor_indexs, nl_neighbor_indexs_real = nl_bank.get_index(nl_vec)
            nl_neighbor_inputs = dataset.get_by_idx(indexs=nl_neighbor_indexs, device=device)[1].to(device)
            # get code k neighbor index and inputs
            code_neighbor_indexs, code_neighbor_indexs_real = code_bank.get_index(code_vec)
            code_neighbor_inputs = to_device(dataset.get_by_idx(indexs=code_neighbor_indexs, device=device)[2:],
                                             device)

        # get k neighbor features
        nl_neighbor_vec, code_neighbor_vec = self.get_vec(nl_inputs=nl_neighbor_inputs,
                                                          code_inputs=code_neighbor_inputs)

        # update k neighbor features
        nl_bank.update(indexs=nl_neighbor_indexs_real, vecs=nl_neighbor_vec)
        code_bank.update(indexs=code_neighbor_indexs_real, vecs=code_neighbor_vec)

        # get neighbor loss
        loss_neigh = self.loss_module.neighbor_loss(nl_vec, code_vec,
                                                    nl_neighbor_vec.view(bs, self.args.nn_k, -1),
                                                    code_neighbor_vec.view(bs, self.args.nn_k, -1))

        if self.args.no_norm_loss:
            return loss_neigh
        else:
            return (loss_neigh + loss_norm) / 2.
