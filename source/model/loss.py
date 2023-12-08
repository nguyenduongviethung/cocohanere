import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LOSS_MODE


class LossModule(nn.Module):
    def __init__(self):
        super(LossModule, self).__init__()
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.dis_loss_fn = nn.CosineSimilarity()

    def forward(self, nl_vec, code_vec, nl_neighbor=None, code_neighbor=None, loss_mode=LOSS_MODE.CLIP_LOSS):
        if loss_mode == LOSS_MODE.CLIP_LOSS:
            return self.clip_loss(nl_vec=nl_vec, code_vec=code_vec)
        elif loss_mode == LOSS_MODE.BERT_LOSS:
            return self.bert_loss(nl_vec=nl_vec, code_vec=code_vec)
        elif loss_mode == LOSS_MODE.SYMI_LOSS:
            return self.symi_loss(nl_vec, code_vec)
        elif loss_mode == LOSS_MODE.NCCL_LOSS:
            return self.neighbor_loss(nl_vec, code_vec, nl_neighbor, code_neighbor)

    def neighbor_loss(self, nl_vec, code_vec, nl_neighbor, code_neighbor):
        return 0.5 * (self.single_loss(nl_vec, code_vec, nl_neighbor) +
                      self.single_loss(code_vec, nl_vec, code_neighbor))

    def single_loss(self, nl_vec, code_vec, nl_neighbor):
        # neighbor: batch * k * dim
        bs, dim = nl_vec.shape
        nl_new_vec = torch.cat([nl_vec.unsqueeze(1), nl_neighbor], dim=1)
        sim_matrix = torch.einsum("kij,kj->ki", nl_new_vec, code_vec)
        labels = torch.zeros(bs, device=nl_vec.device, dtype=torch.long)
        return self.class_loss_fn(sim_matrix, labels)

    def bert_loss(self, nl_vec, code_vec):
        bs = nl_vec.shape[0]
        scores = self.bmm(nl_vec, code_vec)
        labels = torch.arange(bs, device=nl_vec.device)
        return self.class_loss_fn(scores, labels)

    def clip_loss(self, nl_vec, code_vec):
        return (self.bert_loss(nl_vec, code_vec) + self.bert_loss(code_vec, nl_vec)) / 2.

    def symi_loss(self, nl_vec, code_vec):
        n2c_mat, c2n_mat, n2n_mat, c2c_mat = self.get_matrixs(nl_vec, code_vec)
        n2c_mat = F.normalize(n2c_mat, dim=1).flatten().unsqueeze(0)
        c2n_mat = F.normalize(c2n_mat, dim=1).flatten().unsqueeze(0)
        n2n_mat = F.normalize(n2n_mat, dim=1).flatten().unsqueeze(0)
        c2c_mat = F.normalize(c2c_mat, dim=1).flatten().unsqueeze(0)
        return (self.dis_loss_fn(n2c_mat, c2n_mat) +
                self.dis_loss_fn(n2c_mat, n2n_mat) +
                self.dis_loss_fn(n2c_mat, c2c_mat) +
                self.dis_loss_fn(c2n_mat, n2n_mat) +
                self.dis_loss_fn(c2n_mat, c2c_mat) +
                self.dis_loss_fn(n2n_mat, c2c_mat)) / 6.

    def bmm(self, p, q):
        return torch.einsum('md,nd->mn', p, q)

    def get_matrixs(self, nl_vec, code_vec):
        c2c_mat = self.bmm(code_vec, code_vec)
        n2n_mat = self.bmm(nl_vec, nl_vec)
        c2n_mat = self.bmm(code_vec, nl_vec)
        n2c_mat = self.bmm(nl_vec, code_vec)
        return n2c_mat, c2n_mat, n2n_mat, c2c_mat
