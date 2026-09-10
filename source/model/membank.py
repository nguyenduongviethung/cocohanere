import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, size, is_index):
        super(MemoryBank, self).__init__()
        self.size = size
        self.is_index = is_index
        if is_index:
            self.register_buffer('bank', torch.arange(self.size)[torch.randperm(self.size)].view(-1, size))
        else:
            self.register_buffer('bank', torch.randn(768, size))
        self.register_buffer('bank_ptr', torch.LongTensor([0]))

    @torch.no_grad()
    def get_bank(self):
        return self.bank.t()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        bs, dim = batch.shape
        ptr = int(self.bank_ptr)

        if ptr + bs >= self.size:
            self.bank[:, ptr:] = batch[:self.size - ptr].T
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr:ptr + bs] = batch.T
            self.bank_ptr[0] = ptr + bs

    @torch.no_grad()
    def update(self, indexs, vecs): # xxx
        self.bank[:, indexs] = vecs.T



class NeighborBank(nn.Module):
    def __init__(self, size=2 ** 12, K=10):
        super(NeighborBank, self).__init__()
        self.queue = MemoryBank(size=size, is_index=False)
        self.K = K

    @torch.no_grad()
    def put(self, batch):
        self.queue._dequeue_and_enqueue(batch)

    @torch.no_grad()
    def get_neighbors(self, batch):
        bank = self.queue.get_bank()
        sim_mat = torch.einsum('nd,md->nm', batch, bank)
        index = torch.topk(
            sim_mat,
            dim=1,
            k=self.K
        ).indices
        return index

    @torch.no_grad()
    def update(self, indexs, vecs):
        self.queue.update(indexs=indexs, vecs=vecs)


class VecIndexBank(nn.Module):
    def __init__(self, size=2 ** 12, K=10):
        super(VecIndexBank, self).__init__()
        self.size = size
        self.K = K
        self.vec_bank = NeighborBank(size=size, K=K)
        self.index_bank = MemoryBank(size=size, is_index=True)

    @torch.no_grad()
    def get_index(self, batch):
        index = self.vec_bank.get_neighbors(batch=batch).flatten()
        bank = self.index_bank.get_bank()
        input_neighbor_index = torch.index_select(bank, dim=0, index=index).flatten()
        return input_neighbor_index, index

    @torch.no_grad()
    def update(self, indexs, vecs):
        self.vec_bank.update(indexs=indexs, vecs=vecs)

    @torch.no_grad()
    def put(self, batch, indexs):
        self.vec_bank.put(batch)
        self.index_bank._dequeue_and_enqueue(indexs)
