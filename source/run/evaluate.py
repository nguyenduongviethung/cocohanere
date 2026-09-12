import gc
import torch
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, SequentialSampler

from source.utils import DATA_MODE

def evaluate(model, args, pack, accelerator):
    # load dataset
    query_dataset = pack.get_dataset(file_path=args.test_query_path, mode=DATA_MODE.QUERY)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers)

    code_dataset = pack.get_dataset(file_path=args.codebase_path, mode=DATA_MODE.CODE)
    code_dataloader = DataLoader(
        code_dataset,
        sampler=SequentialSampler(code_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers)

    query_dataloader = accelerator.prepare(query_dataloader)
    code_dataloader = accelerator.prepare(code_dataloader)

    model.eval()
    gc.collect()

    # get nl vector
    nl_vecs = []
    for batch in tqdm(query_dataloader, desc='eval_nl_vec',
                      disable=not accelerator.is_local_main_process, leave=False, mininterval=60):
        nl_inputs = batch
        with torch.no_grad():
            nl_vec, _ = model(nl_inputs=nl_inputs, code_inputs=None)
            nl_vec = accelerator.gather(nl_vec)
            nl_vecs.append(nl_vec)
    nl_vecs = torch.cat(nl_vecs)

    # get code vector
    code_vecs = []
    for batch in tqdm(code_dataloader, desc='eval_code_vec',
                      disable=not accelerator.is_local_main_process, leave=False, mininterval=60):
        if args.encoder_name == 'graphcodebert':
            code_inputs = batch[0:]
        else:
            code_inputs = [batch]
        with torch.no_grad():
            _, code_vec = model(nl_inputs=None, code_inputs=code_inputs)
            code_vec = accelerator.gather(code_vec)
            code_vecs.append(code_vec)
    code_vecs = torch.cat(code_vecs)

    model.train()

    # compute scores
    scores = torch.matmul(nl_vecs, code_vecs.T)
    sort_ids = torch.argsort(scores, dim=-1).cpu().numpy()[:, ::-1]

    # compute mrr
    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    # amm = 0
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if idx < 0 or idx >= len(code_urls):
                continue
            if code_urls[idx] == url:
                find = True
                # log first fault
                # if accelerator.is_local_main_process and rank > 5 and amm < 5:
                #     print("Query:", url)
                #     print("False:", code_urls[sort_id[0]])
                #     amm += 1
        if find:
            ranks.append(rank)
        else:
            ranks.append(1000000000)

    ranks = np.array(ranks)
    result = {
        "eval_mrr": float(np.mean(1. / ranks)),
        'top_1': float(np.mean(ranks <= 1)),
        'top_5': float(np.mean(ranks <= 5)),
        'top_10': float(np.mean(ranks <= 10))
    }

    accelerator.print('\n',
                      '\t', 'eval time:', datetime.now().strftime("%d/%m/%y %H:%M:%S.%f"))
    accelerator.print('\t', 'eval res: ', result)
    accelerator.print('\t', args.__str__())

    return result
