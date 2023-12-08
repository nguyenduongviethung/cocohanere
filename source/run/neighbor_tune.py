import os
import torch
from tqdm.auto import tqdm
from utils import DATA_MODE
from torch.optim import AdamW
from run.evaluate import evaluate
from model.membank import VecIndexBank
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler


def nn_train(model, pack, args, accelerator):
    dataset = pack.get_dataset(file_path=args.tune_path, mode=DATA_MODE.TRAIN)
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=args.tune_batch_size,
        num_workers=args.num_workers,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader) * args.tune_epoch)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler
    )

    nl_bank = VecIndexBank(size=args.nn_size, K=args.nn_k).to(accelerator.device)
    code_bank = VecIndexBank(size=args.nn_size, K=args.nn_k).to(accelerator.device)

    tr_num, tr_loss, best_mrr = 0, 0., 0.
    for epoch in tqdm(range(args.tune_epoch), total=args.tune_epoch, disable=not accelerator.is_local_main_process):
        model.zero_grad()
        model.train()
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                disable=not accelerator.is_local_main_process):
            indexs, nl_inputs, code_inputs = batch[0], batch[1], batch[2:]
            loss = model(
                nl_inputs=nl_inputs,
                code_inputs=code_inputs,
                nl_bank=nl_bank,
                code_bank=code_bank,
                accelerator=accelerator,
                dataset=dataset,
                indexs=indexs,
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(model=model, args=args, pack=pack, accelerator=accelerator)

        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            if args.encoder_save_path:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                output_dir = os.path.join(args.encoder_save_path, 'checkpoint')
                if accelerator.is_local_main_process:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)  # xxx
                    output_dir = os.path.join(output_dir, f'{pack.encoder_name}.bin')
                    accelerator.save(unwrapped_model.state_dict(), output_dir)

        accelerator.print('\n ' + '*' * 20)
        accelerator.print(' Best mrr: %s', round(best_mrr, 4), 'epoch:', epoch)
        accelerator.print(' ' + '*' * 20)
