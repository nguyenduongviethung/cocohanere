import os
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from utils import DATA_MODE
from tqdm.auto import tqdm
from torch.optim import AdamW
from model.loss import LossModule
from run.evaluate import evaluate


def train(model, pack, args, accelerator):
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

    model, dataloader, optimizer, scheduler = accelerator.prepare(
        model,
        dataloader,
        optimizer,
        scheduler
    )

    loss_module = LossModule()
    model.zero_grad()
    model.train()

    tr_num, tr_loss, best_mrr = 0, 0., 0.
    for epoch in tqdm(range(args.tune_epoch), total=args.tune_epoch, disable=not accelerator.is_local_main_process):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                disable=not accelerator.is_local_main_process):
            # compute loss
            nl_inputs, code_inputs = batch[1], batch[2:]
            nl_vec, code_vec = model(nl_inputs=nl_inputs, code_inputs=code_inputs)
            loss = loss_module.clip_loss(nl_vec=nl_vec, code_vec=code_vec)

            # backward
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                accelerator.print("epoch {} step {} loss {}".format(epoch, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

        res = evaluate(model=model, args=args, pack=pack, accelerator=accelerator)

        if res['eval_mrr'] > best_mrr:
            best_mrr = res['eval_mrr']
            if args.encoder_save_path:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                output_dir = os.path.join(args.encoder_save_path, 'checkpoint')
                if accelerator.is_local_main_process:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_dir = os.path.join(output_dir, f'{pack.encoder_name}_norm.bin')
                    accelerator.save(unwrapped_model.state_dict(), output_dir)


        accelerator.print('\n ' + '*' * 20)
        accelerator.print(' Best mrr: %s', round(best_mrr, 4), 'epoch:', epoch)
        accelerator.print(' ' + '*' * 20)
