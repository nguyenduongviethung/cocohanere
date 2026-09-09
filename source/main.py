import argparse
from model.pack import Pack, get_encoder
import torch
from accelerate import Accelerator

from run.neighbor_tune import nn_train
from utils import timmer, basedir, set_seed
from run.evaluate import evaluate
from run.normal_tune import train
from model.retriever import Retriever


def get_args():
    parser = argparse.ArgumentParser(description='Semantic Code Search')
    parser.add_argument('--task', default='tune', type=str, help='eval, tune, nncl, keeptune')
    parser.add_argument('--lang', default='python', type=str, help='java, python, ruby, go, javascript, php')

    # path
    parser.add_argument('--encoder_name', default='graphcodebert', type=str, help='roberta, codebert, graphcodebert')
    parser.add_argument('--encoder_id', default='microsoft/graphcode-base', type=str,
                        help='microsoft/graphcodebert-base, microsoft/codebert-base, roberta-base')
    parser.add_argument('--encoder_load_path', default=None, type=str, help='load local model')
    parser.add_argument('--encoder_save_path', default=None, type=str, help='/result/csn/lang/')

    # data
    parser.add_argument('--nl_length', default=128, type=int, help='')
    parser.add_argument('--code_length', default=256, type=int, help='')
    parser.add_argument('--data_flow_length', default=64, type=int, help='')
    parser.add_argument('--identi', action='store_true', help='function name')
    parser.add_argument('--mask', action='store_true', help="random whole word mask")

    # run
    parser.add_argument('--n_gpu', default=4, type=int, help='')
    parser.add_argument('--num_workers', default=16, type=int, help='')
    parser.add_argument('--eval_batch_size', default=256, type=int, help='')
    parser.add_argument('--tune_batch_size', default=256, type=int, help='')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='')
    parser.add_argument('--lr', default=3e-5, type=float, help='')
    parser.add_argument('--tune_epoch', default=8, type=int, help='')
    parser.add_argument('--adam', default=1e-8, type=float, help='')

    parser.add_argument('--nn_size', default=2 ** 12, type=int, help='')
    parser.add_argument('--nn_k', default=10, type=int, help='')
    parser.add_argument('--alpha', default=1, type=float, help='')

    parser.add_argument('--no_norm_loss', action='store_true', help='loss function only nn loss')
    parser.add_argument('--knn', default='bimodal', type=str, help='bimodal, unimodal')


    args = parser.parse_args()

    args.encoder_save_path = basedir + args.encoder_save_path if args.encoder_save_path else None
    args.encoder_load_path = basedir + args.encoder_load_path if args.encoder_load_path else None
    args.valid_query_path = basedir + f'/dataset/csn/{args.lang}/valid.jsonl'
    args.test_query_path = basedir + f'/dataset/csn/{args.lang}/test.jsonl'
    args.codebase_path = basedir + f'/dataset/csn/{args.lang}/codebase.jsonl'
    args.tune_path = basedir + f'/dataset/csn/{args.lang}/train.jsonl'

    return args


@timmer
def main():
    args = get_args()

    # set_seed(123456)

    # accelerator
    accelerator = Accelerator(
        split_batches=True
    )

    accelerator.print(args.__str__())

    # dataset
    pack = Pack(args=args)

    # model
    nl_encoder, code_encoder = get_encoder(args.encoder_name,
                                           args.encoder_id)
    model = Retriever(nl_encoder=nl_encoder,
                      code_encoder=code_encoder,
                      args=args)

    if args.task == 'eval':
        model = accelerator.prepare_model(model)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(torch.load(args.encoder_load_path))
        evaluate(model=model, args=args, pack=pack, accelerator=accelerator)
    else:
        if args.task == 'tune':
            train(model=model, pack=pack, args=args, accelerator=accelerator)
        elif args.task == 'nncl':
            nn_train(model=model, pack=pack, args=args, accelerator=accelerator)


if __name__ == '__main__':
    main()
