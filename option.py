import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MGFN')
    parser.add_argument('--feat_extractor', default='x3d')
    parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
    parser.add_argument('--rgb-list', default='ucf_x3d_train.txt', help='list of rgb features ')
    parser.add_argument('--test-rgb-list', default='ucf_x3d_test.txt', help='list of test rgb features ')

    parser.add_argument('--comment', default='base', help='comment for the ckpt name of the training')

    #dropout rate
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='attention dropout rate')
    parser.add_argument('--lr', type=str, default=2e-4, help='learning rates for steps default:2e-4')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')


    parser.add_argument('--workers', default=0, help='number of workers in dataloader')
    parser.add_argument('--model-name', default='model', help='name to save model')
    parser.add_argument('--pretrained_ckpt', default= None, help='ckpt for pretrained model')
    parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
    parser.add_argument('--max-epoch', type=int, default=30, help='maximum iteration to train (default: 10)')
    parser.add_argument('--warmup', type=int, default=1, help='number of warmup epochs')



    args = parser.parse_args()
    return args
