from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
from utils.utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
from torch.multiprocessing import set_start_method
from tensorboardX import SummaryWriter
import option
args=option.parse_args()
from config import *
from models.model import Model
from datasets.dataset import Dataset
from train import train
from test import test
import datetime
import os
import random



def save_config(save_path):
    path = save_path+'/'
    os.makedirs(path,exist_ok=True)
    f = open(path + "config_{}.txt".format(datetime.datetime.now()), 'w')
    for key in vars(args).keys():
        f.write('{}: {}'.format(key,vars(args)[key]))
        f.write('\n')
savepath = './ckpt/{}_{}_{}_{}'.format(args.feat_extractor, args.lr, args.batch_size, args.comment)

save_config(savepath)
log_writer = SummaryWriter(savepath)
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    

if __name__ == '__main__':
    args=option.parse_args()
    config = Config(args)
    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    device = torch.device('cuda')


    # DO NOT SHUFFLE, shuffling is handled by the Dataset class and not the DataLoader
    train_loader = DataLoader(Dataset(args, test_mode=False),
                               batch_size=args.batch_size // 2)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=args.batch_size)

    model = Model(dropout = args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    model.apply(init_weights)

    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("pretrained loaded")

    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 0.2)

    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial= args.max_epoch * num_steps,
            cycle_mul=1.,
            lr_min=args.lr * 0.2,
            warmup_lr_init=args.lr * 0.01,
            warmup_t=args.warmup * num_steps,
            cycle_limit=20,
            t_in_epochs=False,
            warmup_prefix=True,
            cycle_decay = 0.95,
        )

    test_info = {"epoch": [], "test_AUC": [], "test_PR":[]}
    #test_info = {"epoch": [], "top1": [], "top3":[], "top5":[]}

    best_AUC = -1
    best_PR = -1 # put your own path here

    # for name, value in model.named_parameters():
    #     print(name)

    #summary(model, (args.batch_size * 2, 10, 32, 1024))

    for step in tqdm(
            range(0, args.max_epoch),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        cost = train(train_loader, model, optimizer, scheduler, device, step)
        scheduler.step(step + 1)
        log_writer.add_scalar('loss_contrastive', cost, step)

        auc, pr_auc = test(test_loader, model, args, device)
        log_writer.add_scalar('auc-roc', auc, step)
        log_writer.add_scalar('pr_auc', pr_auc, step)
        #top1, top3, top5 = test(test_loader, model, args, device)
        #log_writer.add_scalar('top-1', top1, step)
        #log_writer.add_scalar('top-3', top3, step)
        #log_writer.add_scalar('top-5', top5, step)

        test_info["epoch"].append(step)
        #test_info["top1"].append(top1)
        #test_info["top3"].append(top3)
        #test_info["top5"].append(top5)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)
        #if test_info["test_AUC"][-1] > best_AUC :
        #    best_AUC = test_info["test_AUC"][-1]
        torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-x3d.pkl'.format(step))
        save_best_record(test_info, os.path.join(savepath + "/", '{}-step.txt'.format(step)))
        # if auc > best_AUC:
        #     best_AUC = auc
        #     best_state = model.state_dict()
        #     best_epoch = step
        # else:
        #     print("Loaded Epoch State:" + str(best_epoch))
        #     model.load_state_dict(best_state)

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
