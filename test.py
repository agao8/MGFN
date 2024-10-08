from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
args=option.parse_args()
from config import *
from models.mgfn import mgfn as Model
from datasets.dataset import Dataset
import train

def test(dataloader, model, args, device):
    plt.clf()
    with torch.no_grad():
        model.eval()
        #pred = torch.zeros(0)
        pred = []
        featurelen =[]
        labels = []
        class_results = [[]] * 14
        for i, inputs in tqdm(enumerate(dataloader)):
            labels.append(inputs[1].cpu().detach())
            atype = inputs[2].cpu().detach()
            input = inputs[0].to(device)
            input = input.permute(0, 2, 1, 3)
            #sig = torch.zeros(0)
            #featurelen_i = []
            #for c  in input.chunk((input.shape[2] // 2000) + 1, 2):
            for c in [input]:
                sa, sn, _, _, logits = model(c)
                #print(sa)
                #print(sn)
                #print()
                logits = torch.squeeze(logits, 1)
                logits = torch.mean(logits, 0)
                #print(logits.squeeze().mean())
                #print(labels[i])
                #print()
                sig = logits
                featurelen.append(len(sig))
                #pred = torch.cat((pred, sig))
                #sig = sig_.cpu().detach().numpy().squeeze()
            #print(i)
            #print(np.mean(sig))
            #print(np.repeat(np.array(np.mean(sig)), featurelen_i).shape)
                #pred = np.concatenate((pred, np.repeat(np.array(np.mean(sig)), featurelen_i)))
                #pred = np.concatenate((pred, sig.cpu().detach().numpy().squeeze()))
                #pred.append(sig.cpu().detach().numpy().squeeze().mean())
                pred.append(sa.cpu().detach().squeeze())
                class_results[atype].append(sa.cpu().detach().numpy().squeeze())
            #featurelen += featurelen_i
        #gt = np.load(args.gt)
        #pred = list(pred.cpu().detach().numpy())
        #pred = list(pred)
        #pred = np.repeat(np.array(pred), 16)

        for i, at in enumerate(classes):
            if i == 0:
                pass
            print(at + ": " + str(np.sum(np.array(class_results[i]) > 0.5)) + "/" + str(len(class_results[i])))
        fpr, tpr, threshold = roc_curve(labels, pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(labels, pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        return rec_auc, pr_auc

if __name__ == '__main__':
    args = option.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    model = Model()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('mgfnfinal.pkl').items()})
    auc = test(test_loader, model, args, device)
