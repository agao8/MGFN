import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import option
args=option.parse_args()
from torch import nn
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss

def smooth(arr, lamda1):
    arr1 = arr[:,:-1,:]
    arr2 = arr[:,1:,:]

    loss = torch.sum((arr2-arr1)**2)

    return lamda1*loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward_(self, feats):
        b = args.batch_size
        n_f = feats[:b]
        a_f = feats[b:]
        max_n, max_a, max_d = -1, -1, 999999
        for i, n in enumerate(n_f):
            for j, a in enumerate(a_f):
                d = torch.mean(torch.pow(torch.clamp(self.margin - F.pairwise_distance(n, a, keepdim=True), min=0.0), 2))
                if d < max_d:
                    max_n, max_a, max_d = i, j, d
        #print(self.margin - F.pairwise_distance(n_f[max_n], a_f[max_a], keepdim=True))
        #print(max_n, max_a, max_d)
        loss_con = 0.001 * max_d
        loss_n = 0
        for i, f in enumerate(n_f):
            if i != max_n:
                d = F.pairwise_distance(f, n_f[max_n], keepdim=True)
                loss_n += 1 * torch.mean(torch.pow(d, 2)) / b
        loss_a = 0
        for i, f in enumerate(a_f):
            if i != max_a:
                d = F.pairwise_distance(f, a_f[max_a], keepdim=True)
                #print(0.001 * torch.mean(torch.pow(d, 2)))
                loss_a += 1 * torch.mean(torch.pow(d, 2)) / b
        print(loss_n, loss_a, loss_con)
        #print(loss_con + loss_n + loss_a)
        return loss_con + loss_n + loss_a

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SigmoidCrossEntropyLoss(nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))




class mgfn_loss(torch.nn.Module):
    def __init__(self, alpha):
        super(mgfn_loss, self).__init__()
        self.alpha = alpha
        self.sigmoid = torch.nn.Sigmoid()
        #self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.contrastive = ContrastiveLoss()

    def forward(self, scores, feats, targets):
        loss_cls = self.criterion(scores, targets)
        bs = args.batch_size
        loss_con = 0

        # loss_con = self.contrastive.forward_(torch.norm(feats, p=1, dim=2))
        # print(loss_cls, 0.001 * loss_con, loss_cls + 0.001 * loss_con)
        # return loss_cls + 0.001 * loss_con
    
        for i in range(len(scores)):
            for j in range(i):
                label = 1 if targets[i] != targets[j] else 0
                weight = 0.001 if targets[i] != targets[j] else 1
                #weight = 1
                loss_con += weight * self.contrastive(feats[i], feats[j], label) / bs
        #print(loss_cls, 0.001 * loss_con, loss_cls + 0.001 * loss_con)
        return loss_cls + 0.001 * loss_con

        # n_feats = feats[:bs]
        # a_feats = feats[bs:]
        # loss_con = self.contrastive(n_feats, a_feats, 1)
        # loss_con_n = self.contrastive(n_feats[:bs // 2], n_feats[bs // 2:], 0)
        # loss_con_a = self.contrastive(a_feats[:bs // 2], a_feats[bs // 2:], 0)
        # #print(loss_cls, 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n), loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n))
        # return loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n)


    def forward_(self, score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()
        label = label.cuda()
        seperate = len(abn_feamagnitude) / 2

        loss_cls = self.criterion(score, label)
        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), torch.norm(nor_feamagnitude, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal
        loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2),
                                      torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),
                                      0)  # try to cluster the same class
        loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2),
                                      torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2), 0)
        loss_total = loss_cls + 0.003 * (0.003 * loss_con + loss_con_a + loss_con_n )
        
        return loss_total



def train(nloader, aloader, model, batch_size, optimizer, device,iterator = 0):
    with torch.set_grad_enabled(True):
        model.train()
        for step, ((ninput, nlabel, _), (ainput, alabel, _)) in tqdm(enumerate(
                zip(nloader, aloader))):
            input = torch.cat((ninput, ainput), 0).to(device)
            
            scores, feats, scores_ = model(input) 
            loss_sparse = sparsity(scores_[:batch_size,:,:].view(-1), batch_size, 8e-3)
            
            loss_smooth = smooth(scores_,8e-4)

            labels = torch.cat((nlabel, alabel), 0).to(device)

            loss_criterion = mgfn_loss(0.0001)
            cost = loss_criterion(scores, feats, labels) + loss_smooth + loss_sparse

            optimizer.zero_grad()
            cost.backward()

            optimizer.step()
            iterator += 1

        return  cost.item()
