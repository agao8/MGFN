import torch
from torch import nn, einsum
from utils.utils import FeedForward,LayerNorm, GLANCE,FOCUS
import option

args=option.parse_args()

def exists(val):
    return val is not None

def MSNSD(features,scores,bs,batch_size,drop_out,ncrops,k,training):
   
    features = features
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)
    scores = scores.unsqueeze(dim=2)
    #scores = scores.view(bs, ncrops, 32, 14).mean(1)

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]

    select_idx = torch.ones_like(feat_magnitudes).cuda()
    if training:
        select_idx = drop_out(select_idx) 

    feat_magnitudes_drop = feat_magnitudes * select_idx
    idx = torch.topk(feat_magnitudes, k, dim=1)[1]
    idx_feat = idx.unsqueeze(2).expand([-1, -1, features.shape[2]])
    
    features = features.view(bs, ncrops, t, f)
    
    feats = []
    for i in range(bs):
        feats += [torch.gather(features[i], 1, idx_feat[i].unsqueeze(0).expand([ncrops, -1, -1]))]

    idx_score = idx.unsqueeze(2).expand([-1, -1, scores.shape[2]])
    score = torch.mean(torch.gather(scores, 1, idx_score), dim=1)
    return score, torch.stack(feats), scores


class Backbone(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mgfn_type = 'gb',
        kernel = 3,
        dim_headnumber = 16,
        ff_repe = 4,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads = heads, dim_head = dim_headnumber, local_aggr_kernel = kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, kernel, padding = kernel // 2),
                attention,
                FeedForward(dim, repe = ff_repe, dropout = dropout),
            ]))

    def forward(self, x):
        for layers in self.layers:
            for layer in layers:
                x = x + layer(x)

        return x

# main class

class mgfn(nn.Module):
    def __init__(
        self,
        *,
        classes=2,
        dims = (64, 128),
        depths = (3, 3),
        mgfn_types = ("gb", "fb"),
        lokernel = 5,
        channels = 2048,
        ff_repe = 3,
        dim_head = 16,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv1d(channels, init_dim, kernel_size=3, stride = 1, padding = 1)

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mgfn_type = mgfn_types,
                    ff_repe = ff_repe,
                    dropout = dropout,
                    attention_dropout = attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride = 1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(last_dim)
        )
        self.batch_size =  args.batch_size
        self.fc = nn.Linear(last_dim, 1 if classes == 2 else classes)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(args.dropout_rate)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)

    # def init_weights(self, m):
    #     print(type(m))
    #     if isinstance(m, nn.Conv1d):
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    #     #elif isinstance(m, nn.BatchNorm1d):
    #     #    nn.init.xavier_uniform_(m.weight)
    #     #    if m.bias is not None:
    #     #        m.bias.data.zero_()
    #     else:
    #         print("Unable to initalize weights for layer: " + str(type(m)))

    def forward(self, video):
        
        k = 4
        bs, ncrops, t, c = video.size()
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)
        x_f = x[:,:2048,:]
        x_m = x[:,2048:,:]
        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m)
        x_f = x_f+args.mag_ratio*x_m
        for backbone, conv in self.stages:
            x_f = backbone(x_f)
            if exists(conv):
                x_f = conv(x_f)

        x_f = x_f.permute(0, 2, 1)
        x = self.to_logits(x_f)
        logits = self.fc(x)
        scores, feats, scores_  = MSNSD(x,logits, bs, self.batch_size, self.drop_out, ncrops, k, self.training)
        return scores, feats, scores_


