import torch
from torch import nn, einsum
from utils.utils import FeedForward,LayerNorm, GLANCE,FOCUS
import option

args=option.parse_args()

def exists(val):
    return val is not None


def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

def MSNSD_(features,scores,bs,batch_size,drop_out,ncrops,k):
    #magnitude selection and score prediction
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
    scores = scores.unsqueeze(dim=2)  # (B,32,1)

    normal_features = features[0:batch_size * 10]  # [b/2*ten,32,1024]
    normal_scores = scores[0:batch_size]  # [b/2, 32,1]

    abnormal_features = features[batch_size * 10:]
    abnormal_scores = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]
    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes).cuda()
    select_idx = drop_out(select_idx)


    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn = torch.topk(afea_magnitudes, k, dim=1)[1]
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0)
    for abnormal_feature in abnormal_features:
        feat_select_abn = torch.gather(abnormal_feature, 1,
                                       idx_abn_feat)
        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #
    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                dim=1)


    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    select_idx_normal = drop_out(select_idx_normal)
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes, k, dim=1)[1]
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)
    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0)
    for nor_fea in normal_features:
        feat_select_normal = torch.gather(nor_fea, 1,
                                          idx_normal_feat)
        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores

def MSNSD(features,scores,bs,batch_size,drop_out,ncrops,k,training):
   
    features = features
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)
    scores = scores.unsqueeze(dim=2)

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

    #features = features.permute(1, 0, 2, 3)

    #idx_score = idx.unsqueeze(2).expand([-1, -1, scores.shape[2]])
    #score = torch.mean(torch.gather(scores, 1, idx_score), dim=1)

    #nor_score = score[:batch_size]

    #temp = {}
    #total_select_feature = torch.zeros(0)
    #for i in range(batch_size):
    #    if i >= features.shape[0]:
    #        break
    #    feat_select = torch.gather(features[i], 1, idx_feat[i].unsqueeze(0).expand([ncrops, -1, -1]))
    #    for j in range(ncrops):
    #        temp[j] = temp.get(j, []) + [feat_select[j].unsqueeze(0)]
    #for j in range(ncrops):
    #    temp[j] = torch.cat(temp[j])  
    #nor_mag = torch.cat(list(temp.values()))
    
    #if bs != 1:
    #    temp = {}
    #    total_select_feature = torch.zeros(0)
    #    for i in range(batch_size, bs):
    #        feat_select = torch.gather(features[i], 1, idx_feat[i].unsqueeze(0).expand([ncrops, -1, -1]))
    #        for j in range(ncrops):
    #            temp[j] = temp.get(j, []) + [feat_select[j].unsqueeze(0)]
    #    for j in range(ncrops):
    #        temp[j] = torch.cat(temp[j])  
    #    abn_mag = torch.cat(list(temp.values()))
    #    abn_score = score[batch_size:]
    #else:
    #    abn_mag = nor_mag
    #    abn_score = nor_score
    #total_select_feature = torch.zeros(0)
    #for i, feature in enumerate(features):
    #    feat_select = torch.gather(feature, 1, idx_feat[i].unsqueeze(0).expand([ncrops, -1, -1]))
    #    total_select_feature = torch.cat((total_select_feature, feat_select))

    #nor_mag = total_select_feature[:batch_size * ncrops]
    #abn_mag = total_select_feature[batch_size * ncrops:]

    #return nor_score, abn_score, nor_mag, abn_mag, scores

class Backbone(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mgfn_type = 'gb',
        kernel = 7,
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
                #attention = FOCUS(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, kernel, padding = kernel // 2),
                #nn.Conv1d(dim, dim, 3, padding = 1),
                attention,
                FeedForward(dim, repe = ff_repe, dropout = dropout),
                #FeedForward(dim, repe = ff_repe, dropout = dropout)
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
        classes=0,
        dims = (64, 128),
        #depths = (args.depths1, args.depths2, args.depths3),
        #mgfn_types = (args.mgfn_type1,args.mgfn_type2, args.mgfn_type3),
        #depths = (args.depths1, args.depths2),
        #mgfn_types = (args.mgfn_type1, args.mgfn_type2),
        depths = (4, 4),
        mgfn_types = ("gb", "fb"),
        lokernel = 5,
        channels = 2048,
        ff_repe = 4,
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
        self.fc = nn.Linear(last_dim, 1)
        #self.fc2 = nn.Linear(10, 1)
        #self.classifier = nn.Linear(args.seg_length, classes)
        #self.pool = nn.MaxPool1d(10) # 10crop
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(args.dropout_rate)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)
        
        #self._initialize_weights()
        #self.stages.apply(self.init_weights)

    def init_weights(self, m):
        print(type(m))
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        #elif isinstance(m, nn.BatchNorm1d):
        #    nn.init.xavier_uniform_(m.weight)
        #    if m.bias is not None:
        #        m.bias.data.zero_()
        else:
            print("Unable to initalize weights for layer: " + str(type(m)))
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
        #scores = self.fc(x).squeeze()
        scores = self.fc(x)
        #scores = self.sigmoid(self.fc(x))
        #scores = scores.view(args.seg_length, bs, ncrops)
        #scores = self.fc2(scores).squeeze(dim=2)
        #scores = self.sigmoid(scores)
        #scores = scores.permute(1, 0)
        #feats = scores
        #scores = self.classifier(scores)
        #score_normal = out[:args.batch_size]
        #score_abnormal = out[args.batch_size:]
        
        #feats = x
        #feats = torch.norm(x, p=2, dim=2)
        #feats = feats.view(bs, ncrops, -1).mean(1)
        #nfeats = feats[:args.batch_size]
        #afeats = feats[args.batch_size:]
        #feats = self.fc(x).squeeze(dim=2)  # (B*10crop,32,1)
        #feats = feats.permute(1, 0)
        #feats = self.pool(feats)
        #feats = feats.permute(1, 0)
        #output = self.fc2(feats)
        #output = self.classifier(output)
        #return feats, output
        #print(x.shape)
        #print(x)
        #print(scores.shape)
        #print(scores)

        scores, feats, scores_  = MSNSD(x,scores, bs, self.batch_size, self.drop_out, ncrops, k, self.training)
        return scores, feats, scores_

        score_normal, score_abnormal, nor_feamagnitude, abn_feamagnitude, _  = MSNSD(x,scores,bs,self.batch_size,self.drop_out,ncrops,k)
        #print(score_abnormal.shape)
        #print(score_abnormal)
        #print(score_normal.shape)
        #print(score_normal)
        #print(abn_feamagnitude.shape)
        #print(abn_feamagnitude)
        #print(nor_feamagnitude.shape)
        #print(nor_feamagnitude)
        #print(scores.shape)
        #print(score_normal)
        #print(score_abnormal)
        #print(scores.shape)
        #print(nfeats.shape)
        #print(afeats.shape)
        #print(scores.shape)
        #print(scores)
        #print(feats.shape)
        #print(feats)
        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores
        #return score_abnormal, score_normal, afeats, nfeats, scores

    def forward1(self, video):
        k = 3
        bs, ncrops, t, c = video.size()
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)
        x_f = x[:,:2048,:]
        x_m = x[:,2048:,:]
        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m)
        x_f = x_f+args.mag_ratio*x_m

        return x_f, k, bs, ncrops

    def get_stages(self):
        return self.stages

    def forward2(self, x_f, backbone, conv):
        x_f = backbone(x_f)
        if exists(conv):
            x_f = conv(x_f)
        return x_f

    def forward3(self, x_f, backbone, conv, k, bs, ncrops):
        x_f = backbone(x_f)
        
        x_f_ = x_f.permute(0, 2, 1)
        x = self.to_logits(x_f_)
        scores = self.sigmoid(self.fc(x))
        score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores  = MSNSD(x,scores,bs,self.batch_size,self.drop_out,ncrops,k)

        if exists(conv):
            x_f = conv(x_f)

        return x_f, score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores


