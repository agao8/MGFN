import visdom
import numpy as np
import torch
import torch as nn
from torch import nn, einsum
from einops import rearrange
from torchvision.ops import DeformConv2d

import option

args=option.parse_args()

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32) #UCF(32,2048)
    r = np.linspace(0, len(feat), length+1, dtype=int) #(33,)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.write(str(test_info["test_PR"][-1]))
    #fo.write(str(test_info["top1"][-1]))
    #fo.write(str(test_info["top3"][-1]))
    #fo.write(str(test_info["top5"][-1]))
    fo.close()

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)

def FeedForward(dim, repe = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1, padding=0),
        #Permute((0, 2, 1)),
        #nn.Linear(dim, dim * repe),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1, padding=0)
        #nn.Linear(dim * repe, dim),
        #Permute((0, 2, 1))
    )

# MHRAs (multi-head relation aggregators)
class FOCUS(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 3
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        # self.to_v = nn.Conv1d(dim, inner_dim, 1, bias = False)
        # self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)
        # self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.bs = 2 * args.batch_size
        self.offset = torch.rand(self.bs, 2 * heads * local_aggr_kernel * local_aggr_kernel, 10, args.seg_length)
        self.mask = torch.rand(self.bs, heads * local_aggr_kernel * local_aggr_kernel, 10, args.seg_length)
        self.deform = DeformConv2d(in_channels = dim, out_channels = dim, kernel_size = local_aggr_kernel, padding = local_aggr_kernel // 2)


    def forward(self, x):
        x = self.norm(x) #(b*crop,c,t)

        b, c, t = x.shape
        x = x.view(b // 10, 10, c, t)
        x = x.permute(0, 2, 1, 3)
        padding_size = self.bs - b // 10
        if padding_size > 0:
            x = torch.cat([x, torch.zeros(padding_size, *x.shape[1:])], dim=0)
        x = self.deform(x, self.offset, self.mask)
        x = x[:b // 10]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, c, t)
        return x
    
        # b, c, *_, h = *x.shape, self.heads
        # v = self.to_v(x) #(b*crop,c,t)
        # v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h) #(b*ten*64,c/64,32)
        # out = self.rel_pos(v)
        # out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        # return self.to_out(out)


class GLANCE(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.norm = LayerNorm(dim)
        self.dim = dim
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.attn =0
        self.mha = nn.MultiheadAttention(dim, self.heads, dropout=dropout)

    def forward(self, x):
        x = self.norm(x)
        #print(x.shape)
        shape, h = x.shape, self.heads
        x = rearrange(x, 'b c ... -> b c (...)')
        #print(x.shape)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))
        #print(q.shape)
        #print(k.shape)
        #print(v.shape)
        #q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)

        #out = self.mha(q, k, v, need_weights=False)[0]
        #print(out.shape)
        #return out.permute(0, 2, 1)
        #return None
        #return rearrange(out, 'b (h n) d -> b (h d) n', h=h)


        q = q * self.scale
        

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        self.attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', self.attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h = h)
        out = self.to_out(out)

        return out.view(*shape)

