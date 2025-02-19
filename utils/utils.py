import numpy as np
import torch
from torch import nn
import option

args=option.parse_args()


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
    f = open(file_path, "w")
    f.write("epoch: {}\n".format(test_info["epoch"][-1]))
    f.write(str(test_info["test_AUC"][-1]))
    f.write("\n")
    f.write(str(test_info["test_PR"][-1]))
    f.close()

def FeedForward(dim, repe = 4, dropout=0.):
    return nn.Sequential(
        nn.Linear(dim, dim * repe),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * repe, dim),
        nn.GELU(),
    )

# MHRAs (multi-head relation aggregators)
class FOCUS(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        kernel = 3
    ):
        super().__init__()
        self.heads = heads
        #self.norm3d = nn.BatchNorm3d(dim)
        self.norm2d = nn.BatchNorm2d(dim)
        self.norm1d = nn.BatchNorm1d(dim)
        self.conv2d = nn.Conv2d(dim, dim, kernel, padding = kernel // 2, groups = heads)
        self.conv1d = nn.Conv1d(dim, dim, kernel, padding = kernel // 2, groups = heads)
        #self.pool = nn.AdaptiveAvgPool3d(1, 1, 1)


    def forward(self, x):
        B, T, H, W, C = x.shape
        #x = x.permute(0, 4, 1, 2, 3)
        x = x.view(B * T, C, H, W)
        x = self.norm2d(x)
        x = self.conv2d(x)
        x = x.view(B * H * W, C, T)
        x = self.norm1d(x)
        x = self.conv1d(x)
        #x = x.view(B, C, T, H, W)
        #x = self.norm3d(x)
        x = x.view(B, T, H, W, C)
        #x = x.permute(0, 2, 3, 4, 1)
        return x
