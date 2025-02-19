import torch.utils.data as data
import numpy as np
import torch
import random
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_float32_matmul_precision('medium')
import option
args=option.parse_args()

classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

class Dataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list

        self.test_mode = test_mode
        self.list = list(open(self.rgb_list_file))

    def __getitem__(self, index):
        if not self.test_mode:
            if index == 0:
                self.n_ind = list(range(810, len(self.list)))
                self.a_ind = list(range(810))
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)

            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()

            path = self.list[nindex].strip('\n')
            nfeatures = np.load(path, allow_pickle=True)
            nfeatures = np.array(nfeatures, dtype=np.float32)
            
            # if random.random() < 0.5:
            #      nfeatures = nfeatures[::-1].copy()
            nlabel = 0.0 if "Normal" in path else 1.0

            path = self.list[aindex].strip('\n')
            afeatures = np.load(path, allow_pickle=True)
            afeatures = np.array(afeatures, dtype=np.float32)
            
            # if random.random() < 0.5:
            #      afeatures = afeatures[::-1].copy()
            alabel = 0.0 if "Normal" in path else 1.0

            return nfeatures, nlabel, afeatures, alabel
    
        else:
            path = self.list[index].strip('\n')
            features = np.load(path, allow_pickle=True)
            label = 0.0 if "Normal" in path else 1.0
            return features, label, self.get_type(index)

    def get_type(self, index):
        for i, atype in enumerate(classes):
            if atype in self.list[index]:
                return torch.tensor(i)
        return -1

    def __len__(self):

        if self.test_mode:
            return len(self.list)
        else:
            return 800
