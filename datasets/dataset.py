import torch.utils.data as data
import numpy as np
from utils.utils import process_feat
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import option
args=option.parse_args()

classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, is_preprocessed=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.is_preprocessed = args.preprocessed

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if args.datasetname == 'UCF':
                if self.is_normal:
                    self.list = self.list[810:] # self.list[810:]
                    #print('normal list')
                    #print(self.list)
                else:
                    self.list = self.list[:810]
                    #print('abnormal list')
                    #print(self.list)

    def __getitem__(self, index):
        label = self.get_label(index)  # get video level label 0/1
        if args.datasetname == 'UCF':
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            if "Normal" not in name:
                label = torch.tensor(1.0)
        if False:
            if args.datasetname == 'UCF':
                mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
                features = np.concatenate((features,mag),axis = 2)
            elif args.datasetname == 'XD':
                mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                features = np.concatenate((features, mag), axis=1)
            return features, name, label
        else:
            if args.datasetname == 'UCF':
                if self.is_preprocessed:
                    return features, label
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, args.seg_length) #ucf(32,2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
                return divided_features, label, self.get_type(index)
            return

    def get_label(self, index):
        if self.is_normal:
        #if "Normal" in self.list[index]:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def get_type(self, index):
        for i, atype in enumerate(classes):
            if atype in self.list[index]:
                return i
        return -1

    def __len__(self):

        return len(self.list)


    def get_num_frames(self):
        return self.num_frame
