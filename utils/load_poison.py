import os
import torch
from PIL import Image


class CIFAR_load(torch.utils.data.Dataset):
    def __init__(self, root, baseset):

        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root))
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split('.')[0])
        true_img, label = self.baseset[true_index]
        return self.transform(Image.open(os.path.join(self.root,
                                            self.samples[idx]))), label, true_img