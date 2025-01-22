import torch
import random
import numpy as np

class IndexDataLoader:
    def __init__(self, dataset, indices, batch_size=1, shuffle = True):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        
        if self.shuffle:
            random.shuffle(self.indices)
        
        batch_imgs = []
        batch_lbls = []
        for idx in self.indices:
            img, lbl = self.dataset[idx]
            batch_imgs.append(img)
            batch_lbls.append(lbl)
            if len(batch_imgs) == self.batch_size:
                yield torch.from_numpy(np.array(batch_imgs)), torch.from_numpy(np.array(batch_lbls))
                batch_imgs = []
                batch_lbls = []
        
        if batch_imgs:
            yield torch.from_numpy(np.array(batch_imgs)), torch.from_numpy(np.array(batch_lbls))