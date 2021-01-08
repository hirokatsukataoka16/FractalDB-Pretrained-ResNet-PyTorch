import glob,os
import numpy as np

import torch
from PIL import Image

from torch.utils.data import Dataset

class DBLoader(Dataset):

	def __init__(self, root, phase, transform=None):
		self.transform = transform
		self.image_paths = []
		self.image_labels = []
		class_name = os.listdir(os.path.join(root, phase))
		class_name.sort()

		for (i,x) in enumerate(class_name):
			temp = glob.glob(os.path.join(root, phase, x, "*"))
			temp.sort()
			self.image_labels.extend([i]*len(temp))
			self.image_paths.extend(temp)
		
	def __getitem__(self, index):
		image_path = self.image_paths[index]
		image = Image.open(image_path).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)
		return {"image":image,"label":self.image_labels[index]}
	
	def __len__(self):
		return len(self.image_paths)
