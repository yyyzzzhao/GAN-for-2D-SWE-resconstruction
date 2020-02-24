# -*- coding:utf-8 -*-
# Coder: Yao Zhao
# Github: https://github.com/yyyzzzhao
# a video dataset for test
# ==============================================================================

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2

class VideoDataset(BaseDataset):
	"""A video dataset for test only.

	Given a test video, read and crop each frame and crop frame then test on the trained model.
	"""

	def __init__(self, opt):
		"""Initialize this dataset class.

		Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		# self.file_name = os.path.join(opt.dataroot, 'test_video.avi')
		self.file_name = os.path.join('./test_ruijin', 'test_video.avi')
		self.cap = cv2.VideoCapture(self.file_name)

	def __getitem__(self, index):
		cap_pos = [250, 150]  # we crop a view of 500*800 size and start position is cap_pos
		ret, frame = self.cap.read(index)
		frame = frame[cap_pos[0]:cap_pos[0]+500, cap_pos[1]:cap_pos[1]+800, :]

		# apply transform to frame
		transform_params = get_params(self.opt, frame.shape[:2])
		A_transform = get_transform(self.opt, transform_params, grayscale=(True))

		frame = Image.fromarray(frame)

		frame = A_transform(frame)
		img_path = 'E:/ela_reconstruction/script/AAA/test_ruijin'

		return {'A': frame, 'B': frame, 'mask': frame, 'A_paths': img_path, 'B_paths': img_path, 'mask_path': img_path}

	def __len__(self):
		return int(self.cap.get(7))  # 7 means the number of frames