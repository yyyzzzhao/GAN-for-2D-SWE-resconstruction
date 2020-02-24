import torch
from .base_model import BaseModel
from . import networks
import itertools

class ColorUSGANModel(BaseModel):
	def name(self):
		return 'ColorUSGANModel'

	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		self.isTrain = opt.isTrain

		self.loss_names = ['D', 'D_real', 'D_fake', 'G', 'G_GAN', 'G_L1', 'G_CE', 'G_GAN_Feat']

		self.visual_names = ['real_A', 'real_B', 'real_mask', 'fake_B', 'fake_b', 'fake_mask']

		if self.isTrain:
			self.model_names = ['G_p1', 'G_p2', 'D']
		else:
			self.model_names = ['G_p1', 'G_p2']

		self.netG_p1 = networks.define_G_multi(opt.input_nc, opt.ngf, opt.netG_p1, opt.norm, 
												not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
		self.netG_p2 = networks.define_G(4, 3, opt.ngf, opt.netG_p2, opt.norm, 
											not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

		if self.isTrain:
			self.netD = networks.define_D(4, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, 
										opt.init_type, opt.init_gain, self.gpu_ids, opt.getIntermFeat)

		if self.isTrain:
			# define loss function
			self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
			self.criterionL1 = torch.nn.L1Loss()
			self.criterionSeg = torch.nn.BCELoss()
			self.criterionFeat = torch.nn.L1Loss()

			# initialize optimizers
			self.optimizers = []
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_p1.parameters(), self.netG_p2.parameters()),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

	def set_input(self, input):
		AtoB = self.opt.direction == 'AtoB'
		self.real_A = input['A' if AtoB else 'B'].to(self.device)
		self.real_B = input['B' if AtoB else 'B'].to(self.device)
		self.real_mask = input['mask']

	def foward(self):
		self.fake_b, self.fake_mask = self.netG_p1(self.real_A)
		combine_Ab = torch.cat((self.real_A, self.fake_b), 1)
		self.fake_B = self.netG_p2(combine_Ab, self.fake_mask.detach())

	def backward_D(self):
		"""Calculate GAN loss for the discriminator"""
		# Fake; stop backprop to the generator by detaching fake_B
		fake_AB = torch.cat((self.real_A, self.fake_B), 1)
		pred_fake_AB = self.netD(fake_AB.detach())
		self.loss_D_fake = self.criterionGAN(pred_fake_AB, False)
		# Real
		real_AB = torch.cat((self.real_A, self.real_B), 1)
		pred_real_AB = self.netD(real_AB)
		self.loss_D_real = self.criterionGAN(pred_real_AB, True)
		# combine loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		self.loss_D.backward()

	def backward_G(self):
		"""Calculate GAN and L1 loss for the generator and BCE loss for segmentation branch"""
		fake_AB = torch.cat((self.real_A, self.fake_B), 1)
		pred_fake_AB = self.netD(fake_AB)
		self.loss_G_GAN = self.criterionGAN(pred_fake_AB, True)
		self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
		# flatten fake_mask and real_mask
		fake_mask_f = self.fake_mask.view(-1)
		real_mask_f = self.real_mask.view(-1)
		self.loss_G_CE = self.criterionSeg(fake_mask_f, real_mask_f) * self.opt.lambda_seg
		# feature matching loss
		self.loss_G_GAN_Feat = 0
		if self.opt.getIntermFeat:
			real_AB = torch.cat((self.real_A, self.real_B), 1)
			pred_real_AB = self.netD(real_AB)
			feat_weights = 4.0 / (self.opt.n_layers_D + 1)
			for i in range(len(pred_fake_AB) - 1):
				self.loss_G_GAN_Feat += feat_weights * self.criterionFeat(pred_fake_AB[i], pred_real_AB[i].detach())
		# combine loss
		self.loss_G = self.loss_G_L1 + self.loss_G_CE + self.loss_G_GAN + self.loss_G_GAN_Feat
		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()
		# update D
		self.set_requires_grad(self.netD, True)
		self.optimizer_D.zero_grad()
		self.backward_D()
		self.optimizer_D.step()
		# update G
		self.set_requires_grad(self.netD, False)
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()
