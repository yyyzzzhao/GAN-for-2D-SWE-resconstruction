# -*- coding:utf-8 -*-
# Coder: Yao Zhao
# Github: https://github.com/yyyzzzhao
# ==============================================================================
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.normalization import SPADE
import torchvision
from torch.autograd import Variable

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # print(netG)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_G_multi(input_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	if which_model_netG == 'unet_128':
		netG = AaaGenerator(input_nc, 7, ngf, norm_layer=norm_layer)
	elif which_model_netG == 'unet_256':
		netG = AaaGenerator(input_nc, 8, ngf, norm_layer=norm_layer)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], getIntermFeat=False):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, getIntermFeat=getIntermFeat)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, getIntermFeat=getIntermFeat)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            # print(target_tensor)
            return self.loss(input[-1], target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

        self.model = unet_block

    def forward(self, x, seg):
        """Standard forward"""
        return self.model(x, seg)


class AaaGenerator(nn.Module):
	def __init__(self, input_nc, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d):
		super(AaaGenerator, self).__init__()
		self.num_downs = num_downs

		# construct unet encoder parts
		self.u_block_base = EncodeBlock(ngf*8, ngf*8, norm_layer=norm_layer, innermost=True)
		self.u_block_0 = EncodeBlock(ngf * 8, ngf * 8, norm_layer=norm_layer)
		self.u_block_1 = EncodeBlock(ngf * 8, ngf * 8, norm_layer=norm_layer)
		if self.num_downs == 8:
			self.u_block_2 = EncodeBlock(ngf * 8, ngf * 8, norm_layer=norm_layer)
		self.u_block_3 = EncodeBlock(ngf * 4, ngf * 8, norm_layer=norm_layer)
		self.u_block_4 = EncodeBlock(ngf * 2, ngf * 4, norm_layer=norm_layer)
		self.u_block_5 = EncodeBlock(ngf, ngf * 2, norm_layer=norm_layer)
		self.u_block_6 = EncodeBlock(input_nc, ngf, norm_layer=norm_layer, outermost=True)

		# construct two decoder
		# one is for generate rgb images
		self.gen_path0_0 = DecodeBlock(ngf*8, ngf*8, innermost=True, norm_layer=norm_layer)
		self.gen_path0_1 = DecodeBlock(ngf*8, ngf*8, norm_layer=norm_layer)
		self.gen_path0_2 = DecodeBlock(ngf*8, ngf*8, norm_layer=norm_layer)
		if self.num_downs == 8:
			self.gen_path0_3 = DecodeBlock(ngf*8, ngf*8, norm_layer=norm_layer)
		self.gen_path0_4 = DecodeBlock(ngf*8, ngf*4, norm_layer=norm_layer)
		self.gen_path0_5 = DecodeBlock(ngf*4, ngf*2, norm_layer=norm_layer)
		self.gen_path0_6 = DecodeBlock(ngf*2, ngf, norm_layer=norm_layer)
		self.gen_path0_7 = DecodeBlock(ngf, 3, outermost=True, norm_layer=norm_layer)

		# the second is for generate mask 
		self.gen_path1_0 = DecodeBlock(ngf*8, ngf*8, innermost=True, norm_layer=norm_layer)
		self.gen_path1_1 = DecodeBlock(ngf*8, ngf*8, norm_layer=norm_layer)
		self.gen_path1_2 = DecodeBlock(ngf*8, ngf*8, norm_layer=norm_layer)
		if self.num_downs == 8:
			self.gen_path1_3 = DecodeBlock(ngf*8, ngf*8, norm_layer=norm_layer)
		self.gen_path1_4 = DecodeBlock(ngf*8, ngf*4, norm_layer=norm_layer)
		self.gen_path1_5 = DecodeBlock(ngf*4, ngf*2, norm_layer=norm_layer)
		self.gen_path1_6 = DecodeBlock(ngf*2, ngf, norm_layer=norm_layer)
		self.gen_path1_7 = DecodeBlock(ngf, 1, outermost=True, is_seg=True, norm_layer=norm_layer)

	def forward(self, x):
		# encoder
		x1 = self.u_block_6(x)
		x2 = self.u_block_5(x1)
		x3 = self.u_block_4(x2)
		x4 = self.u_block_3(x3)
		if self.num_downs == 8:
			x5 = self.u_block_2(x4)
			x6 = self.u_block_1(x5)
		else:
			x6 = self.u_block_1(x4)
		x7 = self.u_block_0(x6)
		base = self.u_block_base(x7)

		# decoder 1
		d_0_0 = self.gen_path0_0(base)
		d_0_1 = self.gen_path0_1(torch.cat([d_0_0, x7], 1))
		d_0_2 = self.gen_path0_2(torch.cat([d_0_1, x6], 1))
		if self.num_downs == 8:
			d_0_3 = self.gen_path0_3(torch.cat([d_0_2, x5], 1))
			d_0_4 = self.gen_path0_4(torch.cat([d_0_3, x4], 1))
		else:
			d_0_4 = self.gen_path0_4(torch.cat([d_0_2, x4], 1))
		d_0_5 = self.gen_path0_5(torch.cat([d_0_4, x3], 1))
		d_0_6 = self.gen_path0_6(torch.cat([d_0_5, x2], 1))
		rgb = self.gen_path0_7(torch.cat([d_0_6, x1], 1))

		# decoder 2
		d_1_0 = self.gen_path1_0(base)
		d_1_1 = self.gen_path1_1(torch.cat([d_1_0, x7], 1))
		d_1_2 = self.gen_path1_2(torch.cat([d_1_1, x6], 1))
		if self.num_downs == 8:
			d_1_3 = self.gen_path1_3(torch.cat([d_1_2, x5], 1))
			d_1_4 = self.gen_path1_4(torch.cat([d_1_3, x4], 1))
		else:
			d_1_4 = self.gen_path1_4(torch.cat([d_1_2, x4], 1))
		d_1_5 = self.gen_path1_5(torch.cat([d_1_4, x3], 1))
		d_1_6 = self.gen_path1_6(torch.cat([d_1_5, x2], 1))
		mask = self.gen_path1_7(torch.cat([d_1_6, x1], 1))

		return rgb, mask



# the EncodeBlock returns a skip connection tensor and a deeper layer interation
# while the innermost return a downconv tensor.
class EncodeBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(EncodeBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.downnorm = norm_layer(inner_nc)

        self.innermost = innermost
        self.outermost = outermost

    def forward(self, x):
        if self.innermost:
            x = self.downrelu(x)
            x = self.downconv(x)
            return x
        elif self.outermost:
            x = self.downconv(x)
            return x
        else:
            x = self.downrelu(x)
            x = self.downconv(x)
            x = self.downnorm(x)
            return x



class DecodeBlock(nn.Module):
	def __init__(self, inner_nc, outer_nc, innermost=False, outermost=False, is_seg=False, norm_layer=nn.BatchNorm2d):
		super(DecodeBlock, self).__init__()
		
		# uprelu = nn.ReLU(True)
		self.uprelu = nn.ReLU()
		self.upnorm = norm_layer(outer_nc)
		self.innermost = innermost
		self.outermost = outermost
		self.is_seg = is_seg
		if outermost:
			# if not is_seg:
			self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
				# up = [upconv, nn.Tanh()]
			# else:
				# self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
				# up = [upconv, nn.Sigmoid()]
		elif innermost:
			self.upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
			# up = [uprelu, upconv, upnorm]
		else:
			self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
			# up = [uprelu, upconv, upnorm]

	def forward(self, x):
		if self.outermost:
			x = self.upconv(x)
			if self.is_seg:
				return nn.Sigmoid()(x)
			else:
				return nn.Tanh()(x)
		elif self.innermost:
			x = self.uprelu(x)
			x = self.upconv(x)
			x = self.upnorm(x)
			return x
		else:
			x = self.uprelu(x)
			x = self.upconv(x)
			x = self.upnorm(x)
			return x


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc

        spade_config_str = 'spadesyncbatch3x3'  # opt.norm_G.replace('spectral', '')

        self.downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        self.downrelu = nn.LeakyReLU(0.2, True)
        # downnorm = norm_layer(inner_nc)
        self.downnorm = SPADE(spade_config_str, inner_nc, 1)   ## opt.semantic_nc = 1
        self.uprelu = nn.ReLU(True)
        # upnorm = norm_layer(outer_nc)
        self.upnorm = SPADE(spade_config_str, outer_nc, 1)

        self.outermost = outermost
        self.innermost = innermost
        self.outer_nc = outer_nc
        self.inner_nc = inner_nc
        self.submodule = submodule
        self.use_dropout = use_dropout
        self.tan = nn.Tanh()

        if outermost:
            self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
        elif innermost:
            self.upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
        else:
            self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

    def forward(self, x, seg):
        in_x = x
        if self.outermost:
            x = self.downconv(x)   # down
            x = self.submodule(x, seg)  # submodule
            x = self.uprelu(x)  #  up
            x = self.upconv(x)
            x = self.tan(x)
            return x
        elif self.innermost:
            x = self.downrelu(x)  # down
            x = self.downconv(x)
            x = self.uprelu(x)  # up
            x = self.upconv(x)
            x = self.upnorm(x, seg)
            return torch.cat([in_x, x], 1)
        else:

            x = self.downrelu(x)  # down
            x = self.downconv(x)
            x = self.downnorm(x, seg)  
            x = self.submodule(x, seg)  # submodule
            x = self.uprelu(x)   # up
            x = self.upconv(x)
            x = self.upnorm(x, seg)
            if self.use_dropout:
                x = nn.Dropout(0.5)(x)
            return torch.cat([in_x, x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = 1
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]

#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         return self.model(input)
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        """Standard forward."""
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)



class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
