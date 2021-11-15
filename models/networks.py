import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .vgg import Vgg19
import math
import cv2
import numpy as np
import scipy.stats as st
import code


###############################################################################
# Helper Functions
###############################################################################

class NoneNorm(torch.nn.Module):
    def __init__(self, *args):
        super(NoneNorm, self).__init__()

    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = NoneNorm
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
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'gen_drop':
        # net = Generator_drop(input_nc, output_nc, ngf)
        net = RAWRRNet(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
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
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type,  use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,  use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class CurvMap(nn.Module):
    def __init__(self, scale=1):
        super(CurvMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale
        img = TF.rgb_to_grayscale(img)
        img_pad = F.pad(img, pad=(1, 1, 1, 1), mode='reflect')

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradXX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradXY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradYY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs()
        grady = (img[..., 1:] - img[..., :-1]).abs()
        gradxx = (img_pad[..., 2:, 1:-1] + img_pad[..., :-2, 1:-1] - 2 * img_pad[..., 1:-1, 1:-1]).abs()
        gradyy = (img_pad[..., 1:-1, 2:] + img_pad[..., 1:-1, :-2] - 2 * img_pad[..., 1:-1, 1:-1]).abs()
        gradxy = (img_pad[..., 2:, 2:] + img_pad[..., 1:-1, 1:-1] - img_pad[..., 2:, 1:-1] - img_pad[..., 1:-1, 2:]).abs()

        gradX[..., :-1, :] += gradx
        gradX[..., 1:, :] += gradx
        gradX[..., 1:-1, :] /= 2

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2
        gradXX = gradxx
        gradYY = gradyy
        gradXY = gradxy

        curv = (gradYY*(1 + torch.square(gradX)) - 2 * gradXY * gradX * gradY + gradXX * (1 + torch.square(gradY))) / \
               torch.sqrt(torch.pow((torch.square(gradX) + torch.square(gradY) + 1), 3))

        return curv


# ------------- from ERRNet --------------#
class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6]
        self.indices = indices or [2]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None

    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class VGGLoss1(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss1, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class DeConvLayer(torch.nn.Sequential):
    def __init__(self, deconv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None,
                 act=None):
        super(DeConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        output_padding = 0
        if padding is None:  # auto cal
            padding = dilation * (kernel_size - stride + 1) // 2
            output_padding = dilation * (kernel_size - stride) % 2
        self.add_module('TransposeConv2d', deconv(in_channels, out_channels, kernel_size, stride, padding=padding, output_padding=output_padding, dilation=dilation))

        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None,
                 act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        if padding is None: # auto cal
            padding = 0
            paddingL = dilation * (kernel_size - stride) // 2
            paddingR = dilation * (kernel_size - stride + 1) // 2
            self.add_module('padding', nn.ReflectionPad2d((paddingL, paddingR, paddingL, paddingR)))

        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm,
                               act=None)
        self.se_layer = None
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)


def syn_data_Fan(t, r, sigma):
    sz = int(2 * np.ceil(2 * sigma) + 1)
    r = r.squeeze().numpy().transpose(1, 2, 0)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    r_blur = torch.from_numpy(r_blur.transpose(2, 0, 1)).unsqueeze_(0).float()
    blend = r_blur + t
    if torch.max(blend) > 1:
        mean = torch.mean(blend[blend > 1])
        r_blur = r_blur - 1.3 * (mean - 1)
        r_blur = torch.clamp(r_blur, min=0, max=1)
        blend = torch.clamp(t + r_blur, min=0, max=1)

        if torch.max(blend) > 1:
            mean = torch.mean(blend[blend > 1])
            r_blur = r_blur - 1.3 * (mean - 1)
            r_blur = torch.clamp(r_blur, min=0, max=1)
            blend = torch.clamp(t + r_blur, min=0, max=1)

        if torch.isnan(r_blur).any() or torch.isnan(blend).any():
            print('sigma = %f, sz = %d, mean = %f' % (sigma, sz, mean))
            code.interact(local=locals())

    return t, r_blur, blend


''' We use similar synthetic model as Zhang's(Single image reflection separation with perceptual losses.)'''
# functions for synthesizing images with reflection (details in the paper)
def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel


# create a vignetting mask
g_mask = gkern(560, 3)
g_mask = np.dstack((g_mask, g_mask, g_mask))


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    paddingl = (kernel_size - 1) // 2
    paddingr = kernel_size - 1 - paddingl
    pad = torch.nn.ReflectionPad2d((paddingl, paddingr, paddingl, paddingr))
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return nn.Sequential(pad, gaussian_filter)


def mosaic(image):
  shape = image.shape
  image1 = torch.zeros(size=(shape[0], 4, shape[2] // 2, shape[3] // 2), device=image.device)
  image1[:, 0, ...] = image[:, 0, 0::2, 0::2]
  image1[:, 1, ...] = image[:, 1, 0::2, 1::2]
  image1[:, 2, ...] = image[:, 1, 1::2, 0::2]
  image1[:, 3, ...] = image[:, 2, 1::2, 1::2]
  return image1

def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  return 0.5 - torch.sin(torch.arcsin(1.0 - 2.0 * image) / 3.0)


def smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  return 3.0 * image.pow(2) - 2.0 * image.pow(3)

class SynData:
    def __init__(self, device):
        self.g_mask = torch.tensor(g_mask.transpose(2, 0, 1)).to(device)
        self.device = device

    def __call__(self, t_rgb: torch.Tensor, r_rgb: torch.Tensor, k_sz):
        device = self.device
        t = t_rgb.pow(2.2)
        t = inverse_smoothstep(t)
        r = r_rgb.pow(2.2)
        r = inverse_smoothstep(r)

        sigma = k_sz[np.random.randint(0, len(k_sz))]
        att = 1.08 + np.random.random() / 10.0
        alpha2 = 1 - np.random.random() / 5.0
        sz = int(2 * np.ceil(2 * sigma) + 1)
        g_kernel = get_gaussian_kernel(sz, sigma)
        g_kernel = g_kernel.to(device)
        r_blur: torch.Tensor = g_kernel(r).float()
        blend: torch.Tensor = r_blur + t

        maski = (blend > 1).float()
        mean_i = torch.clamp(torch.sum(blend * maski, dim=(2, 3)) / (torch.sum(maski, dim=(2, 3)) + 1e-6), min=1).unsqueeze_(-1).unsqueeze_(-1)
        r_blur = r_blur - (mean_i - 1) * att
        r_blur = r_blur.clamp(min=0, max=1)

        h, w = r_blur.shape[2:4]
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[:, newh:newh + h, neww:neww + w].unsqueeze_(0)

        r_blur_mask = r_blur * alpha1
        blend = r_blur_mask + t * alpha2

        t_rgb = smoothstep(t)
        # t_rgb = t
        t_rgb = t_rgb.pow(1 / 2.2)
        r_blur_mask_rgb = smoothstep(r_blur_mask)
        # r_blur_mask_rgb = r_blur_mask
        r_blur_mask_rgb = r_blur_mask_rgb.pow(1 / 2.2)
        blend_rgb = blend.clamp(min=0, max=1)
        blend_rgb = smoothstep(blend_rgb)
        blend_rgb = blend_rgb
        blend_rgb = blend_rgb.pow(1 / 2.2)
        blend_rgb = blend_rgb.clamp(min=0, max=1)

        t_raw = mosaic(t)
        r_blur_mask_raw = mosaic(r_blur_mask)
        blend_raw = mosaic(blend)
        blend_raw = blend_raw.clamp(min=0, max=1)

        return t_rgb, r_blur_mask_rgb, blend_rgb.float(), t_raw, r_blur_mask_raw, blend_raw.float(), alpha2


class Generator_drop(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats):
        super(Generator_drop, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 1, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        #############
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )
        #######
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats * 8 , n_feats*4, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats*4, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats*4, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats*4, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, h, c):
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)

        x = torch.cat((x, h), 1)

        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)

        x = self.conv8(h)
        frame2 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame1 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return x, h, c, frame1, frame2


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=((kernel_size-1)*dilation//2), bias=bias, stride=stride, dilation=dilation)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Feature Enhancement Layer
class FELayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False, pool='avg'):
        super(FELayer, self).__init__()
        # global average pooling: feature --> point
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################
## Dilated Channel Attention Block (DCAB)
class DCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, dilation=1):
        super(DCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        return W_y


##########################################################################
## Non-local Attention Module
class NAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, cha):
        super(NAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        if cha == 3:
            self.conv2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), nn.PReLU())
        elif cha == 1:
            self.conv2 = nn.Sequential(conv(1, n_feat, kernel_size, bias=bias), nn.PReLU())
        elif cha == 4:
            self.conv2 = nn.Sequential(conv(4, n_feat, kernel_size, bias=bias), nn.PReLU())
        self.down = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=8, bias=bias, stride=8), nn.PReLU())
        self.NLatt = NonLocalBlock2D(n_feat, n_feat // 2)
        self.up = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat, kernel_size=8, bias=bias, stride=8), nn.Sigmoid())

    def forward(self, x, img):
        x1 = self.conv1(x)
        fea_down = self.down(self.conv2(img))
        x2 = self.up(self.NLatt(fea_down))
        x1 = x1*x2
        x1 = x1+x
        return x1


class RRM(nn.Module):
    def __init__(self,  n_feat, kernel_size, bias):
        super(RRM, self).__init__()
        self.DM_r = nn.Sequential(conv(n_feat, n_feat * 4, kernel_size, bias=bias), nn.PReLU(),
                                  nn.PixelShuffle(2),
                                  conv(n_feat, n_feat, kernel_size, bias=bias))

        # self.DM_r = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                           conv(n_feat, n_feat, kernel_size, bias=bias))
        self.GCM_r = nn.Sequential(conv(n_feat, n_feat, 1, bias=bias), nn.PReLU(),
                                   conv(n_feat, n_feat, 1, bias=bias), nn.PReLU(),
                                   conv(n_feat, n_feat, 1, bias=bias))
        self.DM_b = nn.Sequential(conv(n_feat, n_feat * 4, kernel_size, bias=bias), nn.PReLU(),
                                  nn.PixelShuffle(2),
                                  conv(n_feat, n_feat, kernel_size, bias=bias))

        # self.DM_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                           conv(n_feat, n_feat, kernel_size, bias=bias))
        self.GCM_b = nn.Sequential(conv(n_feat, n_feat, 1, bias=bias), nn.PReLU(),
                                   conv(n_feat, n_feat, 1, bias=bias), nn.PReLU(),
                                   conv(n_feat, n_feat, 1, bias=bias))


    def forward(self, fea_r, fea_b):
        fea_r = self.DM_r(fea_r)
        fea_r = self.GCM_r(fea_r)
        fea_b = self.DM_b(fea_b)
        fea_b = self.GCM_b(fea_b)

        return torch.cat([fea_r, fea_b], 1)

# ------------- from MPRNet --------------#
##########################################################################
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.encoder_level3 = [DCAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.thicken12 = nn.Sequential(nn.Conv2d(n_feat, n_feat + scale_unetfeats, 1, stride=1, padding=0, bias=False))
        self.thicken23 = nn.Sequential(nn.Conv2d(n_feat + scale_unetfeats, n_feat + (scale_unetfeats * 2), 1, stride=1,
                                                 padding=0, bias=False))

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.thicken12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.thicken23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, pool='avg'):
        super(Decoder, self).__init__()

        self.decoder_level1 = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.decoder_level3 = [DCAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = DCAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)

        # self.fe1 = FELayer(n_feat, reduction, bias=bias, pool=pool)
        # self.fe2 = FELayer(n_feat + scale_unetfeats, reduction, bias=bias, pool=pool)
        self.fe3 = FELayer(n_feat + (scale_unetfeats * 2), reduction, bias=bias, pool=pool)

        self.thin21 = nn.Conv2d(n_feat + scale_unetfeats, n_feat, 1, stride=1, padding=0, bias=False)
        self.thin32 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + scale_unetfeats, 1, stride=1, padding=0,
                                bias=False)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(self.fe3(enc3))

        x = self.thin32(dec3) + self.skip_attn2(enc2)
        dec2 = self.decoder_level2(x)

        x = self.thin21(dec2) + self.skip_attn1(enc1)
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
## Residual Block (RB)
class RB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(RB, self).__init__()
        modules_body = []
        modules_body = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ResNet(nn.Module):
    def __init__(self, n_feat, scale_resnetfeats, kernel_size, reduction, act, bias, scale_edecoderfeats, num_cab):
        super(ResNet, self).__init__()

        self.orb1 = RB(n_feat + scale_resnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = RB(n_feat + scale_resnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = RB(n_feat + scale_resnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.thin_enc1 = nn.Conv2d(n_feat + scale_edecoderfeats, n_feat, 1, stride=1, padding=0, bias=False)
        self.thin_dec1 = nn.Conv2d(n_feat + scale_edecoderfeats, n_feat, 1, stride=1, padding=0, bias=False)

        self.thin_enc2 = nn.Sequential(nn.Conv2d(n_feat + (scale_edecoderfeats * 2), n_feat + scale_edecoderfeats, 1, stride=1,
                                                 padding=0, bias=False),
                                       nn.Conv2d(n_feat + scale_edecoderfeats, n_feat, 1, stride=1, padding=0, bias=False))
        self.thin_dec2 = nn.Sequential(nn.Conv2d(n_feat + (scale_edecoderfeats * 2), n_feat + scale_edecoderfeats, 1, stride=1,
                                                 padding=0, bias=False),
                                       nn.Conv2d(n_feat + scale_edecoderfeats, n_feat, 1, stride=1, padding=0, bias=False))

        self.conv_enc1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_enc2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_enc3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))

        self.conv_dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.thin_enc1(encoder_outs[1])) + self.conv_dec2(self.thin_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.thin_enc2(encoder_outs[2])) + self.conv_dec3(self.thin_dec2(decoder_outs[2]))

        return x


##########################################################################
class RAWRRNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_edecoderfeats=20, scale_resnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(RAWRRNet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(4, n_feat, kernel_size, bias=bias),
                                           DCAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2_r = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           DCAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2_t = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                             DCAB(n_feat, kernel_size, reduction, bias=bias, act=act))


        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, csff=False)
        self.stage1_decoder_r = Decoder(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, pool='avg')
        self.stage1_rconv = conv(n_feat, 4, kernel_size, bias=bias)
        self.stage1_decoder_t = Decoder(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, pool='max')
        self.stage1_tconv = conv(n_feat, 4, kernel_size, bias=bias)


        self.stage2_orsnet_r = ResNet(n_feat, scale_resnetfeats, kernel_size, reduction, act, bias, scale_edecoderfeats,
                                    num_cab)
        self.stage2_orsnet_t = ResNet(n_feat, scale_resnetfeats, kernel_size, reduction, act, bias, scale_edecoderfeats,
                                      num_cab)

        self.nam1_r = NAM(n_feat, kernel_size=1, bias=bias, cha=4)
        self.nam1_t = NAM(n_feat, kernel_size=1, bias=bias, cha=4)


        self.concat2_r = conv(n_feat * 3, n_feat + scale_resnetfeats, kernel_size, bias=bias)
        self.concat2_t = conv(n_feat * 4 + scale_resnetfeats, n_feat + scale_resnetfeats, kernel_size, bias=bias)

        self.RRM = RRM(n_feat, kernel_size, bias)

        self.tail_r = conv(n_feat + scale_resnetfeats, 3, kernel_size, bias=bias)
        self.tail_t = conv(n_feat + scale_resnetfeats, 3, kernel_size, bias=bias)
        self.nam2_r = NAM(n_feat + scale_resnetfeats, kernel_size=1, bias=bias, cha=3)

    def forward(self, x2_img, x2_raw):
        H = x2_img.size(2)
        W = x2_img.size(3)

        x1_img = x2_img
        x1_raw = x2_raw

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1 = self.shallow_feat1(x1_raw)

        ## Process features through Encoder of Stage 1
        feat1 = self.stage1_encoder(x1)

        ## Pass features through Decoders of Stage 1
        res1_r = self.stage1_decoder_r(feat1)

        ## Output results of Stage 1
        stage1_r = self.stage1_rconv(res1_r[0]) + x2_raw

        ## Apply Non-local Attention Module (NAM)
        x1_namfeats_r = self.nam1_r(res1_r[0], stage1_r)

        ## Same operations for t in stage 1
        res1_t = self.stage1_decoder_t(feat1)
        stage1_t = self.stage1_tconv(res1_t[0]) + x2_raw
        x1_namfeats_t = self.nam1_t(res1_t[0], stage1_t)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2_r = self.shallow_feat2_r(x2_img)

        ## Concatenate NAM features of Stage 2 with shallow features of Stage 3
        x2_namfeats = self.RRM(x1_namfeats_r, x1_namfeats_t)

        x2_r = self.concat2_r(torch.cat([x2_r, x2_namfeats], 1))
        x2_r = self.stage2_orsnet_r(x2_r, feat1, res1_r)
        stage2_r = self.tail_r(x2_r) + x2_img
        x2_r_nam = self.nam2_r(x2_r, stage2_r)

        x2_t = self.shallow_feat2_t(x2_img)
        x2_t = self.concat2_t(torch.cat([x2_t, x2_r_nam, x2_namfeats], 1))
        x2_t = self.stage2_orsnet_t(x2_t, feat1, res1_t)
        stage2_t = self.tail_t(x2_t) + x2_img

        return [stage2_r, stage2_t], [stage1_r, stage1_t]