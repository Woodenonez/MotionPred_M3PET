import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

import matplotlib.pyplot as plt


def conv(with_batch_norm, input_channel, output_channel, kernel_size=3, stride=1, padding=0, activate=True):
    if with_batch_norm & activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    elif ~with_batch_norm & activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    elif with_batch_norm & ~activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel)
        )
    else:
        raise(Exception('No need to use compact layers.'))
    return layer

class PELU(nn.Module):
    '''
    Description:
        A positive exponential linear unit/layer.
        Only the negative part of the exponential retains.
        The positive part is linear: y=x+1.
    '''
    def __init__(self, offset=0) -> None:
         super().__init__()
         self.offset = offset

    def forward(self, x):
        l = nn.ELU() # ELU: max(0,x)+min(0,α∗(exp(x)−1))
        return torch.add(l(x), 1+self.offset) # assure no negative sigma produces!!!

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, with_batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv(with_batch_norm, in_channels,  mid_channels, padding=1)
        self.conv2 = conv(with_batch_norm, mid_channels, out_channels, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_batch_norm=True):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, with_batch_norm=with_batch_norm))

    def forward(self, x):
        return self.down_conv(x)

class UpBlock(nn.Module): # with skip connection
    def __init__(self, in_channels, out_channels, doubleconv=True, with_batch_norm=True, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if doubleconv:
                self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels//2, with_batch_norm=with_batch_norm)
            else:
                self.conv = conv(in_channels, out_channels, with_batch_norm=with_batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            if doubleconv:
                self.conv = DoubleConv(in_channels, out_channels, with_batch_norm=with_batch_norm)
            else:
                self.conv = conv(in_channels, out_channels, with_batch_norm=with_batch_norm)

    def forward(self, x1, x2):
        # x1 is the front feature map, x2 is the skip-connection feature map
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class YNetEncoder(nn.Module):
	def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(YNetEncoder, self).__init__()
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
		))

		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True)))

		# Last MaxPool layer before passing the features into decoder
		self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

	def forward(self, x):
		# Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
		features = []
		for stage in self.stages:
			x = stage(x)
			features.append(x)
		return features

class YNetDecoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, num_waypoints=None):
		"""
		Decoder models
		:param encoder_channels: list, encoder channels, used for skip connections
		:param decoder_channels: list, decoder channels
		:param output_len: int, pred_len
		:param num_waypoints: None or int, if None -> Goal and waypoint predictor, if int -> number of waypoints
		"""
		super(YNetDecoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if num_waypoints:
			encoder_channels = [channel+num_waypoints for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)


		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature)
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x


class ENetEncoder(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, channels=(64,128,256,512,512), with_batch_norm=True):
        super(ENetEncoder,self).__init__()
        chs = channels

        self.inc = DoubleConv(in_channels, chs[0], with_batch_norm=with_batch_norm)
        self.downs = nn.ModuleList()
        for i in range(len(chs)-1):
            self.downs.append(DownBlock(chs[i], chs[i+1], with_batch_norm=with_batch_norm))
        # self.out = nn.Conv2d(512, num_classes, kernel_size=1)
        self.out = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        features = []
        x = self.inc(x)
        features.append(x) # full resolution feature
        for down in self.downs:
            x = down(x)
            features.append(x)
        x = self.out(x)
        features.append(x) # final feature
        return features

class ENetDecoder(nn.Module):
    # batch x channel x height x width
    def __init__(self, encoder_channels, decoder_channels, out_channel=1, with_batch_norm=True):
        super(ENetDecoder,self).__init__()

        self.inc = DoubleConv(encoder_channels[-1], out_channels=encoder_channels[-1], with_batch_norm=with_batch_norm)

        up_in_chs  = [encoder_channels[-1]] + decoder_channels[:-1]
        up_out_chs = up_in_chs # for bilinear
        dec_in_chs  = [enc + dec for enc, dec in zip(encoder_channels[::-1], up_out_chs)] # add feature channels
        dec_out_chs = decoder_channels
        self.decoder = nn.ModuleList()
        for in_chs, out_chs in zip(dec_in_chs, dec_out_chs):
            self.decoder.append(UpBlock(in_chs, out_chs, bilinear=True, with_batch_norm=with_batch_norm))
        
        self.out = nn.Conv2d(decoder_channels[-1], out_channel, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, features):
        features = features[::-1]
        x = self.inc(features[0])
        for feature, dec in zip(features[1:], self.decoder):
            x = dec(x, feature)
        logits = self.out(x)
        return self.softplus(logits)

class ENet(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, encoder_channels, decoder_channels, out_channel=1, with_batch_norm=True, axes=None):
        super(ENet,self).__init__()
        self.axes = axes
        self.encoder = ENetEncoder(in_channels, encoder_channels, with_batch_norm=with_batch_norm)
        self.decoder = ENetDecoder(encoder_channels, decoder_channels, out_channel=out_channel, with_batch_norm=with_batch_norm)

    def forward(self, x):
        features = self.encoder(x)
        grid = self.decoder(features)
        return features, grid


class UNetLite(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, axes=None):
        super(UNetLite,self).__init__()

        self.inc = DoubleConv(in_channels, 16, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(16, 32, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(32, 64, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(128, 256 // factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(256, 128 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up2 = UpBlock(128, 64 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up3 = UpBlock(64, 32 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up4 = UpBlock(32, 16, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

        self.axes = axes

    def forward(self, x):
        # _, [ax1,ax2] = plt.subplots(1,2); ax1.imshow(self.outl(logits)[0,-1,:].detach().cpu()), ax2.imshow(x[0,-2,:].detach().cpu())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x0 = self.up1(x5, x4)
        x0 = self.up2(x0, x3)
        x0 = self.up3(x0, x2)
        x0 = self.up4(x0, x1)
        logits = self.outc(x0)
        return logits

class UNetLite_PELU(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, axes=None):
        super(UNetLite_PELU,self).__init__()

        self.inc = DoubleConv(in_channels, 16, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(16, 32, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(32, 64, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(128, 256 // factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(256, 128 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up2 = UpBlock(128, 64 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up3 = UpBlock(64, 32 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up4 = UpBlock(32, 16, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

        self.outl = PELU(1e-6)

        self.axes = axes

    def forward(self, x):
        # _, [ax1,ax2] = plt.subplots(1,2); ax1.imshow(self.outl(logits)[0,-1,:].detach().cpu()), ax2.imshow(x[0,-2,:].detach().cpu())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x0 = self.up1(x5, x4)
        x0 = self.up2(x0, x3)
        x0 = self.up3(x0, x2)
        x0 = self.up4(x0, x1)
        logits = self.outc(x0)
        return self.outl(logits)

class UNet(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, axes=None):
        super(UNet,self).__init__()

        self.inc = DoubleConv(in_channels, 64, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(128, 256, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(256, 512, with_batch_norm=with_batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024 // factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(1024, 512 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up2 = UpBlock(512, 256 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up3 = UpBlock(256, 128 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up4 = UpBlock(128, 64, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

        self.axes = axes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':

    sample = torch.randn((1,3,200,260))
    encoder_channels = [64,128,256,512,512]
    decoder_channels = [512,512,256]
    net1 = UNet(3)
    net2 = ENet(3, encoder_channels, decoder_channels, 1)
    
    full = net1(sample)
    features, grid = net2(sample)

    print(features[-1].shape)
    print(grid.shape)
    print(full.shape)

