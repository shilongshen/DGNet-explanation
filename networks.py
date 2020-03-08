"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torchvision import models
from utils import weights_init

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass


##################################################################################
# Discriminator
##################################################################################
#判别器D
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    """
        论文是这样描述鉴别器的：D（鉴别器）跟随现在流行的多尺度缩放 PatchGAN。
        我们采用了不同尺寸的图像进行输入，64x32,128x64,256x128。
        训练判别器时使用梯度惩罚以稳定训练
    """
    def __init__(self, input_dim, params, fp16):
        #input_dim表示输入通道数，默认为3
        super(MsImageDis, self).__init__()
        #判别器的层数，2
        self.n_layer = params['n_layer']
        #GAN loss，,默认为lsgan
        self.gan_type = params['gan_type']
        #最后一层的filters数目，32
        self.dim = params['dim']
        #正则化方式，normalization layer [none/bn/in/ln]，默认为none
        self.norm = params['norm']
        #激活函数，[relu/lrelu/prelu/selu/tanh]，默认为lrelu
        self.activ = params['activ']
        #图片的缩放次数，3
        self.num_scales = params['num_scales']
        ##填充类型，[zero/reflect]，默认为reflect
        self.pad_type = params['pad_type']
        #正则化的一个超参数
        self.LAMBDA = params['LAMBDA']
        #非本地的层数？，0
        self.non_local = params['non_local']
        #跳跃连接的层数，4
        self.n_res = params['n_res']
        #输入图片的通道数，3
        self.input_dim = input_dim
        #？
        self.fp16 = fp16
        #类似于一个下采样的操作
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        if not self.gan_type == 'wgan':
            self.cnns = nn.ModuleList()
            for _ in range(self.num_scales):
                Dis = self._make_net()
                Dis.apply(weights_init('gaussian'))
                self.cnns.append(Dis)
        else:
             self.cnn = self.one_cnn()

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 1, 1, 0, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            cnn_x += [Conv2dBlock(dim, dim2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        if self.non_local>1:
            cnn_x += [NonlocalBlock(dim)]
        for i in range(self.n_res):
            cnn_x += [ResBlock(dim, norm=self.norm, activation=self.activ, pad_type=self.pad_type, res_type='basic')] 
        if self.non_local>0:
            cnn_x += [NonlocalBlock(dim)]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def one_cnn(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(5):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        cnn_x += [nn.Conv2d(dim, 1, (4,2), 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    # 注意，输出的结果有3个，因为三个尺寸进行预测
    def forward(self, x):
        if not self.gan_type == 'wgan':
            outputs = []
            for model in self.cnns:
                outputs.append(model(x))
                x = self.downsample(x)
        else:
             outputs = self.cnn(x)
             outputs = torch.squeeze(outputs)
        #这里一起3个元素，分别大小为[batch_size,1，64，32], [batch_size,1，32，16], [batch_size,1，16，8]
        return outputs



###################################################################################################################
    ##判别器设定损失函数
###################################################################################################################

    def calc_dis_loss(self, model, input_fake, input_real):
        # calculate the loss to train D
        """
        # 该loss为了训练D
        :param model: D
        :param input_fake:合成的图片
        :param input_real: 真实图片
        :return:
        """
        input_real.requires_grad_()
        # 这里一起3个元素outs，分别大小为[batch_size, 1，64，32], [batch_size, 1，32，16], [batch_size, 1，16，8]
        outs0 = model.forward(input_fake)
        # 这里一起3个元素outs1，分别大小为[batch_size, 1，64，32], [batch_size, 1，32，16], [batch_size, 1，16，8]
        outs1 = model.forward(input_real)
        loss = 0
        reg = 0
        Drift = 0.001
        LAMBDA = self.LAMBDA
        # 默认gan_type = 'lsgan'，即没有执行这里
        if self.gan_type == 'wgan':
            loss += torch.mean(outs0) - torch.mean(outs1)
            # progressive gan
            loss += Drift*( torch.sum(outs0**2) + torch.sum(outs1**2))
            #alpha = torch.FloatTensor(input_fake.shape).uniform_(0., 1.)
            #alpha = alpha.cuda()
            #differences = input_fake - input_real
            #interpolates =  Variable(input_real + (alpha*differences), requires_grad=True)
            #dis_interpolates = self.forward(interpolates) 
            #gradient_penalty = self.compute_grad2(dis_interpolates, interpolates).mean()
            #reg += LAMBDA*gradient_penalty 
            reg += LAMBDA* self.compute_grad2(outs1, input_real).mean() # I suggest Lambda=0.1 for wgan
            loss = loss + reg
            return loss, reg

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            #  默认gan_type == 'lsgan'，最小二乘损失方式，主要解决生成图像不稳定的问题
            if self.gan_type == 'lsgan':
                # out0是假的图片经过model.forward(input_fake)鉴别之后的输出
                # out1是真的图片经过model.forward(input_real)鉴别之后的输出
                #当out0为0，out1为1时，即判别器能够辨别出生成图片和真实图片时，损失函数最小。
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
                # regularization
                reg += LAMBDA* self.compute_grad2(out1, input_real).mean()
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
                reg += LAMBDA* self.compute_grad2(F.sigmoid(out1), input_real).mean()
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = loss+reg
        return loss, reg

    def calc_gen_loss(self, model, input_fake):
        # calculate the loss to train G
        """
        # 生成模块的损失函数
        :param model: D
        :param input_fake: 合成图片
        :return:
        """

        # 这里一起3个元素outs，分别大小为[batch_size, 1，64，32], [batch_size, 1，32，16], [batch_size, 1，16，8]
        outs0 = model.forward(input_fake)
        loss = 0
        Drift = 0.001
        # 该处不执行，因为gan_type==‘lsgan’
        if self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
            # progressive gan
            loss += Drift*torch.sum(outs0**2)
            return loss

        for it, (out0) in enumerate(outs0):
            # 同理我们使用的是gan_type == 'lsgan'
            if self.gan_type == 'lsgan':
                # 这里我们可以看到，当out0=1的时候，其损失是最小的，同时out0是假的图片经过model.forward(input_fake)鉴别之后的输出
                #当out0=1时，即判别器被生成器欺骗，把生成器生成的图片当成了真的图片，此时的损失函数是最小的。
                loss += torch.mean((out0 - 1)**2) * 2  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    #计算梯度
    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        # 这是一个对输出自动求导数的函数，这里表示对outputs=d_out.sum()求inputs=x_in的导数
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params, fp16):
        super(AdaINGen, self).__init__()
        # 这是生成器最后一层filters的数目，默认为16
        dim = params['dim']
        # 进行编码下采样的层数，默认设置为2
        n_downsample = params['n_downsample']
        # 跳跃连接的层数
        n_res = params['n_res']
        # 使用的激活函数，默认为leakReLu
        activ = params['activ']
        # 填补的类型，如补零
        pad_type = params['pad_type']
        # 默认为512filters，mlp表示全连接层
        mlp_dim = params['mlp_dim']
        # 全连接层的正则化类型
        mlp_norm = params['mlp_norm']
        # length of appearance code，把图片编译成ap code的长度，默认为2048
        id_dim = params['id_dim']
        # 注释为basic，偏置吗？表示很懵逼，可选 # [basic/parallel/series]，默认为basic
        which_dec = params['dec']
        # dropout
        dropout = params['dropout']
        # 最后一层的激活函数
        tanh = params['tanh']
        # 非本地层数？
        non_local = params['non_local']

        # content encoder
        # 内容编码，把图片经过ContentEncoder(Es) 编码成st code的过程，进去看看就明白了，论文是这样描述Es的：
        # Es是一个输出st code（128x64x32）的浅层网络，他由四个卷积层然后接着4个跳跃连接块组成，进去查看的确如此
        # 这里创建出来的仅仅是一个类，还没有进行前向传播。input_dim为1，应该是送入的为灰度图，指示通道数
        #enc_content 是 ContentEncoder的一个对象
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dropout=dropout, tanh=tanh, res_type='basic')

        # # 获得解码器输出的特征向量的维度，self.output_dim为128维度（其实是解码器输入的维度=编码器输出的维度），请看论文Figure 2 ，其中黄色的梯形，
        # 解码过程也就是一些列的卷积，我就不详细注释了，
        #论文中是这样说的：G（图中黄色梯形）对s处理的时候，是4个卷积加四个跳跃连接块，
        #其中每个跳跃连接块中都包含了两个instance normalization层，将a当作可以缩放的偏置参数。
        #self.dec：表示ap code和st code(合成，也可没有合成)结合的latent code经过G得到合成图片，就是解码之后的
        #注意，这里也仅仅是创建了类，没有网络搭建和前向传播
        self.output_dim = self.enc_content.output_dim
        if which_dec =='basic':        
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='basic', non_local = non_local, fp16 = fp16)
        elif which_dec =='slim':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='slim', non_local = non_local, fp16 = fp16)
        elif which_dec =='series':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='series', non_local = non_local, fp16 = fp16)
        elif which_dec =='parallel':
            self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='adain', activ=activ, pad_type=pad_type, res_type='parallel', non_local = non_local, fp16 = fp16)
        else:
            ('unkonw decoder type')

        # MLP to generate AdaIN parameters？
        # 全链接层去产生AdaIN的参数，这里的输出mlp_dim=512
        # 我们可以看到w和b的维度是相同的，这个大家要知道输入id_dim=1024，输出为128*2=256
        self.mlp_w1 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w2 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w3 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w4 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        
        self.mlp_b1 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b2 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b3 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b4 = MLP(id_dim, 2*self.output_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        # 权重初始化
        self.apply(weights_init(params['init']))

    # 调用该函数Es将会对输入图像编码为st code
    def encode(self, images):
        # encode an image to its content and style codes
        content = self.enc_content(images)
        return content

    #调用该函数G将会对图像进行解码 ，合成图像
    """
        在初始化的时候，Decoder就拿到了ap code[衣服+鞋子+手机+包包等]，
        大家注意下，这里我说的ap code是经过身份信息分离的ap code?，
        也就是不包含身份信息，只存在[衣服+鞋子等]，
        然后再调用images = self.dec(content)是添加了st code，得到合成的图片。
    """
    def decode(self, content, ID):
        # decode style codes to an image
        """
                :param content: 经过Es得到的st code
                :param ID: 经过Ea得到的ap code
                :return: 合成的图片
        """
        ID1 = ID[:,:2048]
        ID2 = ID[:,2048:4096]
        ID3 = ID[:,4096:6144]
        ID4 = ID[:,6144:]
        # adain_params[batch_size, 1024]=[batch_size,256*4]
        #2048的特征是appearance特征（分成8份，分别结合），最后利用adain（assign parameter）结合到structure 特征中。
        # 通俗来说，相当于appearance 特征是W，structure 特征是x。最后结果是Wx 这样来结合。
        adain_params_w = torch.cat( (self.mlp_w1(ID1), self.mlp_w2(ID2), self.mlp_w3(ID3), self.mlp_w4(ID4)), 1)
        adain_params_b = torch.cat( (self.mlp_b1(ID1), self.mlp_b2(ID2), self.mlp_b3(ID3), self.mlp_b4(ID4)), 1)
        # 分配合适的参数，同时完成ap code和st code的统一
        self.assign_adain_params(adain_params_w, adain_params_b, self.dec)
        # 这里就是进行解码（前面仅仅创建了类）了，也就是论文中的G操作，然后返回获得的图片
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        # assign the adain_params to the AdaIN layers in model
        dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:,:dim].contiguous()
                std = adain_params_w[:,:dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1)>dim :  #Pop the parameters
                    adain_params_b = adain_params_b[:,dim:]
                    adain_params_w = adain_params_w[:,dim:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, dropout=0, tanh=False, res_type='basic')
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        # Here I change the stride to 2. 
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# 定义Es
class ContentEncoder(nn.Module):
    """
        :param n_downsample: 经过卷积下采样的次数，默认为2
        :param n_res: 跳跃链接网络的层数，默认为4
        :param input_dim:输入数据的维度，默认为1
        :param dim:默认为16，输出的维度，应该为filter数目
        :param norm:正则化的方式，可选[none/bn/in/ln]，这里选择为in
        :param activ:激活函数，这里选择的是lrelu
        :param pad_type:填补的方式，默认为'reflect'
        :param dropout:默认为0，前面提到
        :param tanh: 默认为false，表示关闭
        :param res_type:这个暂时不知道啥玩意
    """
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, dropout, tanh=False, res_type='basic'):
        super(ContentEncoder, self).__init__()
        self.model = []
        # Here I change the stride to 2.
        #Conv2dBlock:卷积，池化，正则化，激活等操作来构建
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, 2*dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *=2 # 32dim
        # downsampling blocks
        #两次下采样
        for i in range(n_downsample-1):
            self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type, res_type=res_type)]
        # 64 -> 128，ASPP是网络结构，类似于VGG
        self.model += [ASPP(dim, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        # 最后一层是否添加Tanh激活函数
        if tanh:
            self.model +=[nn.Tanh()]
        # 前面都是一个链表self.model = []，应该是通过该函数把他们链接起来
        self.model = nn.Sequential(*self.model)
        # 输出维度为128，通道数
        self.output_dim = dim
    #  x[torch.Size([batch_size, 1, 256, 128])]
    def forward(self, x):
        # 输出为128*64*32的特征向量，为st code
        return self.model(x)

class ContentEncoder_ImageNet(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder_ImageNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1) 
        # (256,128) ----> (16,8)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

# 定义G
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, dropout=0, res_norm='adain', activ='relu', pad_type='zero', res_type='basic', non_local=False, fp16 = False):
        super(Decoder, self).__init__()
        self.input_dim = dim
        self.model = []
        self.model += [nn.Dropout(p = dropout)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, res_type=res_type)]
        # non-local
        if non_local>0:
            self.model += [NonlocalBlock(dim)]
            print('use non-local!')
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type, fp16 = fp16)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        output = self.model(x)
        return output

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# enlarge the ID 2time
class Deconv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Deconv, self).__init__()
        model = []
        model += [nn.ConvTranspose2d( input_dim, output_dim, kernel_size=(2,2), stride=2)]
        model += [nn.InstanceNorm2d(output_dim)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d( output_dim, output_dim, kernel_size=(1,1), stride=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if res_type=='basic' or res_type=='nonlocal':
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type=='slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim ,dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type)]
        elif res_type=='series':
            model += [Series2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Series2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type=='parallel':
            model += [Parallel2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Parallel2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type=='nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out

class NonlocalBlock(nn.Module):
    def __init__(self, in_dim, norm='in'):
        super(NonlocalBlock, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class ASPP(nn.Module):
    # ASPP (a)
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ASPP, self).__init__()
        dim_part = dim//2
        self.conv1 = Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

        self.conv6 = []
        self.conv6 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv6 += [Conv2dBlock(dim_part,dim_part, 3, 1, 3, norm=norm, activation='none', pad_type=pad_type, dilation=3)]
        self.conv6 = nn.Sequential(*self.conv6)

        self.conv12 = []
        self.conv12 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv12 += [Conv2dBlock(dim_part,dim_part, 3, 1, 6, norm=norm, activation='none', pad_type=pad_type, dilation=6)]
        self.conv12 = nn.Sequential(*self.conv12)

        self.conv18 = []
        self.conv18 += [Conv2dBlock(dim,dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv18 += [Conv2dBlock(dim_part,dim_part, 3, 1, 9, norm=norm, activation='none', pad_type=pad_type, dilation=9)]
        self.conv18 = nn.Sequential(*self.conv18)

        self.fuse = Conv2dBlock(4*dim_part,2*dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type) 

    def forward(self, x):
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        out = torch.cat((conv1,conv6,conv12, conv18), dim=1)
        out = self.fuse(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16 = False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16 = fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



class Series2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Series2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x) + x
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Parallel2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Parallel2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x)) + self.norm(x)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            #reshape input
            out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.view(out.size(0),out.size(2))
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


