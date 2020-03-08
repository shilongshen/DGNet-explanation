"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis
from reIDmodel import ft_net, ft_netAB, PCB
from utils import get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import copy
import os
import cv2
import numpy as np
from random_erasing import RandomErasing
import random
import yaml

#fp16
try:
    from apex import amp
    from apex.fp16_utils import *
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1) 
    return out.cuda()

def scale2(x):
    #对输入进行缩放
    if x.size(2) > 128: # do not need to scale the input
        return x
    x = torch.nn.functional.upsample(x, scale_factor=2, mode='nearest')  #bicubic is not available for the time being.
    return x

def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def fliplr(img):
    #对图片进行水平翻转
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def update_teacher(model_s, model_t, alpha=0.999):
    for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)

def predict_label(teacher_models, inputs, num_class, alabel, slabel, teacher_style=0):
    #定义teacher model的损失函数，实际使用的是smooth dynamic label
# teacher_style:
# 0: Our smooth dynamic label
# 1: Pseudo label, hard dynamic label
# 2: Conditional label, hard static label 
# 3: LSRO, static smooth label
# 4: Dynamic Soft Two-label
# alabel is appearance label
    #实际使用损失函数的是这个
    if teacher_style == 0:
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            #outputs_t1表示身份的特征向量
            _, outputs_t1 = teacher_model(inputs) 
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs)) 
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
    elif teacher_style == 1:  # dynamic one-hot  label
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())  # change softmax to max
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
        _, dlabel = torch.max(outputs_t.data, 1)
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, dlabel[i]] = 1
    elif teacher_style == 2: # appearance label
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, alabel[i]] = 1
    elif teacher_style == 3: # LSRO
        outputs_t = torch.ones(inputs.size(0), num_class).cuda()
    elif teacher_style == 4: #Two-label
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
        mask = torch.zeros(outputs_t.shape)
        mask = mask.cuda()
        for i in range(inputs.size(0)):
            mask[i, alabel[i]] = 1
            mask[i, slabel[i]] = 1
        outputs_t = outputs_t*mask
    else:
        print('not valid style. teacher-style is in [0-3].')

    s = torch.sum(outputs_t, dim=1, keepdim=True)
    s = s.expand_as(outputs_t)
    outputs_t = outputs_t/s
    return outputs_t

######################################################################
# Load model
#---------------------------
def load_network(network, name):
    save_path = os.path.join('./models',name,'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_config(name):
    config_path = os.path.join('./models',name,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    return config


class DGNet_Trainer(nn.Module):
    #初始化函数
    def __init__(self, hyperparameters, gpu_ids=[0]):
        super(DGNet_Trainer, self).__init__()
        # 从配置文件获取生成模型的和鉴别模型的学习率
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']

        # # ID的类别，这里要注意，不同的数据集都是不一样的，应该是训练数据集的ID数目，非测试集
        ID_class = hyperparameters['ID_class']

        # 看是否设置使用float16，估计float16可以增加精确度
        if not 'apex' in hyperparameters.keys():
            hyperparameters['apex'] = False
        self.fp16 = hyperparameters['apex']

        # Initiate the networks
        # We do not need to manually set fp16 in the network for the new apex. So here I set fp16=False.
################################################################################################################
        ##这里是定义Es和G
        # 注意这里包含了两个步骤，Es编码+解码过程，既然解码（论文Figure 2的黄色梯形G）包含到这里了，下面Ea应该不会包含解码过程了
        # 因为这里是一个类，如后续gen_a.encode()可以进行编码，gen_b.encode()可以进行解码
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], fp16 = False)  # auto-encoder for domain a
        self.gen_b = self.gen_a  # auto-encoder for domain b
############################################################################################################################################

############################################################################################################################################
        ##这里是定义Ea
        # ID_stride，外观编码器池化层的stride
        if not 'ID_stride' in hyperparameters.keys():
            hyperparameters['ID_stride'] = 2

        # hyperparameters['ID_style']默认为'AB'，论文中的Ea编码器
        #这里是设置Ea，有三种模型可以选择
        #PCB模型，ft_netAB为改造后的resnet50，ft_net为resnet50
        if hyperparameters['ID_style']=='PCB':
            self.id_a = PCB(ID_class)
        elif hyperparameters['ID_style']=='AB':
            # 这是我们执行的模型，注意的是，id_a返回两个x（表示身份），获得f，具体介绍看函数内部
            # 我们使用的是ft_netAB，是代码中Ea编码的过程，也就得到 ap code的过程，除了ap code还会得到两个分类结果
            # 现在怀疑，该分类结果，可能就是行人重识别的结果
            #ID_class表示有ID_class个不同ID的行人
            self.id_a = ft_netAB(ID_class, stride = hyperparameters['ID_stride'], norm=hyperparameters['norm_id'], pool=hyperparameters['pool']) 
        else:
            self.id_a = ft_net(ID_class, norm=hyperparameters['norm_id'], pool=hyperparameters['pool']) # return 2048 now

        # 这里进行的是浅拷贝，所以我认为他们的权重是一起的，可以理解为一个
        self.id_b = self.id_a
############################################################################################################################################################

############################################################################################################################################################
        ##这里是定义D
        # 鉴别器，行人重识别，这里使用的是一个多尺寸的鉴别器，大概就是说，对图片进行几次缩放,并且对每次缩放都会预测，计算总的损失
        # 经过网络3个元素，分别大小为[batch_size,1，64，32], [batch_size,1，32，16], [batch_size,1，16，8]
        self.dis_a = MsImageDis(3, hyperparameters['dis'], fp16 = False)  # discriminator for domain a
        self.dis_b = self.dis_a # discriminator for domain b
############################################################################################################################################################

############################################################################################################################################################
        # load teachers
        # 加载老师模型
        # teacher:老师模型名称。对于DukeMTMC，您可以设置“best - duke”
        if hyperparameters['teacher'] != "":
            #teacher_name=best
            teacher_name = hyperparameters['teacher']
            print(teacher_name)
            #有这个操作，我怀疑是可以加载多个教师模型
            teacher_names = teacher_name.split(',')
            #构建老师模型
            teacher_model = nn.ModuleList()
            teacher_count = 0

            # 默认只有一个teacher_name='teacher_name'，所以其加载的模型配置文件为项目根目录models/best/opts.yaml模型
            for teacher_name in teacher_names:
                # 加载配置文件models/best/opts.yaml
                config_tmp = load_config(teacher_name)
                if 'stride' in config_tmp:
                    #stride=1
                    stride = config_tmp['stride'] 
                else:
                    stride = 2

                #  老师模型加载，老师模型为ft_net为resnet50
                model_tmp = ft_net(ID_class, stride = stride)
                teacher_model_tmp = load_network(model_tmp, teacher_name)
                # 移除原本的全连接层
                teacher_model_tmp.model.fc = nn.Sequential()  # remove the original fc layer in ImageNet
                teacher_model_tmp = teacher_model_tmp.cuda()
                # summary(teacher_model_tmp, (3, 224, 224))
                #使用浮点型
                if self.fp16:
                    teacher_model_tmp = amp.initialize(teacher_model_tmp, opt_level="O1")
                teacher_model.append(teacher_model_tmp.cuda().eval())
                teacher_count +=1
            self.teacher_model = teacher_model
            # 选择是否使用bn
            if hyperparameters['train_bn']:
                self.teacher_model = self.teacher_model.apply(train_bn)
############################################################################################################################################################


        # 实例正则化
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # RGB to one channel
        # 默认设置signal=gray，Es的输入为灰度图
        if hyperparameters['single']=='edge':
            self.single = to_edge
        else:
            self.single = to_gray(False)

        # Random Erasing when training
        #earsing_p表示随机擦除的概率
        if not 'erasing_p' in hyperparameters.keys():
            self.erasing_p = 0
        else:
            self.erasing_p = hyperparameters['erasing_p']
        #随机擦除矩形区域的一些像素，应该类似于数据增强
        self.single_re = RandomErasing(probability = self.erasing_p, mean=[0.0, 0.0, 0.0])
        # 设置T_w为1,T_w为primary feature learning loss的权重系数
        if not 'T_w' in hyperparameters.keys():
            hyperparameters['T_w'] = 1

        ################################################################################################
        # Setup the optimizers
        # 设置优化器参数
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) #+ list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) #+ list(self.gen_b.parameters())
        #使用Adams优化器，用Adams训练Es，G,D
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # id params
        # 因为ID_style默认为AB，所以这里不执行
        if hyperparameters['ID_style']=='PCB':
            ignored_params = (list(map(id, self.id_a.classifier0.parameters() ))
                            +list(map(id, self.id_a.classifier1.parameters() ))
                            +list(map(id, self.id_a.classifier2.parameters() ))
                            +list(map(id, self.id_a.classifier3.parameters() ))
                            )
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            #Ea 的优化器
            self.id_opt = torch.optim.SGD([
                 {'params': base_params, 'lr': lr2},
                 {'params': self.id_a.classifier0.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier1.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier2.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier3.parameters(), 'lr': lr2*10}
            ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)

        #     这里是我们执行的代码
        elif hyperparameters['ID_style']=='AB':
            # 忽略的参数，应该是适用于'PCB'或者其他的，但是不适用于'AB'的
            ignored_params = (list(map(id, self.id_a.classifier1.parameters()))
                            + list(map(id, self.id_a.classifier2.parameters())))
            # 获得基本的配置参数，如学习率
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']

            #对Ea使用SGD
            self.id_opt = torch.optim.SGD([
                 {'params': base_params, 'lr': lr2},
                 {'params': self.id_a.classifier1.parameters(), 'lr': lr2*10},
                 {'params': self.id_a.classifier2.parameters(), 'lr': lr2*10}
            ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, self.id_a.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
            lr2 = hyperparameters['lr2']
            self.id_opt = torch.optim.SGD([
                 {'params': base_params, 'lr': lr2},
                 {'params': self.id_a.classifier.parameters(), 'lr': lr2*10}
            ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)

        # 选择各个网络的优化
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters)
        self.id_scheduler.gamma = hyperparameters['gamma2']

        #ID Loss
        #交叉熵损失函数
        self.id_criterion = nn.CrossEntropyLoss()
        # KL散度
        self.criterion_teacher = nn.KLDivLoss(size_average=False)


        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        # save memory
        if self.fp16:
            # Name the FP16_Optimizer instance to replace the existing optimizer
            assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
            self.gen_a = self.gen_a.cuda()
            self.dis_a = self.dis_a.cuda()
            self.id_a = self.id_a.cuda()

            self.gen_b = self.gen_a
            self.dis_b = self.dis_a
            self.id_b = self.id_a

            self.gen_a, self.gen_opt = amp.initialize(self.gen_a, self.gen_opt, opt_level="O1")
            self.dis_a, self.dis_opt = amp.initialize(self.dis_a, self.dis_opt, opt_level="O1")
            self.id_a, self.id_opt = amp.initialize(self.id_a, self.id_opt, opt_level="O1")

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out.cuda()
        for i in range(x.size(0)):
            out[i,:,:,:] = self.single_re(x[i,:,:,:])
        return out

    # L1 loss，（差的绝对值）
    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))
    #L1 loss 开根号（（差的绝对值后开根号））
    def recon_criterion_sqrt(self, input, target):
        diff = input - target
        return torch.mean(torch.sqrt(torch.abs(diff[:])+1e-8))
    # L2 loss
    def recon_criterion2(self, input, target):
        diff = input - target
        return torch.mean(diff[:]**2)
    # cos loss
    def recon_cos(self, input, target):
        cos = torch.nn.CosineSimilarity()
        cos_dis = 1 - cos(input, target)
        return torch.mean(cos_dis[:])

    # x_a,x_b, xp_a, xp_b[4, 3, 256, 128],
    # 第一个参数表示bitch size,第二个参数表示输入通道数，第三个参数表示输入图片的高度，第四个参数表示输入图片的宽度
    def forward(self, x_a, x_b, xp_a, xp_b):
        #送入x_a，x_b两张图片（来自训练集不同ID）
        #通过st编码器，编码成两个stcode，structure code
        # s_a[batch,128,64,32]
        # s_b[batch,128,64,32]
        # single会根据参数设定判断是否转化为灰度图
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))

        # 先把图片进行下采样，图示我们可以看到ap code的体积比st code是要小的,这样会出现一个情况，那么他们是没有办法直接融合的，所以后面有个全链接成把他们统一
        # f_a[batch_size,2024*4=8192],     p_a[0]=[batch_size, class_num=751], p_a[1]=[batch_size, class_num=751]
        # f_b[batch_size,2024*4=8192],     p_b[0]=[batch_size, class_num=751], p_b[1]=[batch_size, class_num=751]
        # f代表的是经过ap编码器得到的ap code,
        # p表示对身份的预测（有两个身份预测，也就是p_a了两个元素，这里不好解释），
        # 前面提到过，ap编码器，不仅负责编码，还要负责身份的预测（行人重识别），也是我们落实项目的关键所在
        # 这里是第一个重难点，在论文的翻译中提到过，后续详细讲解
        f_a, p_a = self.id_a(scale2(x_a))
        f_b, p_b = self.id_b(scale2(x_b))

        # 进行解码操作，就是Figure 2中的黄色梯形G操作，这里的x_a，与x_b进行衣服互换，不同ID
        # s_b[batch,128,64,32] f_a[batch_size,2028,4,1] -->  x_ba[batch_size,3,256,128]
        x_ba = self.gen_a.decode(s_b, f_a)
        x_ab = self.gen_b.decode(s_a, f_b)

        #同一张图片进行重构，相当于autoencoder
        x_a_recon = self.gen_a.decode(s_a, f_a)
        x_b_recon = self.gen_b.decode(s_b, f_b)

        fp_a, pp_a = self.id_a(scale2(xp_a))
        fp_b, pp_b = self.id_b(scale2(xp_b))

        # decode the same person
        #x_a，xp_a表示同ID的不同图片，以下即表示同ID不同图片的重构
        x_a_recon_p = self.gen_a.decode(s_a, fp_a)
        x_b_recon_p = self.gen_b.decode(s_b, fp_b)

        # Random Erasing only effect the ID and PID loss.
        #把图片擦除一些像素，然后进行ap code编码
        if self.erasing_p > 0:
            #先把每一张图片都擦除一些像素
            x_a_re = self.to_re(scale2(x_a.clone()))
            x_b_re = self.to_re(scale2(x_b.clone()))
            xp_a_re = self.to_re(scale2(xp_a.clone()))
            xp_b_re = self.to_re(scale2(xp_b.clone()))

            # 然后经过编码成ap code，暂时不知道作用，感觉应该是数据增强
            # 类似于，擦除了图片的一些像素，但是已经能够识别出来这些图片是谁
            _, p_a = self.id_a(x_a_re)
            _, p_b = self.id_b(x_b_re)
            # encode the same ID different photo
            _, pp_a = self.id_a(xp_a_re) 
            _, pp_b = self.id_b(xp_b_re)


        # 混合合成图片：x_ab[images_a的st，images_b的ap]    混合合成图片x_ba[images_b的st，images_a的ap]
        # s_a[输入图片images_a经过Es编码得到的 st code]     s_b[输入图片images_b经过Es编码得到的 st code]
        # f_a[输入图片images_a经过Ea编码得到的 ap code]     f_b[输入图片images_b经过Ea编码得到的 ap code]
        # p_a[输入图片images_a经过Ea编码进行身份ID的预测]    p_b[输入图片images_b经过Ea编码进行身份ID的预测]
        # pp_a[输入图片pos_a经过Ea编码进行身份ID的预测]      pp_b[输入图片pos_b经过Ea编码进行身份ID的预测]
        # x_a_recon[输入图片images_a（s_a）与自己（f_a）合成的图片，当然和images_a长得一样]
        # x_b_recon[输入图片images_b（s_b）与自己（f_b）合成的图片，当然和images_b长得一样]
        # x_a_recon_p[输入图片images_a（s_a）与图片pos_a（fp_a）合成的图片，当然和images_a长得一样]
        # x_b_recon_p[输入图片images_a（s_a）与图片pos_b（fp_b）合成的图片，当然和images_b长得一样]

        return x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p

    def gen_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b, xp_a, xp_b, l_a, l_b, hyperparameters, iteration, num_gpu):
        """

        :param x_ab:[images_a的st，images_b的ap]
        :param x_ba:[images_b的st，images_a的ap]
        :param s_a:[输入图片images_a经过Es编码得到的 st code]
        :param s_b:[输入图片images_b经过Es编码得到的 st code]
        :param f_a:[输入图片images_a经过Ea编码得到的 ap code]
        :param f_b:[输入图片images_b经过Ea编码得到的 ap code]
        :param p_a:[输入图片images_a经过Ea编码进行身份ID的预测]
        :param p_b:[输入图片images_b经过Ea编码进行身份ID的预测]
        :param pp_a:[输入图片pos_a经过Ea编码进行身份ID的预测]
        :param pp_b:[输入图片pos_b经过Ea编码进行身份ID的预测]
        :param x_a_recon:[输入图片images_a（s_a）与自己（f_a）合成的图片，当然和images_a长得一样]
        :param x_b_recon:[输入图片images_b（s_b）与自己（f_b）合成的图片，当然和images_b长得一样]
        :param x_a_recon_p:[输入图片images_a（s_a）与图片pos_a（fp_a）合成的图片，当然和images_a长得一样]
        :param x_b_recon_p:[输入图片images_b（s_b）与图片pos_b（fp_b）合成的图片，当然和images_b长得一样]
        :param x_a:images_a
        :param x_b:images_b
        :param xp_a:pos_a
        :param xp_b:pos_b
        :param l_a:labels_a
        :param l_b:labels_b
        :param hyperparameters:
        :param iteration:
        :param num_gpu:
        :return:
        """
        # ppa, ppb is the same person？
        self.gen_opt.zero_grad()#梯度清零
        self.id_opt.zero_grad()
 
        # no gradient
        # 对合成x_ba和x_ab分别进行一份拷贝
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)

        rand_num = random.uniform(0,1)
        #################################
        # encode structure
        # enc_content是类ContentEncoder对象
        if hyperparameters['use_encoder_again']>=rand_num:
            # encode again (encoder is tuned, input is fixed)
            # Es编码得到s_a_recon与s_b_recon即st code
            # 如果是理想模型，s_a_recon=s_a, s_b_recon=s_b
            s_a_recon = self.gen_b.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen_a.enc_content(self.single(x_ba_copy))
        else:
            # copy the encoder
            # 这里的是深拷贝
            #enc_content_copy=gen_a.enc_content
            self.enc_content_copy = copy.deepcopy(self.gen_a.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy(self.single(x_ab))
            s_b_recon = self.enc_content_copy(self.single(x_ba))

        #################################
        # encode appearance
        #id_a_copy=id_a=Ea
        self.id_a_copy = copy.deepcopy(self.id_a)
        self.id_a_copy = self.id_a_copy.eval()
        if hyperparameters['train_bn']:
            self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_a_copy
        # encode again (encoder is fixed, input is tuned)
        # 对混合生成的图片x_ba，x_ab进行Es编码操作，同时对身份进行鉴别#
        # f_a_recon，f_b_recon表示的ap code，p_a_recon，p_b_recon表示对身份的鉴别
        f_a_recon, p_a_recon = self.id_a_copy(scale2(x_ba))
        f_b_recon, p_b_recon = self.id_b_copy(scale2(x_ab))

        # teacher Loss
        #  Tune the ID model
        log_sm = nn.LogSoftmax(dim=1)
        #如果使用了教师网络
        #默认ID_style为AB
        if hyperparameters['teacher_w'] >0 and hyperparameters['teacher'] != "":
            if hyperparameters['ID_style'] == 'normal':
                #p_a_student表示x_ba_copy的身份编码，使用的是Ea进行身份编码，也就是使用学生模型进行身份编码
                _, p_a_student = self.id_a(scale2(x_ba_copy))
                #对p_a_student使用logsoftmax，输出结果为x_ba_copy像某张图片的概率（就是一个分布）
                p_a_student = log_sm(p_a_student)
                #使用教师模型对生成图像x_ba_copy进行分类，输出结果为x_ba_copy像某张图片的概率（就是一个分布）
                p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy), num_class = hyperparameters['ID_class'], alabel = l_a, slabel = l_b, teacher_style = hyperparameters['teacher_style'])
                #通过最小化KL散度损失函数，目的是让分布p_a_student与p_a_teacher尽可能的一致
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)

                #对x_ab_copy进行同样的操作
                _, p_b_student = self.id_b(scale2(x_ab_copy))
                p_b_student = log_sm(p_b_student)
                p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy), num_class = hyperparameters['ID_class'], alabel = l_b, slabel = l_a, teacher_style = hyperparameters['teacher_style'])
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)

            #######################################################################################################################################################################################################
            # primary feature learning loss
            #######################################################################################################################################################################################################
            #  ID_style为AB
            elif hyperparameters['ID_style'] == 'AB':
                # normal teacher-student loss
                # BA -> LabelA(smooth) + LabelB(batchB)
                # 合成的图片经过身份鉴别器，得到每个ID可能性的概率，注意这里去的是p_ba_student[0]，我们知有两个身份预测结果，这里只取了一个
                # 并且赋值给了p_a_student，用于和教师模型结合的，共同计算损失
                #p_a_student分为两个部分，p_a_student[0]表示L_prim,p_a_student[1]表示L_fine。
                _, p_ba_student = self.id_a(scale2(x_ba_copy))# f_a, s_b
                p_a_student = log_sm(p_ba_student[0])

                with torch.no_grad():
                    ##使用教师模型对生成图像x_ba_copy进行分类，输出结果为x_ba_copy像某张图片(x_a/x_b)的概率（就是一个分布）
                    p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy), num_class = hyperparameters['ID_class'], alabel = l_a, slabel = l_b, teacher_style = hyperparameters['teacher_style'])

                # criterion_teacher = nn.KLDivLoss(size_average=False)
                # 计算离散距离，可以理解为p_a_student与p_a_teacher每个元素的距离之和，然后除以p_a_student.size(0)取平均值
                # 就是说学生网络（Ea）的预测越与教师网络结果相同，则是最好的
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)

                # 对另一张合成图片进行同样的操作
                _, p_ab_student = self.id_b(scale2(x_ab_copy)) # f_b, s_a
                p_b_student = log_sm(p_ab_student[0])
                with torch.no_grad():
                    p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy), num_class = hyperparameters['ID_class'], alabel = l_b, slabel = l_a, teacher_style = hyperparameters['teacher_style'])
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)
                ########################################################################################################################################################################################################

                ########################################################################################################################################################################################################
                #fine—grained feature mining loss
                ########################################################################################################################################################################################################
                # branch b loss
                # here we give different label
                # p_ba_student[1]表示的是f_fine特征，l_b表示的是images_b,即为生成图像提供st code 的图片
                loss_B = self.id_criterion(p_ba_student[1], l_b) + self.id_criterion(p_ab_student[1], l_a)
                #######################################################################################################################################################################################################

                # 对两部分损失进行权重调整
                self.loss_teacher = hyperparameters['T_w'] * self.loss_teacher + hyperparameters['B_w'] * loss_B
        else:
            self.loss_teacher = 0.0


        ## 剩下的就是重构图像之间的损失了
        # 前面提到，重构和合成是不一样的，重构是构建出来和原来图片一样的图片
        # 所以也就是可以把重构的图片和原来的图像直接计算像素直接的插值
        # 但是合成的图片是没有办法的，因为训练数据集是没有合成图片的，所以，没有办法计算像素之间的损失
        # #######################################################################################################################################################################################################
        # auto-encoder image reconstruction
        # 同ID图像进行重构时的损失函数
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)
        # #######################################################################################################################################################################################################

        #######################################################################################################################################################################################################
        # feature reconstruction
        # 不同ID图像进行图像合成时，为了保证合成图像的st code和ap code与为合成图像提供st code 和 ap code保持一致所使用的损失函数
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if hyperparameters['recon_f_w'] > 0 else 0
        # #######################################################################################################################################################################################################

        # 又一次进行图像合成
        x_aba = self.gen_a.decode(s_a_recon, f_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(s_b_recon, f_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None


        # ID loss AND Tune the Generated image
        if hyperparameters['ID_style']=='PCB':
            self.loss_id = self.PCB_loss(p_a, l_a) + self.PCB_loss(p_b, l_b)
            self.loss_pid = self.PCB_loss(pp_a, l_a) + self.PCB_loss(pp_b, l_b)
            self.loss_gen_recon_id = self.PCB_loss(p_a_recon, l_a) + self.PCB_loss(p_b_recon, l_b)


        ########################################################################################################################################################################################################
        #   使用的是  ['ID_style']=='AB'
        elif hyperparameters['ID_style']=='AB':
            weight_B = hyperparameters['teacher_w'] * hyperparameters['B_w']
            #计算的是L^s_id
            self.loss_id = self.id_criterion(p_a[0], l_a) + self.id_criterion(p_b[0], l_b) \
                         + weight_B * ( self.id_criterion(p_a[1], l_a) + self.id_criterion(p_b[1], l_b) )

            #对同ID不同图片计算L^s_id
            self.loss_pid = self.id_criterion(pp_a[0], l_a) + self.id_criterion(pp_b[0], l_b) #+ weight_B * ( self.id_criterion(pp_a[1], l_a) + self.id_criterion(pp_b[1], l_b) )

            # 对生成图像计算L^C_id
            self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a) + self.id_criterion(p_b_recon[0], l_b)
        ########################################################################################################################################################################################################

        else:
            self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b)
            self.loss_pid = self.id_criterion(pp_a, l_a) + self.id_criterion(pp_b, l_b)
            self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b)

        #print(f_a_recon, f_a)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0


        ########################################################################################################################################################################################################
        # GAN loss
        #计算生成器G的对抗损失函数
        ########################################################################################################################################################################################################
        if num_gpu>1:
            self.loss_gen_adv_a = self.dis_a.module.calc_gen_loss(self.dis_a, x_ba)
            self.loss_gen_adv_b = self.dis_b.module.calc_gen_loss(self.dis_b, x_ab)
        else:
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(self.dis_a, x_ba)
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(self.dis_b, x_ab)
        ########################################################################################################################################################################################################

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])

        if iteration > hyperparameters['warm_teacher_iter']:
            hyperparameters['teacher_w'] += hyperparameters['warm_scale']
            hyperparameters['teacher_w'] = min(hyperparameters['teacher_w'], hyperparameters['max_teacher_w'])
        # total loss，计算总的loss
        #1个teacher loss+4个同ID图片重构loss+4个不同ID图片合成loss++3个ID loss+2个生成器loss、
        #teacher loss包括了primary feature learning loss和fine_grain mining loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['id_w'] * self.loss_id + \
                              hyperparameters['pid_w'] * self.loss_pid + \
                              hyperparameters['recon_id_w'] * self.loss_gen_recon_id + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['teacher_w'] * self.loss_teacher

        if self.fp16:
            with amp.scale_loss(self.loss_gen_total, [self.gen_opt,self.id_opt]) as scaled_loss:
                scaled_loss.backward()
            self.gen_opt.step()
            self.id_opt.step()
        else:
            self.loss_gen_total.backward()#计算梯度
            self.gen_opt.step()#梯度更新
            self.id_opt.step()#梯度更新
        print("L_total: %.4f, L_gan: %.4f,  Lx: %.4f, Lxp: %.4f, Lrecycle:%.4f, Lf: %.4f, Ls: %.4f, Recon-id: %.4f, id: %.4f, pid:%.4f, teacher: %.4f"%( self.loss_gen_total, \
                                                        hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b), \
                                                        hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b), \
                                                        hyperparameters['recon_xp_w'] * (self.loss_gen_recon_xp_a + self.loss_gen_recon_xp_b), \
                                                        hyperparameters['recon_x_cyc_w'] * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b), \
                                                        hyperparameters['recon_f_w'] * (self.loss_gen_recon_f_a + self.loss_gen_recon_f_b), \
                                                        hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b), \
                                                        hyperparameters['recon_id_w'] * self.loss_gen_recon_id, \
                                                        hyperparameters['id_w'] * self.loss_id,\
                                                        hyperparameters['pid_w'] * self.loss_pid,\
hyperparameters['teacher_w'] * self.loss_teacher )  )

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def PCB_loss(self, inputs, labels):
       loss = 0.0
       for part in inputs:
           loss += self.id_criterion(part, labels)
       return loss/len(inputs)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ab1, x_aba, x_bab = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            s_a = self.gen_a.encode( self.single(x_a[i].unsqueeze(0)) )
            s_b = self.gen_b.encode( self.single(x_b[i].unsqueeze(0)) )
            f_a, _ = self.id_a( scale2(x_a[i].unsqueeze(0)))
            f_b, _ = self.id_b( scale2(x_b[i].unsqueeze(0)))
            x_a_recon.append(self.gen_a.decode(s_a, f_a))
            x_b_recon.append(self.gen_b.decode(s_b, f_b))
            x_ba = self.gen_a.decode(s_b, f_a)
            x_ab = self.gen_b.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ab1.append(x_ab)
            #cycle
            s_b_recon = self.gen_a.enc_content(self.single(x_ba))
            s_a_recon = self.gen_b.enc_content(self.single(x_ab))
            f_a_recon, _ = self.id_a(scale2(x_ba))
            f_b_recon, _ = self.id_b(scale2(x_ab))
            x_aba.append(self.gen_a.decode(s_a_recon, f_a_recon))
            x_bab.append(self.gen_b.decode(s_b_recon, f_b_recon))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ba1, x_ab1 = torch.cat(x_ba1), torch.cat(x_ab1)
        self.train()

        return x_a, x_a_recon, x_aba, x_ab1, x_b, x_b_recon, x_bab, x_ba1

    def dis_update(self, x_ab, x_ba, x_a, x_b, hyperparameters, num_gpu):
        self.dis_opt.zero_grad()#梯度清零
        # D loss
        #计算判别器的损失函数，然后计算梯度，进行梯度更新
        #输入为（x_ba，x_a），（x_ab，x_b）两对图片，损失为两对图片的总和
        if num_gpu>1:
            self.loss_dis_a, reg_a = self.dis_a.module.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.module.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        else:
            # 计算判别器的损失函数
            self.loss_dis_a, reg_a = self.dis_a.calc_dis_loss(self.dis_a, x_ba.detach(), x_a)
            self.loss_dis_b, reg_b = self.dis_b.calc_dis_loss(self.dis_b, x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        print("DLoss: %.4f"%self.loss_dis_total, "Reg: %.4f"%(reg_a+reg_b) )
        if self.fp16:
            with amp.scale_loss(self.loss_dis_total, self.dis_opt) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_dis_total.backward()#计算梯度
        self.dis_opt.step()#梯度更新

    def update_learning_rate(self):
        #对学习率的更新
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b = self.gen_a
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b = self.dis_a
        # Load ID dis
        last_model_name = get_model_list(checkpoint_dir, "id")
        state_dict = torch.load(last_model_name)
        self.id_a.load_state_dict(state_dict['a'])
        self.id_b = self.id_a
        # Load optimizers
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
            self.id_opt.load_state_dict(state_dict['id'])
        except:
            pass
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, num_gpu=1):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        id_name = os.path.join(snapshot_dir, 'id_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict()}, gen_name)
        if num_gpu>1:
            torch.save({'a': self.dis_a.module.state_dict()}, dis_name)
        else:
            torch.save({'a': self.dis_a.state_dict()}, dis_name)
        torch.save({'a': self.id_a.state_dict()}, id_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'id': self.id_opt.state_dict(),  'dis': self.dis_opt.state_dict()}, opt_name)



