"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from trainer import DGNet_Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy.random as random
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()

#配置文件路径
parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')

#输出的根目录
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

#上次模型中断，保存模型在输出目录下的目录名字
parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")

#是否继续训练，如果之前中断过，继续训练设置为store_false,重零开始训练设为store_true
parser.add_argument("--resume", action="store_true")

# 选择训练的的网络把，不是很明白，可能是为了以后扩展，或者对比其他的模型，预留了选择，为以后加入模型提供方便，
# 不用想太多，我们默认使用DGNet即可
parser.add_argument('--trainer', type=str, default='DGNet', help="DGNet")

#指定训练的GPU
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
opts = parser.parse_args()

#对GPU进行一些操作
str_ids = opts.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gpu_ids.append(int(str_id))
num_gpu = len(gpu_ids)

cudnn.benchmark = True

# Load experiment setting
#如果是在上次训练模型的中断基础上进行训练，则获得上次训练模型的配置文件
if opts.resume:
    config = get_config('./outputs/'+opts.name+'/config.yaml')
#否则就用config指定的yaml配置文件
else:
    config = get_config(opts.config)
#最大迭代次数，默认迭代100000次后停止训练
max_iter = config['max_iter']

#outputs/latest/images目录下图片中，每行人的数目
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
#设置模型和数据加载,模型为trainer.py中的DGNet_Trainer
if opts.trainer == 'DGNet':
    trainer = DGNet_Trainer(config, gpu_ids)
    trainer.cuda()

#设定随机种子
random.seed(7) #fix random result
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
#？
train_a_rand = random.permutation(train_loader_a.dataset.img_num)[0:display_size] 
train_b_rand = random.permutation(train_loader_b.dataset.img_num)[0:display_size] 
test_a_rand = random.permutation(test_loader_a.dataset.img_num)[0:display_size] 
test_b_rand = random.permutation(test_loader_b.dataset.img_num)[0:display_size] 

#把所有训练数据拼接起来
train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in train_a_rand]).cuda()
train_display_images_ap = torch.stack([train_loader_a.dataset[i][2] for i in train_a_rand]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in train_b_rand]).cuda()
train_display_images_bp = torch.stack([train_loader_b.dataset[i][2] for i in train_b_rand]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in test_a_rand]).cuda()
test_display_images_ap = torch.stack([test_loader_a.dataset[i][2] for i in test_a_rand]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in test_b_rand]).cuda()
test_display_images_bp = torch.stack([test_loader_b.dataset[i][2] for i in test_b_rand]).cuda()

# Setup logger and output folders
#设置输出目录和打印log
#
    # 如果不是继续训练,也就是重新训练，拷贝一些文件到outputs/latest下面，其目的是为了，保留了训练文件和网络结构。
if not opts.resume:
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    shutil.copyfile('trainer.py', os.path.join(output_directory, 'trainer.py')) # copy file to output folder
    shutil.copyfile('reIDmodel.py', os.path.join(output_directory, 'reIDmodel.py')) # copy file to output folder
    shutil.copyfile('networks.py', os.path.join(output_directory, 'networks.py')) # copy file to output folder
#如果是继续训练，则不需要拷贝
else:
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", opts.name))
    output_directory = os.path.join(opts.output_path + "/outputs", opts.name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Start training
#开始训练模型
# 如果是继续训练，其实也可以理解为使用预训练模型把，则先加载模型
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
config['epoch_iteration'] = round( train_loader_a.dataset.img_num  / config['batch_size'] )
print('Every epoch need %d iterations'%config['epoch_iteration'])
nepoch = 0 
    
print('Note that dataloader may hang with too much nworkers.')

#如果使用多个GPU训练
if num_gpu>1:
    print('Now you are using %d gpus.'%num_gpu)
    trainer.dis_a = torch.nn.DataParallel(trainer.dis_a, gpu_ids)
    trainer.dis_b = trainer.dis_a
    trainer = torch.nn.DataParallel(trainer, gpu_ids)

#训练
while True:
    #循环获取训练数据a,b
    #train_loader分为image，label,pos；注意，这里的image和pos是同一ID的不同图片，label表示ID
    for it, ((images_a,labels_a, pos_a),  (images_b, labels_b, pos_b)) in enumerate(zip(train_loader_a, train_loader_b)):
        if num_gpu>1:
            trainer.module.update_learning_rate()
        else:
            # 进行学习率更新
            trainer.update_learning_rate()

        #images_a[batch_size,3,256，128],images_b[batch_size,3,256，128]
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        #pos_a[batch_size,3,256，128],pos_b[batch_size,3,1024]
        pos_a, pos_b = pos_a.cuda().detach(), pos_b.cuda().detach()
        # labels_a[batch_size],labels_b[batch_size]
        labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            #进行前向传播
            # Main training code
            # 进行前向传播
            # 混合合成图片：x_ab[images_a的st，images_b的ap]    混合合成图片x_ba[images_b的st，images_a的ap]
            # s_a[输入图片images_a经过Es编码得到的 st code]     s_b[输入图片images_b经过Es编码得到的 st code]
            # f_a[输入图片images_a经过Ea编码得到的 ap code]     f_b[输入图片images_b经过Ea编码得到的 ap code]
            # p_a[输入图片images_a经过Ea编码进行身份ID的预测]    p_b[输入图片images_b经过Ea编码进行身份ID的预测]
            # pp_a[输入图片pos_a经过Ea编码进行身份ID的预测]      pp_b[输入图片pos_b经过Ea编码进行身份ID的预测]
            # x_a_recon[输入图片images_a（s_a）与自己（f_a）合成的图片，当然和images_a长得一样]
            # x_b_recon[输入图片images_b（s_b）与自己（f_b）合成的图片，当然和images_b长得一样]
            # x_a_recon_p[输入图片images_a（s_a）与图片pos_a（fp_a）合成的图片，当然和images_a长得一样]
            # x_b_recon_p[输入图片images_a（s_a）与图片pos_b（fp_b）合成的图片，当然和images_b长得一样]
            x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = \
                                                                          trainer.forward(images_a, images_b, pos_a, pos_b)
            #进行反向传播
            if num_gpu>1:
                trainer.module.dis_update(x_ab.clone(), x_ba.clone(), images_a, images_b, config, num_gpu)
                trainer.module.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, pos_a, pos_b, labels_a, labels_b, config, iterations, num_gpu)
            else:
                #计算判别器D的损失，然后进行反向传播
                trainer.dis_update(x_ab.clone(), x_ba.clone(), images_a, images_b, config, num_gpu=1)
                #计算Ea，Es，G的损失，然后进行方向传播
                trainer.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, pos_a, pos_b, labels_a, labels_b, config, iterations, num_gpu=1)

            #使用torch.cuda.synchronize()可以等待当前设备上所有流中的所有内核完成。
            torch.cuda.synchronize()

        # Dump training stats in log file
        #打印log
        if (iterations + 1) % config['log_iter'] == 0:
            print("\033[1m Epoch: %02d Iteration: %08d/%08d \033[0m" % (nepoch, iterations + 1, max_iter), end=" ")
            if num_gpu==1:
                write_loss(iterations, trainer, train_writer)
            else:
                write_loss(iterations, trainer.module, train_writer)

        # Write images
        #达到迭代次数进行图像保存
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                if num_gpu>1:
                    test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                else:
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            del test_image_outputs

        #每image_display_iter次，进行图像保存
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                if num_gpu>1:
                    image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
                else:
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            del image_outputs
        # Save network weights
        #模型保存
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            if num_gpu>1:
                trainer.module.save(checkpoint_directory, iterations)
            else:
                trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

    # Save network weights by epoch number
    #每训练完epoch次，保存模型
    nepoch = nepoch+1
    if(nepoch + 1) % 10 == 0:
        if num_gpu>1:
            trainer.module.save(checkpoint_directory, iterations)
        else:
            trainer.save(checkpoint_directory, iterations)

