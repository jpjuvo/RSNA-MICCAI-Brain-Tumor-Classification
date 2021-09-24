import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
import torch.optim as optim
import time
import shutil
from sklearn.metrics import roc_curve, auc
from argparse import ArgumentParser, Namespace

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math
from functools import partial
from torch.utils.tensorboard import SummaryWriter

import torchio as tio
from tqdm.auto import tqdm
from seg_model_utils.torchio_transforms import *
from seg_model_utils.brats2021_dataset import BraTS2021
from seg_model_utils.augmentations3d import *
from seg_model_utils.visualization import *
from seg_model_utils.seg_model import UNet3D_v2

import json
import wandb

LOG_WANDB = False

def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)

def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)

class TensorboardWriter():

    def __init__(self, args):

        name_model = args['log_dir'] + args['model'] + "_" + args['dataset_name'] + "_" + datestr()
        self.writer = SummaryWriter(log_dir=args['log_dir'] + name_model, comment=name_model)

        make_dirs(args['save'])
        self.csv_train, self.csv_val = self.create_stats_files(args['save'])
        self.dataset_name = args['dataset_name']
        self.classes = args['classes']
        self.label_names = args['class_names']

        self.data = self.create_data_structure()

    def create_data_structure(self, ):
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['loss'] = 0.0
        data['val']['loss'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        data['train']['dsc'] = 0.0
        data['val']['dsc'] = 0.0
        data['train']['acc'] = 0.0
        data['val']['acc'] = 0.0
        data['train']['mse'] = 0.0
        data['val']['mse'] = 0.0
        return data

    def display_terminal(self, iter, epoch, mode='train', summary=False):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:
            info_print = "\nSummary {} Epoch {:2d}:  Loss:{:.4f} \t DSC:{:.4f}\n Acc:{:.4f} MSE:{:.4f}".format(mode, epoch,
                                                                                         self.data[mode]['loss'] /
                                                                                         self.data[mode]['count'],
                                                                                         self.data[mode]['dsc'] /
                                                                                         self.data[mode]['count'],
                                                                                         self.data[mode]['acc'] / self.data[mode]['count'],
                                                                                         self.data[mode]['mse'] / self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{} : {:.4f}".format(self.label_names[i],
                                                     self.data[mode][self.label_names[i]] / self.data[mode]['count'])

            print(info_print)
        else:

            info_print = "\nEpoch: {:.2f} Loss:{:.4f} \t DSC:{:.4f}\n Acc:{:.4f} MSE:{:.4f}".format(iter, self.data[mode]['loss'] /
                                                                            self.data[mode]['count'],
                                                                            self.data[mode]['dsc'] /
                                                                            self.data[mode]['count'],
                                                                            self.data[mode]['acc'] / self.data[mode]['count'],
                                                                            self.data[mode]['mse'] / self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{}:{:.4f}".format(self.label_names[i],
                                                   self.data[mode][self.label_names[i]] / self.data[mode]['count'])
            print(info_print)

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        return train_f, val_f

    def reset(self, mode):
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['acc'] = 0.0
        self.data[mode]['mse'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def update_scores(self, iter, loss, mse, channel_score, acc, mode, writer_step):
        """
        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param mse: mse loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param acc: classification accuracy
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        dice_coeff = np.mean(channel_score) * 100

        num_channels = len(channel_score)
        self.data[mode]['dsc'] += dice_coeff
        self.data[mode]['loss'] += loss
        self.data[mode]['acc'] += acc
        self.data[mode]['mse'] += mse
        self.data[mode]['count'] = iter + 1
        
        for i in range(num_channels):
            chan_i = i
            self.data[mode][self.label_names[i]] += channel_score[chan_i]
            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[chan_i], global_step=writer_step)

    def write_end_of_epoch(self, epoch):

        self.writer.add_scalars('DSC/', {'train': self.data['train']['dsc'] / self.data['train']['count'],
                                         'val': self.data['val']['dsc'] / self.data['val']['count'],
                                         }, epoch)
        self.writer.add_scalars('Loss/', {'train': self.data['train']['loss'] / self.data['train']['count'],
                                          'val': self.data['val']['loss'] / self.data['val']['count'],
                                          }, epoch)
        self.writer.add_scalars('Acc/', {'train': self.data['train']['acc'] / self.data['train']['count'],
                                          'val': self.data['val']['acc'] / self.data['val']['count'],
                                          }, epoch)
        self.writer.add_scalars('MSE/', {'train': self.data['train']['mse'] / self.data['train']['count'],
                                          'val': self.data['val']['mse'] / self.data['val']['count'],
                                          }, epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars(self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] / self.data['train']['count'],
                                     'val': self.data['val'][self.label_names[i]] / self.data['train']['count'],
                                     }, epoch)

        train_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f} Acc:{:.4f} MSE:{:.4f}'.format(epoch,
                                                                     self.data['train']['loss'] / self.data['train'][
                                                                         'count'],
                                                                     self.data['train']['dsc'] / self.data['train'][
                                                                         'count'],
                                                                     self.data['train']['acc'] / self.data['train'][
                                                                         'count'],
                                                                     self.data['train']['mse'] / self.data['train'][
                                                                         'count'])
        val_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f} Acc:{:.4f} MSE:{:.4f}'.format(epoch,
                                                                   self.data['val']['loss'] / self.data['val'][
                                                                       'count'],
                                                                   self.data['val']['dsc'] / self.data['val'][
                                                                       'count'],
                                                                   self.data['val']['acc'] / self.data['val'][
                                                                       'count'],
                                                                   self.data['val']['mse'] / self.data['val'][
                                                                         'count'])
        self.csv_train.write(train_csv_line + '\n')
        self.csv_val.write(val_csv_line + '\n')

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.classes = None
        self.skip_index_after = None
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input, target):
        """
        Expand to one hot added extra for consistency reasons
        """
        target = expand_as_one_hot(target.long(), self.classes)

        assert input.dim() == target.dim() == 5, "'input' and 'target' have different number of dims"

        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            print("Target {} after skip index {}".format(before_size, target.size()))

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # get probabilities from logits
        #input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        loss = (1. - torch.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        # average Dice score across all channels/classes
        return loss, per_channel_dice

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the lib tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
    
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

def compute_channel_fusion_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # reduce channel dimension to one
    inp,_ = torch.max(input, dim=1, keepdim=True)
    tgt,_ = torch.max(target, dim=1, keepdim=True)
    #print(f'compute_channel_fusion_dice inp.shape {inp.shape}, tgt.shape {tgt.shape}')
    
    inp = flatten(inp)
    tgt = flatten(tgt)
    tgt = tgt.float()

    # compute per channel Dice Coefficient
    intersect = (inp * tgt).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (inp * inp).sum(-1) + (tgt * tgt).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=1, skip_index_after=None, weight=None, sigmoid_normalization=True ):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)
    
class Segmentation_class_accuracy():
    
    def __init__(self, classes=2, class_axis=1):
        self.classes = classes
        self.class_axis = class_axis
        
    def __call__(self, input, target):
        #assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
        input_classes = (input>0.0).flatten().type(torch.cuda.LongTensor)
        target_classes = (target>0.0).flatten().type(torch.cuda.LongTensor)
        return torch.sum(input_classes == target_classes) / input_classes.size()[0]

class BraTS2021_Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        self.terminal_show_freq = 50
        self.start_epoch = 1
        self.acc = Segmentation_class_accuracy()
        self.mse_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss_weight = 2.0
        self.save_dir = self.args['save']

    def training(self):
        for epoch in range(self.start_epoch, self.args['nEpochs']):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            if self.args['save'] is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(self.args['save'],
                                           epoch, val_loss,
                                           optimizer=self.optimizer)

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_samples in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()

            input_tensor, target_seg, target_clf = input_samples['image'], input_samples['segmentation'], input_samples['label']
            input_tensor, target_seg = input_tensor.to(self.device), target_seg.to(self.device)
            target_clf = target_clf.to(self.device)
            
            input_tensor.requires_grad = True
            
            output_seg, output_clf = self.model(input_tensor)

            loss_dice, per_ch_score = self.criterion(output_seg, target_seg)
            loss_mse = self.mse_loss(output_clf, target_clf.unsqueeze(1).type(torch.cuda.FloatTensor))
            
            loss_combined = loss_dice + loss_mse * self.mse_loss_weight
            loss_combined.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            
            with torch.no_grad():
                acc = self.acc(output_clf, target_clf)
            
            try:
                self.writer.update_scores(batch_idx, loss_dice.item(), loss_mse.item(), per_ch_score, acc.item(), 'train',
                                      epoch * self.len_epoch + batch_idx)
            except Exception as e:
                print(e)
                
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_samples in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target_seg, target_clf = input_samples['image'], input_samples['segmentation'], input_samples['label']
                input_tensor, target_seg = input_tensor.to(self.device), target_seg.to(self.device)
                target_clf = target_clf.to(self.device)
                
                input_tensor.requires_grad = False

                output_seg, output_clf = self.model(input_tensor)
                
                loss, per_ch_score = self.criterion(output_seg, target_seg)
                loss_mse = self.mse_loss(output_clf, target_clf.unsqueeze(1).type(torch.cuda.FloatTensor))
                
                acc = self.acc(output_clf, target_clf)
                
                try:
                    self.writer.update_scores(batch_idx, loss.item(), loss_mse.item(), per_ch_score, acc.item(), 'val',
                                          epoch * self.len_epoch + batch_idx)
                except Exception as e:
                    print(e)
                # preview one batch
                if batch_idx == 0:
                    show_mri_sample(input_samples, pred_mask=output_seg, 
                    pred_lbl=torch.sigmoid(output_clf.detach().cpu()).numpy(),
                    save_fn=os.path.join(self.save_dir, f'vis_epoch_{epoch}.png'))
                    if LOG_WANDB:
                        wandb.log({'validation': wandb.Image(os.path.join(self.save_dir, f'vis_epoch_{epoch}.png'))})
        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

def inference_sample_tta(model, image, batch_size=1):
    """Inference 3d image with tta and average predictions. Image shape: 3xWxHxD"""
    model.eval()
    
    def _flip(im, index=0):
        if index == 0:
            return im
        elif index == 1:
            return torch.flip(im, [1])
        elif index == 2:
            return torch.flip(im, [2])
        elif index == 3:
            return torch.flip(im, [3])
        elif index == 4:
            return torch.flip(im, [1,2])
        elif index == 5:
            return torch.flip(im, [1,3])
        elif index == 6:
            return torch.flip(im, [1,2,3])
        elif index == 7:
            return torch.flip(im, [2,3])
        
    def _predict(batch):
        batch.requires_grad=False
        seg_batch_flipped, clf_batch = model(batch.cuda())
        seg_batch_flipped, clf_batch = seg_batch_flipped.detach().cpu(), clf_batch.detach().cpu()
        # logits to preds
        clf_batch = torch.sigmoid(clf_batch)
        return seg_batch_flipped, clf_batch
    
    batch = torch.stack([_flip(image.clone(), index) for index in range(4)], dim=0)
    seg_batch_flipped_list, clf_batch_list = [],[]
    
    with torch.no_grad():    
        for start in range(0, 4, batch_size):
            seg_batch_flipped, clf_batch = _predict(batch[start:start + batch_size])
            
            seg_batch_flipped_list = seg_batch_flipped_list + [seg for seg in seg_batch_flipped]
            clf_batch_list = clf_batch_list + [clf for clf in clf_batch]
    
    # flip masks back
    seg_batch = torch.stack([_flip(seg, index) for index, seg in enumerate(seg_batch_flipped_list)], dim=0)
    
    # average results
    seg = torch.mean(seg_batch, dim=0)
    clf = torch.mean(torch.stack(clf_batch_list, dim=0), dim=0)
    return seg, clf

def eval_val_fold(out_dir, val_ds, model):
    # create dirs
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    pred_dir = os.path.join(out_dir, 'oof_preds')
    if not os.path.exists(pred_dir): os.mkdir(pred_dir)
    vis_dir = os.path.join(out_dir, 'vis')
    if not os.path.exists(vis_dir): os.mkdir(vis_dir)
    
    preds = []
    gts = []
    
    for val_index in tqdm(range(len(val_ds))):
        sample = val_ds.__getitem__(val_index)
        bratsid = f'{int(sample["BraTSID"]):05d}'
        
        gts.append(float(sample['label']))
        
        seg, clf = inference_sample_tta(model, sample['image'])
        preds.append(float(clf.cpu().numpy()))
        
        # save oof preds
        seg_fn = os.path.join(pred_dir, f'{bratsid}_seg.npy')
        np.save(seg_fn, seg.cpu().numpy())
        pred_fn = os.path.join(pred_dir, f'{bratsid}_pred.npy')
        np.save(pred_fn, clf.cpu().numpy())
        
        vis_fn = os.path.join(vis_dir, f'{bratsid}.png')
        show_mri_sample(
            sample, 
            pred_mask=seg.unsqueeze(0), 
            pred_lbl=[clf.numpy()],
            save_fn=vis_fn
        )
        plt.close('all')
    
    auc_fn = os.path.join(out_dir, 'auc.png')
    fpr, tpr, _ = roc_curve(np.array(gts), np.array(preds))
    roc_auc = auc(fpr, tpr)
    
    acc = np.sum((np.array(gts) > 0.5) == (np.array(preds) > 0.5)) / len(gts)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=f'ROC curve (area = {roc_auc:.2f}), Acc. = {acc*100:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(auc_fn, transparent=False)

def main(fold:int, train_df:str, npy_dir:str, bs:int, epochs:int):

    # start logging
    global LOG_WANDB
    wandb_config_fn = None
    if os.path.exists('../wandb_params.json'): 
        wandb_config_fn = '../wandb_params.json'
    if os.path.exists('./wandb_params.json'): 
        wandb_config_fn = './wandb_params.json'
    if wandb_config_fn is not None:
        with open(wandb_config_fn) as f:
            config = json.load(f)
        wandb.init(**config, 
            tags=['brain-segmentation', f'fold-{fold}'], 
            config={'bs':bs, 'epochs':epochs, 'fold':fold},
            sync_tensorboard=True)
        LOG_WANDB = True

    df = pd.read_csv(train_df)
    df_train = df[df.fold != fold]
    df_val = df[df.fold == fold]

    if len(df_val) == 0:
        df_val = df[df.fold == 0]

    sample_fns_train = [os.path.join(npy_dir, str(_id).zfill(5) + '.npy') for _id in df_train.BraTS21ID.values]
    lbls_train = list(df_train.MGMT_value.values)

    sample_fns_val = [os.path.join(npy_dir, str(_id).zfill(5) + '.npy') for _id in df_val.BraTS21ID.values]
    lbls_val = list(df_val.MGMT_value.values)

    crop_sz = (128,128,64)
    max_out_size=crop_sz

    tio_augmentations = tio.Compose([
        tio.RandomAffine(p=0.5),
        tio.RandomBiasField(p=0.3),
        tio.RandomGhosting(p=0.05),
        tio.RandomElasticDeformation(p=0.2),
        tio.RandomSpike(p=0.05),
        tio.RandomNoise(p=0.1),
        tio.RandomAnisotropy(p=0.05),
        tio.RandomFlip(p=0.5),
        #tio.RandomSwap(p=0.05),
        tio.RandomBlur(p=0.1),
        tio.RandomGamma(p=0.15),
    ])

    augmentations = ComposeTransforms([
        RandomCropToSize(crop_sz=crop_sz),
    ], p=1.0)

    train_ds = BraTS2021(
        mode='train', 
        npy_fns_list=sample_fns_train, 
        label_list=lbls_train,
        augmentations=augmentations,
        tio_augmentations=tio_augmentations,
        volume_normalize=True,
        max_out_size=max_out_size
    )

    val_augmentations = ComposeTransforms([
        RandomCropToSize(crop_sz=crop_sz),
    ], p=1.0)

    val_ds = BraTS2021(
        mode='val', 
        npy_fns_list=sample_fns_val, 
        label_list=lbls_val,
        augmentations=val_augmentations,
        volume_normalize=True,
        max_out_size=max_out_size
    )

    dl_args = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': 8,
    }

    train_generator = DataLoader(train_ds, **dl_args)
    val_generator = DataLoader(val_ds, **dl_args)

    # Load model
    model_unet3d_v2 = UNet3D_v2(out_channels=1)
    empty_state = model_unet3d_v2.state_dict()

    model2d = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)

    trained_2d_state = model2d.state_dict()

    for key_3d in empty_state.keys():
        if key_3d not in trained_2d_state.keys(): 
            print(f'skip {key_3d}')
            continue
        weight_3d, weight_2d = empty_state[key_3d], trained_2d_state[key_3d]
        
        # if shapes are same, regular copy
        if weight_3d.shape == weight_2d.shape:
            empty_state[key_3d] = trained_2d_state[key_3d].clone()
        # don't copy final layer
        elif key_3d != 'conv.weight' and key_3d != 'conv.bias': 
            weight = trained_2d_state[key_3d].clone()
            empty_state[key_3d] = weight.unsqueeze(2).repeat(1, 1, weight.size(2), 1, 1)
        
    model_unet3d_v2.load_state_dict(empty_state)
    model = model_unet3d_v2.cuda()

    if LOG_WANDB:
        wandb.watch(model, log_freq=100)

    optimizer_name = "adam" #'sgd'
    lr= 3e-3
    weight_decay = 0.0000000001

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6500, eta_min=1e-6, verbose=False)

    loss = DiceLoss(classes=1)

    train_args = {
        'nEpochs' : epochs, # 100
        'classes' : 1,
        'class_names' : ['tumor'],
        'inChannels' : 3,
        'log_dir' : './runs/',
        'save': f'./output/seg_model_256-{fold}',
        'model':'3DUnet1chan',
        'dataset_name':'registeredv1'
    }

    if 1:
        trainer = BraTS2021_Trainer(train_args, model, loss, optimizer, train_data_loader=train_generator,
                                    valid_data_loader=val_generator, lr_scheduler=scheduler)
        trainer.training()

    
    if fold == -1: return
    
    # eval

    # reload model
    model = UNet3D_v2(out_channels=1).cuda()
    model.load_state_dict(torch.load(f'./output/seg_model_256-{fold}/seg_model_256-{fold}_last_epoch.pth')['model_state_dict'])
    _ = model.eval()
    
    # load validation set
    val_ds = BraTS2021(
        mode='val', 
        npy_fns_list=sample_fns_val, 
        label_list=lbls_val,
        augmentations=None,
        volume_normalize=True,
        max_out_size=(256,256,96)
    )

    out_dir = os.path.join(f'./output/seg_model_256-{fold}', 'eval')
    eval_val_fold(out_dir, val_ds, model)

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    
    parser.add_argument('--fold', type=int)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_df', type=str, default='./input/train_labels_folds-v1.csv')
    parser.add_argument('--npy_dir', type=str, default='./input/registered_cases/train/')

    params = parser.parse_args()
    fold = params.fold
    train_df = params.train_df
    npy_dir = params.npy_dir
    bs = params.bs
    epochs = params.epochs
    
    main(fold, train_df, npy_dir, bs, epochs)