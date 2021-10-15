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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torchio as tio
from tqdm.auto import tqdm
from clf_model_utils.miccai_2d_dataset import MICCAI2DDataset

import json
import wandb

import fastai
from fastai.vision.all import *
from fastai.data.core import DataLoaders
from fastai.callback.all import *
from fastai.callback.wandb import WandbCallback
import torch.nn.functional as F
from timm import create_model
from fastai.vision.learner import _update_first_layer
from fastai.vision.learner import _add_norm

LOG_WANDB = False

# This is modified from https://libauc.org/
class AUCMLoss(torch.nn.Module):
    """
    AUCM Loss with squared-hinge function: a novel loss function to directly optimize AUROC
    
    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
    outputs:
        loss value 
    
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None, device=None):
        super(AUCMLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p = imratio
        # https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) #cuda()
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) #.cuda()
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) #.cuda()
        
    def forward(self, input, target):

        y_pred = (torch.softmax(input, 1)[:,1]).unsqueeze(1)
        y_true = target.unsqueeze(1)

        if self.p is None:
           self.p = (y_true==1).float().sum()/y_true.shape[0]   
     
        y_pred = y_pred.reshape(-1, 1) # be carefull about these shapes
        y_true = y_true.reshape(-1, 1) 
        loss = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float())   + \
                    2*self.alpha*(self.p*(1-self.p)*self.margin + \
                    torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                    self.p*(1-self.p)*self.alpha**2
        return loss

def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)

def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)

def show_2d_batch(batch, preds=None, scale=4, save_fn=None):
    _images, _labels = batch
    images = _images.cpu().numpy()[:,0,:,:] # reduce rgb dimension to grayscale
    labels = _labels.cpu().numpy()
    
    cmap = matplotlib.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=np.percentile(images, 2), vmax=np.percentile(images, 98))
    
    if preds is not None:
        pred_lbls = list(preds.cpu().numpy())
    else:
        pred_lbls = [-1 for _ in labels]
        
    n_root = int(np.ceil(np.sqrt(len(images))))
    plt.close('all')
    f, axs = plt.subplots(n_root, n_root, figsize=((scale + 1)*n_root, scale*n_root))
    axs = axs.flatten()
    for img, lbl, pred, ax in zip(images, labels, pred_lbls, axs):
        axim = ax.imshow(img, cmap=cmap, norm=norm)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
        
        ax.set_title(f'GT: {lbl}, Pred: {pred:.3f}', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        
    # hide empties
    for ax_index in range(len(images), len(axs)):
        axs[ax_index].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(left = 0.1, right = 0.9, wspace=0.2, hspace=0.05)
    
    if save_fn is not None:
        plt.savefig(save_fn, transparent=False)
    else:
        plt.show()

class RocStarLoss(torch.nn.Module):
    """Smooth approximation for ROC AUC
    """
    def __init__(self, delta = 1.0, sample_size = 100, sample_size_gamma = 100, update_gamma_each=100):
        r"""
        Args:
            delta: Param from article
            sample_size (int): Number of examples to take for ROC AUC approximation
            sample_size_gamma (int): Number of examples to take for Gamma parameter approximation
            update_gamma_each (int): Number of steps after which to recompute gamma value.
        """
        super().__init__()
        self.delta = delta
        self.sample_size = sample_size
        self.sample_size_gamma = sample_size_gamma
        self.update_gamma_each = update_gamma_each
        self.steps = 0
        size = max(sample_size, sample_size_gamma)

        # Randomly init labels
        self.y_pred_history = torch.rand((size, 1)).cuda()
        self.y_true_history = torch.randint(2, (size, 1)).cuda()
        

    def forward(self, y_pred, target):
        """
        Args:
            y_pred: Tensor of model predictions in [0, 1] range. Shape (B x 1)
            y_true: Tensor of true labels in {0, 1}. Shape (B x 1)
        """
        y_pred_1 = (torch.softmax(y_pred, 1)[:,1]).unsqueeze(1)
        y_true = target.unsqueeze(1)
        
        if self.steps % self.update_gamma_each == 0:
            self.update_gamma()
        self.steps += 1
        
        positive = y_pred_1[y_true > 0]
        negative = y_pred_1[y_true < 1]
        
        # Take last `sample_size` elements from history
        y_pred_history = self.y_pred_history[- self.sample_size:]
        y_true_history = self.y_true_history[- self.sample_size:]
        
        positive_history = y_pred_history[y_true_history > 0]
        negative_history = y_pred_history[y_true_history < 1]
        
        if positive.size(0) > 0:
            diff = negative_history.view(1, -1) + self.gamma - positive.view(-1, 1)
            loss_positive = torch.nn.functional.relu(diff ** 2).mean()
        else:
            loss_positive = 0
 
        if negative.size(0) > 0:
            diff = negative.view(1, -1) + self.gamma - positive_history.view(-1, 1)
            loss_negative = torch.nn.functional.relu(diff ** 2).mean()
        else:
            loss_negative = 0
            
        loss = loss_negative + loss_positive
        
        # Update FIFO queue
        batch_size = y_pred_1.size(0)
        self.y_pred_history = torch.cat((self.y_pred_history[batch_size:], y_pred_1.clone().detach()))
        self.y_true_history = torch.cat((self.y_true_history[batch_size:], y_pred_1.clone().detach()))
        return loss

    def update_gamma(self):
        # Take last `sample_size_gamma` elements from history
        y_pred = self.y_pred_history[- self.sample_size_gamma:]
        y_true = self.y_true_history[- self.sample_size_gamma:]
        
        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]
        
        # Create matrix of size sample_size_gamma x sample_size_gamma
        diff = positive.view(-1, 1) - negative.view(1, -1)
        AUC = (diff > 0).type(torch.float).mean()
        num_wrong_ordered = (1 - AUC) * diff.flatten().size(0)
        
        # Adjuct gamma, so that among correct ordered samples `delta * num_wrong_ordered` were considered
        # ordered incorrectly with gamma added
        correct_ordered = diff[diff > 0].flatten().sort().values
        idx = min(int(num_wrong_ordered * self.delta), len(correct_ordered)-1)
        self.gamma = correct_ordered[idx]

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    save_fn = None
    if 'save_fn' in kwargs:
        save_fn = kwargs['save_fn']
    if save_fn is not None:
        plt.savefig(save_fn, transparent=False)
    else:
        plt.show()

# timm + fastai functions copied from https://walkwithfastai.com/vision.external.timm
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    if 'vit' in arch:
        model = create_model(arch, pretrained=pretrained, num_classes=0)
    else:
        model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

def create_timm_model(arch:str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model

def timm_learner(dls, arch:str, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, y_range=y_range, **config)
    kwargs.pop('ps')
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn

def main(fold:int, train_df_fn:str, npy_dir:str, bs:int, epochs:int, 
        lr:float=1e-4, arch:str='resnet34', ps:float=0.6, 
        optim:str='ranger', im_sz:int=256, loss_name:str="rocstar"):

    modality = str(os.path.dirname(npy_dir)).split('_')[-1]
    name = f'fold-{fold}'
    group_name = f'{modality}_{arch}_bs{bs}_ep{epochs}_{loss_name}_lr{lr}_ps{ps}_{optim}_sz{im_sz}'
    train_dir = npy_dir
    

    out_folder = os.path.join('./output', group_name, name)
    make_dirs(out_folder)

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
            name=name, group=group_name,
            tags=['MGMT-classification', f'fold-{fold}', modality], 
            config={
                'bs':bs, 'epochs':epochs, 'fold':fold,
                'ep':epochs, 'lr':lr, 'arch':arch, 'ps':ps, 
                'optim':optim, 'sz':im_sz, 'loss_name': loss_name,
                'modality' : modality
                },
            sync_tensorboard=True)
        LOG_WANDB = True

    df = pd.read_csv(train_df_fn)
    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]
    image_size = (im_sz,im_sz)

    if len(val_df) == 0:
        val_df = df[df.fold == 0]

    tio_augmentations = tio.Compose([
        tio.RandomAffine(p=0.5),
        tio.RandomBiasField(p=0.3),
        tio.RandomGhosting(p=0.05),
        tio.RandomElasticDeformation(p=0.2),
        tio.RandomSpike(p=0.05),
        tio.RandomNoise(p=0.1),
        tio.RandomAnisotropy(p=0.05),
        tio.RandomBlur(p=0.1),
        tio.RandomGamma(0.1, p=0.15),
    ])

    ds_t = MICCAI2DDataset(
        train_df, 
        npy_dir=npy_dir,
        image_size=image_size,
        tio_augmentations=tio_augmentations,
        is_train=True
    )

    ds_v = MICCAI2DDataset(
        val_df, 
        npy_dir=npy_dir,
        image_size=image_size,
        tio_augmentations=None,
        is_train=False
    )

    num_workers = 8
    dls = DataLoaders.from_dsets(ds_t, ds_v, bs=bs, device='cuda', num_workers=num_workers)

    loss = LabelSmoothingCrossEntropyFlat(eps=0.2)
    create_learner = cnn_learner

    if arch == 'densetnet121':
        base = densenet121
    elif arch == 'resnet18':
        base = resnet18
    elif arch == 'resnet34':
        base = resnet34
    elif arch == 'resnet50':
        base = resnet50
    elif arch == 'resnet101':
        base = resnet101
    elif arch == 'densenet169':
        base = densenet169
    else:
        create_learner = timm_learner
        base = arch
    
    if optim == "ranger":
        opt_func = fastai.optimizer.ranger
    else:
        opt_func = fastai.optimizer.Adam

    if loss_name == 'rocstar':
        second_loss = RocStarLoss()
    elif loss_name == 'bce':
        second_loss = loss
    elif loss_name == 'libauc':
        second_loss = AUCMLoss()
    else:
        raise Exception

    learn = create_learner(
            dls, 
            base,
            pretrained=True,
            n_out=2,
            loss_func=loss,
            opt_func=opt_func,
            metrics=[
                RocAucBinary(),
                accuracy
            ],
            ps=ps
        ).to_fp16()

    # train head first with CE
    learn.fit_one_cycle(1, lr)
    learn.unfreeze()

    model_path = os.path.join('..', out_folder, 'final')
    cbs = [WandbCallback(log=None, log_preds=False, log_model=False)] if LOG_WANDB else []
    
    #best_path = os.path.join('..', out_folder, 'best')
    #save_cb = SaveModelCallback(monitor='roc_auc_score', fname=best_path, reset_on_fit=True)
    #cbs.append(save_cb)

    # continue with main loss
    learn.loss_func = second_loss
    learn.fit_flat_cos(epochs, lr, div_final=2, pct_start=0.99, cbs=cbs)
    
    learn.save(model_path, with_opt=False)

    #plot_fn = os.path.join(out_folder, 'plot_metrics.png')
    #plt.close('all')
    #learn.recorder.plot_metrics()
    #plt.savefig(plot_fn)

    #if LOG_WANDB:
    #    wandb.log({'training': wandb.Image(plot_fn)})
        
    # eval
    if fold >= 0:
        dl_test = DataLoader(ds_v, 32, num_workers=8, shuffle=False)
        test_preds = learn.get_preds(dl=dl_test)

        test_p, test_gt = test_preds
        test_p = torch.softmax(test_p, 1)
        test_p = test_p.numpy()[:,1]
        test_gt = test_gt.numpy()

        tta_preds = learn.tta(dl=dl_test)
        tta_p, tta_gt = tta_preds
        tta_p = torch.softmax(tta_p, 1)
        tta_p = tta_p.numpy()[:,1]
        tta_gt = tta_gt.numpy()

        fpr, tpr, _ = roc_curve(np.array(test_gt), np.array(test_p))
        tta_fpr, tta_tpr, _ = roc_curve(np.array(tta_gt), np.array(tta_p))

        roc_auc = auc(fpr, tpr)
        tta_roc_auc = auc(tta_fpr, tta_tpr)

        acc = np.sum((np.array(test_gt) > 0.5) == (np.array(test_p) > 0.5)) / len(test_gt)
        tta_acc = np.sum((np.array(test_gt) > 0.5) == (np.array(test_p) > 0.5)) / len(test_gt)

        auc_fn = os.path.join(out_folder, 'auc.png')
        plt.close('all')
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label=f'ROC curve (area = {roc_auc:.2f}), Acc. = {acc*100:.2f}')
        plt.plot(tta_fpr, tta_tpr, color='red',
                lw=lw, label=f'TTA ROC curve (area = {tta_roc_auc:.2f}), Acc. = {tta_acc*100:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(auc_fn, transparent=False)
        
        if LOG_WANDB:
            wandb.log({'validation': wandb.Image(auc_fn)})
            wandb.log({'auc' : roc_auc})
            wandb.log({'auc-tta' : tta_roc_auc})
            wandb.log({'acc' : acc})
            wandb.log({'acc-tta' : tta_acc})

        result_df = val_df.copy()
        result_df['pred_mgmt'] = list(test_p)
        result_df['pred_mgmt_tta'] = list(tta_p)
        result_df.to_csv(os.path.join(out_folder, 'oof.csv'))

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    
    parser.add_argument('--fold', type=int)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_df', type=str, default='./input/train_feature_data_v2.csv')
    parser.add_argument('--npy_dir', type=str, default='./input/aligned_and_cropped_t2w/')
    parser.add_argument('--arch', type=str, default='resnet34')
    parser.add_argument('--ps', type=float, default=0.6)
    parser.add_argument('--optim', type=str, default='ranger')
    parser.add_argument('--im_sz', type=int, default=256)
    parser.add_argument('--loss_name', type=str, default='auclib')


    params = parser.parse_args()
    fold = params.fold
    train_df = params.train_df
    npy_dir = params.npy_dir
    bs = params.bs
    epochs = params.epochs
    lr = params.lr
    arch = params.arch
    ps = params.ps
    optim = params.optim
    im_sz = params.im_sz
    loss_name = params.loss_name
    
    main(fold, train_df, npy_dir, bs, epochs, lr, arch, ps, optim, im_sz, loss_name)