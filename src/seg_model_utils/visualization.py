"""
See notebook 5 for example use of show_mri_sample()
"""
import glob
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage


def make_bg_transparent(im, bg_th=0.0, set_to_color=None):
    # create transparency alpha channel
    # convert image to RGBA
    if len(im.shape) == 3:
        alpha_c = (np.sum(im[:,:,:],axis=2) > bg_th).astype(im.dtype)
        c1,c2,c3 = cv2.split(im)
    else:
        alpha_c = (im[:,:] > bg_th).astype(im.dtype)
        c1,c2,c3 = im.copy(), im.copy(), im.copy()
    if set_to_color is not None:
        zeros = np.zeros_like(c1)
        if set_to_color == 'green':
            merged = np.stack([zeros,c2,zeros,alpha_c], axis=-1)
        elif set_to_color == 'red':
            merged = np.stack([c1,zeros,zeros,alpha_c], axis=-1)
        elif set_to_color == 'royalblue':
            merged = np.stack([c1,zeros,zeros,alpha_c], axis=-1)
        elif set_to_color == 'violet':
            merged = np.stack([c1,zeros,c3,alpha_c], axis=-1)
        elif set_to_color == 'yellow':
            merged = np.stack([c1,c2,zeros,alpha_c], axis=-1)
    else:
        merged = np.stack([c1,c2,c3,alpha_c], axis=-1)
    return merged

def to_3d_points(im, th=1e-6, downsample=5):
    xs,ys,ds = [],[],[]
    if len(im.shape) == 4:
        im3d = np.sum(im,axis=3)
    else:
        im3d = im
    depth,width,height = im3d.shape
    step_vol = downsample**3
    for x in range(0, width - downsample, downsample):
        for y in range(0, height - downsample, downsample):
            for d in range(0, depth - downsample, downsample):
                if (np.sum(im3d[d:d+downsample, x:x+downsample, y:y+downsample]) / step_vol) > th:
                    xs.append(x + (downsample//2))
                    ys.append(y + (downsample//2))
                    ds.append(d + (downsample//2))
    return np.array(xs), np.array(ys), np.array(ds)

def adjust_saturation(img, sat_scale=0.3):
    hsv_im = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    (h, s, v) = cv2.split(hsv_im)
    s = s*sat_scale
    s = np.clip(s,0,255)
    hsv_im = np.stack([h,s,v],axis=2).astype(np.uint8)
    return cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB) / 255.

def show_mri_sample(sample, pred_mask=None, pred_lbl=None, seg_downsample=None, save_fn=None):
    """ Plot sample in three projections """
    plt.close('all')
    
    alpha=0.5
    image_alpha=1.0
    
    ims = sample['image'].numpy()
    means = sample['mean'].numpy()
    stds = sample['std'].numpy()
    segs = sample['segmentation'].numpy() if 'segmentation' in sample else None
    
    # add batch dims if missing
    if ims.ndim == 4:
        ims = np.expand_dims(ims, 0)
        means = np.expand_dims(means, 0)
        stds = np.expand_dims(stds, 0)
        if segs is not None:
            segs = np.expand_dims(segs, 0)
    
    n_images = len(ims)
    n_root = int(np.ceil(np.sqrt(n_images)))
    n_cols = n_root * 2
    n_rows = n_root * 2
    # special case fix to get with correct with small bs
    if n_images == 2:
        n_rows = 2
    
    fig_scale = 2
    f = plt.figure(figsize=(fig_scale*n_cols,fig_scale*n_rows))
    
    # Read additional meta from batch
    brats_ids = [sample['BraTSID']] if n_images == 1 else sample['BraTSID']
    labels = None
    if 'label' in sample:
        labels = [sample['label']] if n_images == 1 else sample['label']
    
    def _subplot_index(index, row_off, col_off):
        startrow = (index * 2)//n_cols
        startcol = (index * 2)%n_cols
        return (2*startrow+row_off)*n_cols + (startcol + col_off) + 1
    
    for index in range(n_images):
        im = ims[index]
        seg = segs[index]
        seg = np.swapaxes(seg, 0,3)
        # upsample seg back to original size if it has been downsampled
        if seg_downsample is not None:
            seg = seg.repeat(seg_downsample, axis=0).repeat(seg_downsample, axis=1).repeat(seg_downsample, axis=2)
        
        # Normalize images for visualization
        im = np.swapaxes(im, 0,3) # swap depth and chan axes
        im = (im * stds[index]) + means[index]
    
        title = f'BraTSID: {brats_ids[index]}'
        if labels is not None:
            title += f', GT-MGMT:{labels[index]}'
        if pred_lbl is not None:
            title += f'\nPred-MGMT:{float(pred_lbl[index][0]):.3f}'
    
        d,x,y,c = im.shape
        
        coronal_ax = f.add_subplot(n_rows,n_cols, _subplot_index(index,0,0))
        coronal_ax.set_title(title + ' - coronal', fontsize=8)
        coronal_ax.imshow(make_bg_transparent(adjust_saturation(im[::-1,x//2,:,:])), alpha=image_alpha)
    
        sagittal_ax = f.add_subplot(n_rows,n_cols,_subplot_index(index,0,1))
        sagittal_ax.set_title(title + ' - sagittal', fontsize=8)
        sagittal_ax.get_yaxis().set_visible(False)
        sagittal_ax.imshow(make_bg_transparent(adjust_saturation(im[::-1,:,y//2,:])), alpha=image_alpha)
    
        axial_ax = f.add_subplot(n_rows,n_cols,_subplot_index(index,1,0))
        axial_ax.set_title(title + ' - axial', fontsize=8)
        axial_ax.imshow(make_bg_transparent(adjust_saturation(im[d//2,:,:,:])), alpha=image_alpha)
    
        proj_ax = f.add_subplot(n_rows, n_cols, _subplot_index(index,1,1), projection='3d')
        proj_ax.scatter(*to_3d_points(im), color='gray', alpha=0.015, s=5, depthshade=False)
        proj_ax.set_title(f'Green=GT-tumor, Red=Pred-tumor\n{title}', fontsize=6)
        proj_ax.set_xticks([])                               
        proj_ax.set_yticks([])                               
        proj_ax.set_zticks([])
    
        if seg is not None:
            for seg_chan, color in zip(range(seg.shape[3]),['green']):
                coronal_ax.imshow(make_bg_transparent(seg[::-1,x//2,:,seg_chan], set_to_color=color), alpha=alpha)
                sagittal_ax.imshow(make_bg_transparent(seg[::-1,:,y//2,seg_chan], set_to_color=color), alpha=alpha)
                axial_ax.imshow(make_bg_transparent(seg[d//2,:,:,seg_chan], set_to_color=color), alpha=alpha)
                proj_ax.scatter(*to_3d_points(seg[:,:,:,seg_chan]), color=color, s=5, alpha=0.05)
    
        if pred_mask is not None:
            pred = np.swapaxes(pred_mask[index].cpu().numpy(), 0,3)
            pred = np.clip(pred, 0, 1.)
            # upsample seg back to original size if it has been downsampled
            if seg_downsample is not None:
                pred = pred.repeat(seg_downsample, axis=0).repeat(seg_downsample, axis=1).repeat(seg_downsample, axis=2)
            for seg_chan, color in zip(range(pred.shape[3]),['red']):
                coronal_ax.imshow(make_bg_transparent(pred[::-1,x//2,:, seg_chan], set_to_color=color, bg_th=0.5), alpha=alpha)
                sagittal_ax.imshow(make_bg_transparent(pred[::-1,:,y//2, seg_chan], set_to_color=color, bg_th=0.5), alpha=alpha)
                axial_ax.imshow(make_bg_transparent(pred[d//2,:,:, seg_chan], set_to_color=color, bg_th=0.5), alpha=alpha)
                proj_ax.scatter(*to_3d_points(pred[:,:,:,seg_chan], th=0.5), color=color, s=5, alpha=0.05)
        
        # draw axial lines
        coronal_ax.plot([0,x-1],[d//2,d//2],'--',color='white', linewidth=1) # coronal horizontal
        coronal_ax.plot([x//2,x//2],[0,d-1],'--',color='white', linewidth=1) # coronal vertical
        sagittal_ax.plot([0,y-1],[d//2,d//2],'--',color='white', linewidth=1) # sagittal horizontal
        sagittal_ax.plot([y//2,y//2],[0,d-1],'--',color='white', linewidth=1) # sagittal vertical
        axial_ax.plot([0,y-1],[x//2,x//2],'--',color='white', linewidth=1) # axial horizontal
        axial_ax.plot([x//2,x//2],[0,y-1],'--',color='white', linewidth=1) # axial vertical
    
    plt.subplots_adjust(left=0.00,top=1.,right=1.,bottom=0.00, wspace=0.15, hspace=0.15)
    
    bbox = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    width, height = bbox.width*f.dpi, bbox.height*f.dpi
    width *= 1.05
    height *= 1.05
    #if n_images == 2:
    #    n_rows = 2
    
    for row in range(0, n_rows,2):
        if n_images == 2 and row > 0:
            break
        for col in range(0, n_cols,2):
            different_color = (row//2) % 2 == (col//2) % 2
            color = (1,1,1) if different_color else (0.8,0.8,0.8)
            
            f.patches.extend([
                plt.Rectangle(
                    (width * col / n_cols, height * (n_rows - row - 2) / n_rows), 
                    width / max(1,n_cols//2), 
                    height / max(1,n_rows//2),
                    fill=True, 
                    color=color,  
                    zorder=-1, # below axes
                    alpha=0.5,
                    transform=None, 
                    figure=f)
            ])
    
    if save_fn is not None:
        plt.savefig(save_fn, transparent=False)
    else:
        plt.show()