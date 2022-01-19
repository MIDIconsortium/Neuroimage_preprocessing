import os
import pandas as pd
import numpy as np
import nibabel as nib
import monai
from IPython.display import clear_output
from monai.data import CacheDataset, DataLoader, NiftiDataset
from monai.transforms import (
    AddChannel, 
    Resize, 
    ScaleIntensity, 
    ToTensor,
    Randomizable,
    LoadNifti,
    Spacing,
    ResizeWithPadOrCrop
)
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from monai.config import print_config


def preprocess(input_path, save_path, border=15, min_dim=130, plot=True):
    
    resize = Resize(spatial_size=(120, 120, 120), mode='trilinear')
    crop_pad = ResizeWithPadOrCrop(spatial_size=(180,180,180))
    
    arr, _ = LoadNifti()(input_path)
    shape = arr.shape
    dims = np.round(_['pixdim'][1:4],2)
    arr = AddChannel()(arr)
    arr_resampled =  Spacing(pixdim=(1., 1., 1.), mode='bilinear')(arr,_['affine'])[0]

    if arr_resampled.shape[-1] > min_dim and arr_resampled.shape[-2] > min_dim and arr_resampled.shape[-3] > min_dim:
        mid_slice = arr_resampled.squeeze()[:,:,int(arr_resampled.shape[-1]/2)]
        mask = mid_slice>0.5*mid_slice.std()
        a, b = np.argmax(mask, axis=0)[int(mask.shape[1]/2)], np.argmax(mask, axis=1)[int(mask.shape[0]/2)]
        a1, b1 = np.argmax(np.flipud(mask), axis=0)[int(mask.shape[1]/2)], np.argmax(np.fliplr(mask), axis=1)[int(mask.shape[0]/2)]


        if a > border:
            a -= border
        else:
            a = 0
        if b > border:
            b -= border
        else:
            b = 0
        if b1 > border:
            b1 -= border
        else:
            b1 = 1
        if a1 > border:
            a1 -= border
        else:
            a1 = 1

        cropped = crop_pad(arr_resampled[:,a:-a1, b:-b1,:])
        resized = resize(cropped)

        new_image = nib.Nifti1Image(resized, affine=np.eye(4))

        nib.save(new_image, save_path)
        
        if plot:
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8,4, figsize = (15,15))

            cropped = crop_pad(arr_resampled[:,a:-a1, b:-b1,:])
            resized = resize(cropped)

            pad = 180 - arr_resampled.shape[-1]

            title = [str(a) + 'mm' for a in dims]
            
            ax1[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]*0.5/8)]), cmap='gray')
            ax1[0].set_title('{}\n{}, {}, {}'.format(shape, title[0], title[1],title[2]))
            ax1[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]*0.5/8)]), cmap='gray')
            ax1[1].set_title('{}\n1mm, 1mm, 1mm'.format(arr_resampled.squeeze().shape, (1,1,1)))
            ax1[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]*0.5/8 + pad/2)]), cmap='gray')
            ax1[2].set_title('{},\n1mm, 1mm, 1mm'.format(cropped.squeeze().shape, (1,1,1)))
            ax1[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]*0.5/8 + pad/2))]), cmap='gray')
            ax1[3].set_title('(120,120,120)\n N/A')


            ax2[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]/8)]), cmap='gray')
            ax2[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]/8)]), cmap='gray')
            ax2[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]/8 + pad/2)]), cmap='gray')
            ax2[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]/8 + pad/2))]), cmap='gray')

            ax3[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]/4)]), cmap='gray')
            ax3[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]/4)]), cmap='gray')
            ax3[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]/4 + pad/2)]), cmap='gray')
            ax3[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]/4 + pad/2))]), cmap='gray')

            ax4[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]/2)]), cmap='gray')
            ax4[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]/2)]), cmap='gray')
            ax4[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]/2 + pad/2)]), cmap='gray')
            ax4[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]/2 + pad/2))]), cmap='gray')

            ax5[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]*3/4)]), cmap='gray')
            ax5[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]*3/4)]), cmap='gray')
            ax5[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]*3/4 + pad/2)]), cmap='gray')
            ax5[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]*3/4 + pad/2))]), cmap='gray')

            ax6[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]*7/8)]), cmap='gray')
            ax6[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]*7/8)]), cmap='gray')
            ax6[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]*7/8 + pad/2)]), cmap='gray')
            ax6[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]*7/8 + pad/2))]), cmap='gray')

            ax7[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1]*7.5/8)]), cmap='gray')
            ax7[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1]*7.5/8)]), cmap='gray')
            ax7[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1]*7.5/8 + pad/2)]), cmap='gray')
            ax7[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1]*7.5/8 + pad/2))]), cmap='gray')

            ax8[0].imshow(np.rot90(arr.squeeze()[:,::-1,int(arr.shape[-1] - 1)]), cmap='gray')
            ax8[1].imshow(np.rot90(arr_resampled.squeeze()[:,::-1,int(arr_resampled.shape[-1] - 1)]), cmap='gray')
            ax8[2].imshow(np.rot90(cropped.squeeze()[:,::-1,int(arr_resampled.shape[-1] - 1 + pad/2)]), cmap='gray')
            ax8[3].imshow(np.rot90(resized.squeeze()[:,::-1,int(120/180*(arr_resampled.shape[-1] - 1 + pad/2))]), cmap='gray')

            for Ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8):
                for ax in Ax:
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.savefig(save_path[:-3] + 'png')


