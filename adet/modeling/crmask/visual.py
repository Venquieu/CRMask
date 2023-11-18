import math
import torch
import matplotlib.pyplot as plt

def batch_visual(images:torch.tensor, rows = 1, title = None):
    '''
    images: Tensor with shape [N, 3, H, W] or [N, H, W] 
    '''
    if title == None:
        title = 'Batched images'
    if images.dim() == 4:
        images = images.permute(0, 2, 3, 1)
    images = images.cpu().detach()
    N = images.size(0) # num of inages
    cols = math.ceil(N/rows)
    
    for i in range(N):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i])
    plt.suptitle(title)
    plt.show()