import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

def accuracy(x, y):
    """ Accuracy.

    Args:
        x (torch.Tensor of float32): Predictions (logits), shape (B, C), B is
            batch size, C is num classes.
        y (torch.Tensor of int64): GT labels, shape (B, ),
            B = {b: b \in {0 .. C-1}}.

    Returns:
        Accuracy, in [0, 1].
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.mean(np.argmax(x, axis=1) == y)


def plot_interpolated_faces(*faces):
    fig = plt.figure(figsize=(15, 10))
    titles = ['Original Face 0: Alice',
             '0.75 Alice, 0.25 Bob',
             '0.5 Alice, 0.5 Bob',
             '0.25 Alice, 0.75 Bob',
             'Original Face 1: Bob']
    for ind, face in enumerate(faces):
        # Visualization
        ax = plt.subplot(1,len(faces),ind+1)
        plt.imshow(face.reshape(64,64),cmap='gray')
        ax.set_title(titles[ind])
    plt.show()
    plt.close(fig)

def plot_resnet_pred(x, gt_labels, pred_labels):
    fig = plt.figure(figsize=(15, 10))

    for ind, data in enumerate(x):
        # Visualization
        ax = plt.subplot(3,3,ind+1)
        plt.imshow(data[0,:,:],cmap='gray_r')
        ax.set_title("GT label {}, pred label {}".format(gt_labels[ind], pred_labels[ind]))
        if ind == 8:
            break
    plt.show()
    plt.close(fig)

    
    
class DummyBlock(nn.Module):
    """
    Dummy block, simply computes the identity function H(x) = x.
    
    You may use it instead of your ResBlock in the model.
    """
    
    def __init__(self, n_channels):
        # The following line initialize the parent class, you can safely ignore it.
        super().__init__()
        
        self.n_channels = n_channels

    def forward(self, x):
        # We verify if the memorized number of channels is correct
        assert x.shape[1] == self.n_channels, f"{x.shape[1]} != {self.n_channels}"
        
        # We return a copy of x to implement the identity function
        return x.clone()