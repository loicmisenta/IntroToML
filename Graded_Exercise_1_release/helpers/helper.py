import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import axes3d, Axes3D

def visualize_blob_data(X, labels=None, C=None):
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    if C is not None:
        plt.scatter(C[:, 0], C[:, 1], color='C1', marker='x')
    plt.grid()


def visualize_cake_data(X, y):
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    # Creating plot
    ax.scatter3D(X[y==-1,0], X[y==-1,-1], X[y==-1,1], color = "r", label="leftover")
    ax.scatter3D(X[y==1,0], X[y==1,-1], X[y==1,1], color = "b", label="fresh")

    ax.legend()

    ax.set_xlabel('temperature')
    ax.set_ylabel('coffee')
    ax.set_zlabel('num students')

    # show plot
    plt.show()
    plt.close()


def visualize_cake_data_predictions(X, y, y_pred):
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    # Creating plot
    ax.scatter3D(X[y==y_pred,0], X[y==y_pred,-1], X[y==y_pred,1], color = "b", label="correct")
    ax.scatter3D(X[y!=y_pred,0], X[y!=y_pred,-1], X[y!=y_pred,1], color = "r", label="incorrect")

    ax.legend()

    ax.set_xlabel('temperature')
    ax.set_ylabel('coffee')
    ax.set_zlabel('num students')

    # show plot
    plt.show()
    plt.close()
    


def visualize_avg_distances(distances):
    plt.figure(figsize=(8, 5))
    x_axis = range(1, len(distances) + 1)
    plt.scatter(x_axis, distances)
    plt.plot(x_axis, distances)
    plt.xlabel("$K$ (number of clusters)", fontsize=15)
    plt.ylabel("Average Cluster Distance", fontsize=15)
    plt.grid()
    
