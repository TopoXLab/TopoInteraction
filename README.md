# TopoInteraction
This repository contains the implementation for our work "[Learning Topological Interactions for Multi-Class Medical Image Segmentation](https://arxiv.org/pdf/2207.09654.pdf)", **accepted to ECCV 2022 (Oral)**. 

## Method
The loss function is for topological constraints: containment and exclusion. An overview of the method is as shown in the figure.
![Overview](pipeline-fig.png?raw=true)

## Installation
The loss can be applied to the training of any model. The code is written in PyTorch.

## Usage
    """
    Sample usage. In order to test the code, Input and GT are randomly populated with values.
    Set the dim (2 for 2D; 3 for 3D) correctly to run relevant code.

    The samples provided enforce the following interactions:
        Enforce class 1 to be completely surrounded by class 2
        Enforce class 2 to be excluded from class 3
        Enforce class 3 to be excluded from class 4
    """

    # Parameters for creating random input
    num_classes = height = width = depth = 5

    dim = 2

    if dim == 2:
        x = torch.rand(1,num_classes,height,width)
        y = torch.randint(0, num_classes, (1,1,height,width))

        ti_loss_weight = 1e-4
        ti_loss_func = TI_Loss(dim=2, connectivity=4, inclusion=[[1,2]], exclusion=[[2,3],[3,4]])
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)


    elif dim == 3:
        x = torch.rand(1,num_classes,depth,height,width)
        y = torch.randint(0, num_classes, (1,1,depth,height,width))

        ti_loss_weight = 1e-6
        ti_loss_func = TI_Loss(dim=3, connectivity=26, inclusion=[[1,2]], exclusion=[[2,3],[3,4]], min_thick=1)
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)

