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

## Backbone
The proposed method is a loss function and can be used with any network backbone. In the paper, we used UNet, FCN, and nnUNet backbones (both 2D and 3D for each). The backbone codes are publicly available and can be found as follows:
- [UNet](https://github.com/johschmidt42/PyTorch-2D-3D-UNet-Tutorial)
- [FCN](https://github.com/pochih/FCN-pytorch)
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)

## Datasets
In the paper, we evaluated our method on 4 datasets. The aorta dataset is a private dataset and is not publicly available. The remaining 3 datasets can be found as follows:
- [Multi-Atlas](https://www.synapse.org/#!Synapse:syn3193805/wiki/217753) and its [test-set GT](https://zenodo.org/record/1169361#.ZCr_HvbMJD8) : The GT was improved with consultation with clinicians so as to maintain the `Exclusion' property. 
- [SegTHOR](https://competitions.codalab.org/competitions/21145#learn_the_details-dataset) : As the test set is unavailable, we randomly created a split of 30 training volumes and 10 testing volumes from the given 40 training volumes.
- [IVUS](https://repository.ubn.ru.nl/bitstream/handle/2066/136858/1/136858.pdf) : We use Dataset B in this work. We use this dataset with permission from the authors.

## Citation
If you found this work useful, please consider citing it as
```
@inproceedings{gupta2022learning,
  title={Learning Topological Interactions for Multi-Class Medical Image Segmentation},
  author={Gupta, Saumya and Hu, Xiaoling and Kaan, James and Jin, Michael and Mpoy, Mutshipay and Chung, Katherine and Singh, Gagandeep and Saltz, Mary and Kurc, Tahsin and Saltz, Joel and others},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXIX},
  pages={701--718},
  year={2022},
  organization={Springer}
}
```
