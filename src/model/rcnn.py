from torch import nn
from torchvision.models import resnet50
from torchvision.ops.misc import FrozenBatchNorm2d

from ..domain import pooling

class RCNN(nn.Module):
    """
    An R-CNN inspired parking lot classifier.
    Pools ROIs directly from image and separately
    passes each pooled ROI through a CNN.
    """

    def __init__(self, roi_res=100, pooling_type: str = 'square'):
        super().__init__()
        
        # load pretrained resnet50 model
        self.backbone = resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=2)
        
        # # remove the final layers (avgpool and fc)
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # # add a new classification head
        # self.head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(2048 * pool_size * pool_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # )
        
        # self.pooling_type = pooling_type
        # self.pool_size = pool_size

        # freeze bottom layers
        layers_to_train = ['layer2', 'layer3', 'layer4']
        for name, parameter in self.backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad = False  

        # ROI pooling     
        self.pooling_type = pooling_type
        self.roi_res = roi_res
    
    def forward(self, image, rois):

        # pool roi from image
        warps = pooling.roi_pool(image, rois, self.roi_res, self.pooling_type)
        
        # pass warped images trouth classifier
        class_logits = self.backbone(warps)
        
        return class_logits