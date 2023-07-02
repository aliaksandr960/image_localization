import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm


class TimmMobilenet(nn.Module):
    def __init__(self, timm_name='mobilenetv3_small_100'):
        super(TimmMobilenet, self).__init__()
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
        self.source_model = timm.create_model(timm_name, pretrained=True)

    def forward(self, x):
        # Normalization
        x = x / 255
        x = self.normalize(x)

        # Featrue extraction
        x = self.source_model.forward_features(x)
        x = self.source_model.global_pool(x)
        x = self.source_model.conv_head(x)
        x = self.source_model.act2(x)
        
        # Norm 2 for facenet triplet loss
        x = torch.flatten(x, start_dim=1)
        x = F.normalize(x, p=2, dim=1)

        return x


class TimmVGG(nn.Module):
    def __init__(self, timm_name='vgg16'):
        super(TimmVGG, self).__init__()
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
        self.source_model = timm.create_model(timm_name, pretrained=True)

    def forward(self, x):
        # Normalization
        x = x / 255
        x = self.normalize(x)
        
        # Featrue extraction
        x = self.source_model.features(x)
        x = self.source_model.pre_logits(x)

        # Norm 2 for facenet triplet loss
        x = torch.flatten(x, start_dim=1)
        x = F.normalize(x, p=2, dim=1)
        return x