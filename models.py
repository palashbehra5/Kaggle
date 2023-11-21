import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation
from transformers import Mask2FormerForUniversalSegmentation

import segmentation_models_pytorch as smp

class baseline(nn.Module):

    def __init__(self):
        super(baseline, self).__init__()

        self.pool1 = nn.MaxPool2d(2,1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.dropout = nn.Dropout(p = 0.1)
        
        self.conv1 = nn.Conv2d(3, 8, 3)     
        self.conv2 = nn.Conv2d(8, 16, 3)    
        self.conv3 = nn.Conv2d(16, 32, 3)    
        self.conv4 = nn.Conv2d(32, 64, 3)    
        self.conv5 = nn.Conv2d(64, 128, 3)   

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, dilation=1)    
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, dilation=1)    
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, dilation=1)    
        self.deconv4 = nn.ConvTranspose2d(16, 8, 3, dilation=1)     
        self.deconv5 = nn.ConvTranspose2d(8, 3, 3, dilation=1)     


    def forward(self, x):

        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.conv5(x)))

        # print(x.shape)

        x = self.dropout(F.relu(self.deconv1(x)))
        x = self.dropout(F.relu(self.deconv2(x)))
        x = self.dropout(F.relu(self.deconv3(x)))
        x = self.dropout(F.relu(self.deconv4(x)))

        x = F.relu(self.deconv5(x))

        return x
    
class baseline_v2(nn.Module):

    def __init__(self):
        super(baseline_v2, self).__init__()

        # Stream 1
        self.conv1 = nn.Conv2d(11, 4, 3)     
        self.conv2 = nn.Conv2d(4, 6, 3)    
        self.conv3 = nn.Conv2d(6, 10, 3)    
        self.conv4 = nn.Conv2d(10, 20, 3)      

        self.deconv1 = nn.ConvTranspose2d(20, 10, 3, dilation=1)    
        self.deconv2 = nn.ConvTranspose2d(10, 6, 3, dilation=1)    
        self.deconv3 = nn.ConvTranspose2d(6, 4, 3, dilation=1)    
        self.deconv4 = nn.ConvTranspose2d(4, 3, 3, dilation=1)   


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        return x
    
class segformer(nn.Module):

    def __init__(self):
        super(segformer, self).__init__()

        self.pretrained = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
        self.conv = nn.Conv2d(19, 3, 3, 1, 1)
                

    def forward(self, x):

        x = F.relu(self.pretrained(x).logits)
        x = F.relu(self.conv(x))

        return x
    
class segformer_v2(nn.Module):

    def __init__(self):
        super(segformer_v2, self).__init__()

        self.conv1 = nn.Conv2d(11, 3, 3, 1, 1)
        self.pretrained = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
        self.conv2 = nn.Conv2d(19, 3, 3, 1, 1)
                

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.pretrained(x).logits)
        x = F.relu(self.conv2(x))

        return x
    
class maskformer(nn.Module):

    def __init__(self):
        super(maskformer, self).__init__()

        self.pretrained = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
        self.conv = nn.Conv2d(100, 3, 3, 1, 1)
                

    def forward(self, x):

        x = F.relu(self.pretrained(x).masks_queries_logits)
        x = F.relu(self.conv(x))

        return x
    


# model = maskformer()
# model.to('cuda')
# X = torch.rand((1, 3, 512, 512)).to('cuda')
# Y = model(X)
# print(X.shape, Y.shape)


