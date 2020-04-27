import torch
import torch.nn as nn
import torch.nn.functional as F

from NoisyAnd_layer import NoisyAnd

class CAM_MIL(torch.nn.Module):

    #Shape for input x: (B, 2048, 7, 7)

    def __init__(self):
        super(CAM_MIL, self).__init__()

        self.num_classes = 2

        self.cam_conv = nn.Conv2d(2048, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            NoisyAnd()
        )

        self.cls_loss = nn.CrossEntropyLoss()
        self.mil_loss = self.MIL_loss

    def forward(self, x, label):

        out1 = self.cam_conv(x)
#         print("out1", out1.shape)
        #att_map = self.get_atten_map(out1, label)
#         print("att_map", att_map.shape)

        cls_logit = self.classifier(out1)
        cls_logit = cls_logit.view((cls_logit.shape[0], cls_logit.shape[1]))
        return [cls_logit, out1]

    def MIL_loss(self, activation_maps, labels):

        cam_shape = activation_maps.shape
        n_batch, n_class = cam_shape[0], cam_shape[1]

        flatten_maps = activation_maps.view(n_batch, n_class, -1) #B x K x (W*H)
        flatten_maps = torch.softmax(flatten_maps, dim=2) # B x K x (W*H)
        instance_logits = torch.max(flatten_maps, dim=2, keepdim=False)[0] # B x K

        mil_loss = -torch.mean(torch.sum(labels * torch.log(instance_logits), dim=1), dim=0) # scalar

        return mil_loss

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):


        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_loss(self, logits, labels):
        cls_logit, att_map = logits

        labels_onehot = F.one_hot(labels.long(), 2).view(-1, 2)
        n_batch = labels.shape[0]
#         if n_batch > 1:
        labels_idx = torch.squeeze(labels.long(), dim=-1)
#         else:
#             labels_idx = labels.long()
#             print("label_idx", labels_idx.shape)
#         if labels_idx.shape[0] == 1:
#             print("label shape", labels_idx.shape, cls_logit.shape)
        loss_cls = self.cls_loss(cls_logit, labels_idx)
        loss_mil = self.mil_loss(att_map, labels_onehot)

        loss_val = loss_cls + loss_mil
        return [loss_val, loss_cls, loss_mil]


