import torch
import torch.nn as nn

class NoisyAnd(nn.Module):
    def __init__(self, a=10, b=3.0):
        super(NoisyAnd, self).__init__()
        self.a = a
        self.b = b
    def forward(self, input): # input_dim = [8,2,7,7]
        sig = nn.Sigmoid()
        mean = torch.mean(input, dim=[2,3])
        t11 = sig(self.a * (mean - self.b))
        t12 = sig(torch.tensor(-1 * self.a * self.b))
        t21 = sig(torch.tensor(self.a * (1 - self.b)))
        t22 = sig(torch.tensor(-1 * self.a * self.b))
        ans = (t11 - t12)/(t21 - t22)
        ans = ans.reshape([input.shape[0], input.shape[1], 1, 1]) # output_dim = [8,2,1,1]
        return ans

'''
fcn_model.py:

self.classifier = nn.Sequential(
    nn.ReLU(inplace=True),
    nn.AvgPool2d(7) # TODO: Replace this with 'NoisyAnd()'
)
'''
