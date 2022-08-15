# part of this code borrows from https://github.com/layumi/Person_reID_baseline_pytorch && https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/head/metrics.py
from audioop import bias
from operator import ne
from select import select
from tkinter.tix import Tree
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


class ft_net(nn.Module):
    def __init__(self, feature_dim):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.backbone = nn.Sequential(
                    model_ft.conv1,
                    model_ft.bn1,
                    model_ft.relu,
                    model_ft.maxpool,
                    model_ft.layer1,
                    model_ft.layer2,
                    model_ft.layer3,
                    model_ft.layer4,
                    model_ft.avgpool
                )
        self.features = nn.Linear(2048, feature_dim)
        # if am:
        #     self.classifier = FullyConnected_AM(feature_dim, num_classes, num_process, model_parallel, class_split)
        # else:
        #     self.classifier = FullyConnected(feature_dim, num_classes, num_process, model_parallel, class_split)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), x.size(1))
        x = self.features(x)
        # x = self.classifier(x, labels)
        return x

import ray

@ray.remote
class FCActors(object):
    def __init__(self) -> None:
        self.linear = None
        self.input = None
        self.output = None
        self.metric = nn.MSELoss()

    def buildParams(self,indim, outdim,bias):
        self.linear = nn.Linear(indim, outdim, bias)

    def forward(self, x):
        self.input = x
        self.input.retain_grad()
        output = self.linear(self.input)
        self.output = output
        return output

    def backward(self,label):
        loss = self.metric(self.output,label)
        loss.backward()
        print(self.input.grad.size())
        return self.input



# class FullyConnectedParallelFunc(Function):

#     @staticmethod
#     def forward(ctx, input, rayworkers):
#         ctx.save(input, rayworkers)
#         x_list = ray.get([rayworker.forward.remote(input) for rayworker in rayworkers])
#         return x_list

#     @staticmethod
#     def backward(ctx, grad_output):

#         pass


class FullyConnected(object):
    def __init__(self, in_dim, num_process=1, class_split=None):
        super(FullyConnected, self).__init__()
        self.num_process = num_process
        self.fcworkers = [FCActors.remote() for _ in range(num_process)]
        self.output = None

        ## build FC modules
        ray.get([
            worker.buildParams.remote(in_dim,class_split[i],bias) for i,worker in enumerate(self.fcworkers)
        ])


    def __call__(self, x):
        x_list = ray.get([worker.forward.remote(x) for worker in self.fcworkers])
        self.output = tuple(x_list)
        # print(type(self.output[0]))
        return tuple(x_list)

    # simulate parallel backward 
    def backward(self, label):
        grad_outputs = ray.get([
            worker.backward.remote(label[i]) for i,worker in enumerate(self.fcworkers)
        ])
        res = torch.zeros_like(grad_outputs[0])
        for grad_output in grad_outputs:
            res+=grad_output
        return res   


class FullyConnected_AM(nn.Module):
    def __init__(self, in_dim, out_dim, num_process=1, model_parallel=False, class_split=None, margin=0.35, scale=30):
        super(FullyConnected_AM, self).__init__()
        self.num_process = num_process
        self.model_parallel = model_parallel
        if self.model_parallel:
            self.am_branches = nn.ModuleList()
            for i in range(num_process):
                self.am_branches.append(AM_Branch(in_dim, class_split[i], margin, scale))
        else:
            self.am = AM_Branch(in_dim, out_dim, margin, scale)

    def forward(self, x, labels=None):
        if self.model_parallel:
            output_list = []
            for i in range(self.num_process):
                output = self.am_branches[i](x, labels[i])
                output_list.append(output)
            return tuple(output_list)
        else:
            return self.am(x, labels)


class AM_Branch(nn.Module):
    def __init__(self, in_dim, out_dim, margin=0.35, scale=30):
        super(AM_Branch, self).__init__()
        self.m = margin
        self.s = scale
        #  training parameter
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, label):
        x_norm = x.pow(2).sum(1).pow(0.5)
        w_norm = self.weight.pow(2).sum(0).pow(0.5)
        cos_theta = torch.mm(x, self.weight) / x_norm.view(-1, 1) / w_norm.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)
        phi = cos_theta - self.m

        index = label.data
        index = index.byte()

        output = cos_theta * 1.0
        output[index] = phi[index]
        output *= self.s

        return output


if __name__ == '__main__':
    net = ft_net(256)
    classifier = FullyConnected(256, 4,[16384] * 4)
    # print(net)
    input = torch.FloatTensor(8, 3, 256, 128)
    feature_x = net(input)
    output = classifier(feature_x)

    # print(output)
    print('net output size:')
    if isinstance(output, tuple):
        for o in output:
            print(o.shape)
    else:
        print(output.shape)

    grad_output = classifier.backward([torch.ones((8,16384))] * 4)
    print(grad_output.shape)
    feature_x.backward(grad_output)


