from common import *
from operator import sub
from collections import OrderedDict

## block ##-------
class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        padding = kernel_size - 2 if isinstance(kernel_size, int) else tuple(map(sub, kernel_size, (2,2)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class SeScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SeScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3), reduction=16):
        super(ResBlock, self).__init__()
        assert(in_planes==out_planes)

        self.conv_bn1 = ConvBn2d(in_planes,  out_planes, kernel_size=kernel_size, stride=1)
        self.conv_bn2 = ConvBn2d(out_planes, out_planes, kernel_size=kernel_size, stride=1)
        self.scale    = SeScale(out_planes, reduction)

    def forward(self, x):
        z  = F.relu(self.conv_bn1(x),inplace=True)
        z  = self.conv_bn2(z)
        z  = self.scale(z)*z + x
        z  = F.relu(z,inplace=True)
        return z



## net ##-------

class SeResNet3(nn.Module):
    def __init__(self, in_shape=(1,40,101), num_classes=12, base_channels=16, kernel_size = (3,3)):
        super(SeResNet3, self).__init__()
        in_channels = in_shape[0]

        self.layer1a = ConvBn2d(in_channels, base_channels, kernel_size=kernel_size, stride=(1, 1))
        self.layer1b = ResBlock(base_channels, base_channels)

        self.layer2a = ConvBn2d(base_channels, base_channels * 2 , kernel_size=kernel_size, stride=(1, 1))
        self.layer2b = ResBlock(base_channels * 2, base_channels * 2)
        self.layer2c = ResBlock(base_channels * 2, base_channels * 2)

        self.layer3a = ConvBn2d(base_channels * 2, base_channels * 4, kernel_size=kernel_size, stride=(1, 1))
        self.layer3b = ResBlock(base_channels * 4, base_channels * 4)
        self.layer3c = ResBlock(base_channels * 4, base_channels * 4)

        self.layer4a = ConvBn2d(base_channels * 4, base_channels * 8, kernel_size=kernel_size, stride=(1, 1))
        self.layer4b = ResBlock(base_channels * 8, base_channels * 8)
        self.layer4c = ResBlock(base_channels * 8, base_channels * 8)

        self.layer5a = ConvBn2d(base_channels  *  8, base_channels * 16, kernel_size=kernel_size, stride=(1, 1))
        self.layer5b = nn.Linear(base_channels * 16, base_channels * 8)

        self.fc1 = nn.Linear(base_channels * 8, base_channels * 8)
        self.fc2 = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x):

        x = F.relu(self.layer1a(x),inplace=True)
        x = self.layer1b(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.1,training=self.training)
        x = F.relu(self.layer2a(x),inplace=True)
        x = self.layer2b(x)
        x = self.layer2c(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer3a(x),inplace=True)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = F.max_pool2d(x,kernel_size=(2,2),stride=(2,2))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer4a(x),inplace=True)
        x = self.layer4b(x)
        x = self.layer4c(x)

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.layer5a(x),inplace=True)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer5b(x))

        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.2,training=self.training)
        x = self.fc2(x)

        return x  #logits

## check ##############################################################################


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((2,4)) # 2,7 for 40,201; 2,4 for 40,101
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #print(x.size())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

## check ##############################################################################

def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model

def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
    return model

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=(1,3)).view(features.size(0), -1) # 1,6 for 40,201; 1,3 for 40,101
        out = self.classifier(out)
        return out


## check ##############################################################################


class Ensemble(nn.Module):
    def __init__(self, in_shape=(1,40,101), num_classes=12):
        super(Ensemble, self).__init__()

        num_classes_each = num_classes

        self.seResNet_16 = SeResNet3(in_shape=in_shape, base_channels=16, kernel_size=5, num_classes=num_classes_each)
        self.seResNet_32 = SeResNet3(in_shape=in_shape, base_channels=32, num_classes=num_classes_each)
        self.densenet121 = densenet121(num_classes=num_classes_each)
        self.densenet169 = densenet169(num_classes=num_classes_each)
        self.resnet34    = resnet34(num_classes=num_classes_each)
        self.resnet152   = resnet152(num_classes=num_classes_each)

        self.seResNet_16_mfcc = SeResNet3(in_shape=in_shape, base_channels=16, kernel_size=5, num_classes=num_classes_each)
        self.seResNet_32_mfcc = SeResNet3(in_shape=in_shape, base_channels=32, num_classes=num_classes_each)
        self.densenet161_mfcc = densenet161(num_classes=num_classes_each)
        self.densenet201_mfcc = densenet201(num_classes=num_classes_each)
        self.resnet50_mfcc    = resnet50(num_classes=num_classes_each)
        self.resnet101_mfcc   = resnet101(num_classes=num_classes_each)

        self.fc1 = nn.Linear(num_classes_each * 9 , 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def _softmax(self, x, dim):
        # https://arxiv.org/pdf/1511.05042.pdf Taylor softmax
        # Ok = (1 + Ok + 1/2 Ok**2) / (SUMi (1 + Oi + 1/2 Oi**2))
        x2 = torch.mul(x,x)
        num = 1 + x + 0.5 * x2
        den =  torch.sum(x + 1., dim=dim) + 0.5 * torch.sum(x2, dim = dim)
        taylor_softmax = torch.div(num, torch.unsqueeze(den, dim=1))
        return taylor_softmax

    def forward(self, x_mel, x_mfcc):
        #x1 = self.seResNet_16.forward(x_mel)
        #x2 = self.seResNet_32.forward(x_mel)
        x3 = self.densenet121.forward(x_mel)
        x4 = self.densenet169.forward(x_mel)
        x5 = self.resnet34.forward(x_mel)
        x6 = self.resnet152.forward(x_mel)

        #x7  = self.seResNet_16_mfcc.forward(x_mfcc)
        x8  = self.seResNet_32_mfcc.forward(x_mfcc)
        x9  = self.densenet161_mfcc.forward(x_mfcc)
        x10 = self.densenet201_mfcc.forward(x_mfcc)
        x11 = self.resnet50_mfcc.forward(x_mfcc)
        x12 = self.resnet101_mfcc.forward(x_mfcc)

        #x = torch.cat((x1,x2,x3,x4,x5,x6, x7,x8,x9,x10,x11,x12), 1) # batch_size, n * num_classes
        x = torch.cat((x3,x4,x5,x6, x8,x9,x10,x11,x12), 1) # batch_size, n * num_classes

        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.2,training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x,p=0.2,training=self.training)
        x = self.fc4(x)

        return x, x3,x4,x5,x6, x8,x9,x10,x11,x12

## check ##############################################################################

def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 32
    num_classes = 12
    height = 40
    width  = 101
    labels = torch.randn(batch_size,num_classes)
    inputs = torch.randn(batch_size,1,height,width)
    y = Variable(labels).cuda()
    x = Variable(inputs).cuda()


    net = SeResNet3(in_shape=(1,height,width), num_classes=num_classes)
    net.cuda()
    net.train()


    logits = net.forward(x)
    probs  = F.softmax(logits, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    #print(net)
    print('probs')
    print(probs)




########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print('sucess')