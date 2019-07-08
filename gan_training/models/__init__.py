from gan_training.models import (
    resnet, resnet2, resnet3, resnet4,
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet3': resnet3.Generator,
    'resnet4': resnet4.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet3': resnet3.Discriminator,
    'resnet4': resnet4.Discriminator,
}
