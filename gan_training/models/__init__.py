from gan_training.models import (
    resnet, resnet2, 
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
}
