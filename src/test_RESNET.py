# import torchvision.models as models
# from torchvision.models import ResNet18_Weights

# model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# layers = [module for module in model.modules() if len(list(module.children())) == 0]
# print("Total leaf layers:", len(layers))

import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Load pretrained ResNet18
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Print high-level architecture
print(model)
