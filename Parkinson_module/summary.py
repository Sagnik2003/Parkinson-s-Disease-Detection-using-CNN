from torchsummary import summary
from .model import Model, device

# print(len(train_loader.dataset))
summary = summary(Model, input_size=(3, 224, 224), device=device.type)
print(summary)