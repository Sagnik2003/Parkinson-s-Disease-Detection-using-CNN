import torch
from .dataloader import dataset,transform
from .model import Model, device
from PIL import Image

def predict_image(image_path, model_path,test_model=Model.to(device)):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    test_model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    with torch.no_grad():
        outputs = test_model(image)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).long().item()
    class_names = dataset.classes
    return class_names[predicted]

