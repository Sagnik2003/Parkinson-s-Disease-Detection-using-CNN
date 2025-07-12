from matplotlib import pyplot as plt
import torch
from .dataloader import val_loader
from .model import Model, device

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

model_path = 'E:/IMPORTANT_PROJECTS/Project_1/Perkinsons_Disease/FINAL/PD_torch_logs/best_model0.9677.pth'

test_model = Model.to(device)
test_model.load_state_dict(torch.load(model_path))
# Collect all labels and predictions from the validation set
all_labels = []
all_preds = []
test_model.eval()
with torch.no_grad():
	for inputs, labels in val_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)
		outputs = test_model(inputs)
		probs = torch.sigmoid(outputs)
		preds = (probs > 0.5).long().squeeze(1)
		all_labels.extend(labels.cpu().numpy())
		all_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()