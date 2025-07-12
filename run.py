from Parkinson_module.prediction import Model, predict_image



predicted_class = predict_image(
    'Geometric_Augmentation/test_set/spiral/PatientSpiral/img (1).jpg',
    'PD_torch_logs/best_model0.9677.pth',
    test_model= Model)

print(f"Predicted class: {predicted_class}")