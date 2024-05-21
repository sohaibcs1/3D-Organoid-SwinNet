import matplotlib.pyplot as plt
import torch

def visualize_prediction(val_ds, model_path, case_num=1, slice_num=20, device='cuda'):
    """
    Visualize the prediction of a SwinSegFormerEncoder model on a validation case.

    Parameters:
    - val_ds: Dataset object containing validation data
    - model_path: Path to the saved model file
    - case_num: Index of the case to visualize
    - slice_num: Slice number to visualize
    - device: Device to run the model on ('cuda' or 'cpu')
    """
    with torch.no_grad():
        # Get the image and label from the dataset
        img_name = os.path.split(val_ds[case_num]['image'].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]

        # Prepare the inputs and labels for the model
        val_inputs = torch.unsqueeze(img, 1).to(device)
        val_labels = torch.unsqueeze(label, 1).to(device)

        # Perform sliding window inference
        val_outputs = sliding_window_inference(
            val_inputs, (96, 96, 96), 1, model, overlap=0.8
        )

        # Create the plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num], cmap="gray")
        ax1.set_title('Image')
        ax2.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num])
        ax2.set_title(f'Label')
        ax3.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num])
        ax3.set_title(f'Predict')
        plt.show()

# Example usage
# visualize_prediction(val_ds, "/saved_model/best_metric_model.pth", case_num=1, slice_num=20, device='cuda')
