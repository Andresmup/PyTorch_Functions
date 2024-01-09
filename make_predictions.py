# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):

  '''
  Function to make predictions using a trained PyTorch neural network model on a given set of data.
  This function takes a list of data samples, performs inference using the model, and returns the predicted probabilities.
  Parameters:
    - model: PyTorch neural network model for making predictions.
    - data: List of data samples to be used for prediction.
    - device: PyTorch device where the model should perform inference (default is the device the model is currently on).
  Returns:
    - Tensor containing the predicted probabilities for each class for each input sample.
  Example:
    predictions = make_predictions(my_model, my_data)
  '''
  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      # Prepare sample
      sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

      # Forward pass (model outputs raw logit)
      pred_logit = model(sample)

      # Get prediction probability (logit -> prediction probability)
      pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

      # Get pred_prob off GPU for further calculations
      pred_probs.append(pred_prob.cpu())

  # Stack the pred_probs to turn list into a tensor
  return torch.stack(pred_probs)
