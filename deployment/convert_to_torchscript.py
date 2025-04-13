import torch
import torch.nn as nn
from typing import NamedTuple
from models import MultiModalFusion


# Define a NamedTuple to represent the output of the model.
class ModelOutput(NamedTuple):
    emotions: torch.Tensor
    sentiments: torch.Tensor

# Define a wrapper class to convert the model's output to a NamedTuple.
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, text_inputs, video_frames, audio_features):
        # Call the original forward (which returns a dict)
        output = self.model(text_inputs, video_frames, audio_features)
        # Convert the dict into a NamedTuple
        return ModelOutput(emotions=output['emotions'], sentiments=output['sentiments'])


# 1. Reconstruct the model.
model = MultiModalFusion()

# 2. Load the state dictionary.
device = torch.device("cpu")  # or use "cuda" if appropriate
model.load_state_dict(torch.load("deployment/model/best_model.pt", map_location=device,  weights_only=True))

# 3. Set model to evaluation mode.
model.eval()

# 4. Wrap the model for TorchScript.
wrapped_model = ModelWrapper(model)
wrapped_model.eval()

# 5. Create dummy inputs that match your model’s expected input shapes.
device = torch.device("cpu")  # or "cuda" if appropriate
text_inputs  = {
    'input_ids': torch.randint(0, 768, (1, 128), device=device),
    'attention_mask': torch.randint(0, 2, (1, 128), device=device)
}
video_frames = torch.randn(1, 30, 3, 224, 224, device=device)
audio_features = torch.randn(1, 1, 64, 300, device=device)

# 6. Trace the wrapped model.
traced_model = torch.jit.trace(wrapped_model, (text_inputs, video_frames, audio_features))
traced_model.save("deployment/model/best_model_traced.pt")
print("TorchScript model successfully traced and saved!")
