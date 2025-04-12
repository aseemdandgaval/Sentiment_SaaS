import torch
import inspect
import torch.nn as nn
from transformers import AutoModel
from torchvision import models as vision_models
from torchvision.models.video import R3D_18_Weights
from meld_dataset import MELDDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextEncoder(nn.Module):
    def __init__(self, model_name='answerdotai/ModernBERT-base'):
        super().__init__()
        self.model_name = model_name
        self.modern_bert = AutoModel.from_pretrained(model_name)
        self.mlp = nn.Linear(768, 128)
 
        for param in self.modern_bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        if self.model_name == 'answerdotai/ModernBERT-base':
            out = self.modern_bert(input_ids, attention_mask=attention_mask)
            out = out.last_hidden_state
            out = out.mean(dim=1)  # shape: [batch_size, hidden_size]
            out = self.mlp(out)
        else:
            out = self.modern_bert(input_ids, attention_mask=attention_mask)
            out = out.pooler_output
            out = self.mlp(out)

        return out


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(num_features, 128),
                                         nn.ReLU(),
                                         nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        x = self.backbone(x)
        return x
    

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
            )
        
    def forward(self, x):
        # [batch_size=32, channels=1, mel_freq=64, time_steps=300] -> [32, 64, 300]
        x = x.squeeze(1)
        x = self.network(x)
        x = self.mlp(x.squeeze(-1))
        return x
    

class MultiModalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion Layer (Outputs of all modalities are concatenated)
        self.fusion = nn.Sequential(nn.Linear(128 * 3, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2)
                                    )
        
        # Emotion Classifier
        self.emotion_classifier = nn.Sequential(nn.Linear(256, 64),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(64, 7),
                                            # nn.Softmax(dim=1)
                                            )
        
        # Sentiment Classifier
        self.sentiment_classifier = nn.Sequential(nn.Linear(256, 64),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(64, 3),
                                            # nn.Softmax(dim=1)
                                            )
    
    def forward(self, text_inputs, video_frames, audio_features):
        text_out = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])
        video_out = self.video_encoder(video_frames)
        audio_out = self.audio_encoder(audio_features)

        fusion_out = torch.cat((text_out, video_out, audio_out), dim=1)
        fusion_out = self.fusion(fusion_out) # [batch_size, 128 * 3]

        emotion_out = self.emotion_classifier(fusion_out)
        sentiment_out = self.sentiment_classifier(fusion_out)

        return {
            'emotions': emotion_out,
            'sentiments': sentiment_out
        }
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        param_dict = {param_name: param for param_name, param in param_dict.items() if param.requires_grad}

        # Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # Embeddings and weights in matmul
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # 1D tensors like LayerNorms, biases
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        # Fuses the kernels used in the updation of parameters to make it faster
        # Check if the fused version of AdamW is available and if the device is CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        print(f"Using fused AdamW: {use_fused} \n")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
        


def main():
    dataset = MELDDataset(
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/train/train_sent_emo.csv',
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/train/train_splits/'
        )
    sample = dataset[0]

    model = MultiModalFusion()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    print("\nEmotion Predictions:")
    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    print("\nSentiment Predictions:")
    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")


    ## Check outputs of each model using dummy inputs
    # text_encoder = TextEncoder().to(device).eval()
    # video_encoder = VideoEncoder().to(device).eval()
    # audio_encoder = AudioEncoder().to(device).eval()
    # fusion_layer = MultiModalFusion().to(device).eval()

    # text_inputs  = {'input_ids': torch.randint(0, 768, (1, 128)).to(device), 'attention_mask': torch.randint(0, 1, (1, 128)).to(device)}
    # video_inputs = torch.randn(1, 30, 3, 224, 224).to(device)
    # audio_inputs = torch.randn(1, 1, 64, 300).to(device)
    # text_out = text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])                                        
    # video_out = video_encoder(video_inputs)
    # audio_out = audio_encoder(audio_inputs)
    # fusion_out = fusion_layer(text_inputs, video_inputs, audio_inputs)

    # print("Text Encoder Output Shape", text_out.shape)
    # print("Video Encoder Output Shape", video_out.shape)
    # print("Audio Encoder Output Shape", audio_out.shape)
    # print("Fusion Output Shape (Emotion)", fusion_out['emotions'].shape)
    # print("Fusion Output Shape (Sentiment)", fusion_out['sentiments'].shape)

    
if __name__ == '__main__':
    main()  
