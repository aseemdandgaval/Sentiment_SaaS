import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import cv2
import numpy as np
import subprocess
import torchaudio
from torch.utils.data._utils.collate import default_collate

class MeldDataset(Dataset):
    def __init__(self, data_path, video_dir):
        self.data = pd.read_csv(data_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
        self.sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.emotion_map = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
        
    def __len__(self):
        return len(self.data)
    
    def get_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError("Error opening video file: {video_path}")
            
            # Try to read the first frame to validate the video
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Error reading video file: {video_path}")
            
            # Reset index to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Read frames until the end of the video 
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

            # Padd or truncate frames to 30 frames
            if len(frames) == 0:
                raise ValueError("No frames could be extracted}")
            elif len(frames) < 30:
                frames += [np.zeros_like(frames[0])] * (30 - len(frames))
            else:
                frames = frames[:30]
            
            # After permutating the frames, the shape is (30, 3, 224, 224)
            return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
                
        except Exception as e:
            raise ValueError(f"Error opening video file: {e}")
        finally:
            cap.release()

    def get_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run(['ffmpeg',
                            '-i', video_path,
                            '-vn',
                            '-acodec', 'pcm_s16le',
                            '-ar', '16000',
                            '-ac', '1',
                            audio_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spec= torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                                  n_fft=1024,                                            
                                                                  n_mels=64,
                                                                  hop_length=512)
            mel_spec = mel_spec(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction failed: {e}")
        except Exception as e:
            raise ValueError(f"Other audio error: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]

        try:
            # Locate video from index and Get Video Frames
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"No video found for filename: {video_path}")
            
            video_frames = self.get_video_frames(video_path)

            # Get Audio Features from video
            audio_features = self.get_audio_features(video_path)
            
            # Tokenize Text
            text_inputs = self.tokenizer(row['Utterance'],
                                         padding='max_length',
                                         truncation=True,
                                         max_length=128,
                                         return_tensors='pt')
            
            # Map Emotion and Sentiment to Labels
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
        
def collate_fn(batch):
    # Filter out None values
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloaders(train_csv, train_video_dir,
                       dev_csv, dev_video_dir,
                       test_csv, test_video_dir, batch_size=32):
    
    train_dataset = MeldDataset(train_csv, train_video_dir)
    dev_dataset = MeldDataset(dev_csv, dev_video_dir)
    test_dataset = MeldDataset(test_csv, test_video_dir)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader
    
if __name__ == '__main__':
    data_path = 'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/train/train_sent_emo.csv'
    video_dir = 'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/train/train_splits/'
    
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/train/train_sent_emo.csv',
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/train/train_splits/',
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/dev/dev_sent_emo.csv',
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/dev/dev_splits_complete/',
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/test/test_sent_emo.csv',
        'C:/Users/aseem/Downloads/Deep Learning/Sentiment_SaaS/dataset/test/output_repeated_splits_test/'
    )

    print(len(train_loader))
    print(len(dev_loader))
    print(len(test_loader))

    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break