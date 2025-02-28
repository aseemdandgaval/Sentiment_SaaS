import os
import torch 
from collections import namedtuple
from datetime import datetime as time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import MultiModalFusion
from sklearn.metrics import precision_score, accuracy_score, f1_score
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class MultiModalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset size
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print(f"Train Dataset Size: {train_size:,}")
        print(f"Validation Dataset Size: {val_size:,}")
        print(f"Batches per Epoch: {len(train_loader):,}")

        # Tensorboard logging
        time_start = time.now().strftime('%b%d_%H-%M-%S') # e.g. Jan01_14-30-00
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{time_start}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.current_train_losses = None
        self.global_step = 0

        # Set up optimizer
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
            ],
            weight_decay=1e-5)
        
        # Learning rate scheduler and loss functions
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )
        self.emotion_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else: # Validation
            self.writer.add_scalar(f"loss/total/train", self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar(f"loss/total/val", losses['total'], self.global_step)
            self.writer.add_scalar(f"loss/sentiment/train",  self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar(f"loss/sentiment/val", losses['sentiment'], self.global_step)
            self.writer.add_scalar(f"loss/emotion/train",  self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar(f"loss/emotion/val", losses['emotion'], self.global_step)

        if metrics:
            self.writer.add_scalar(f"{phase}/emotion_accuracy/",  metrics['emotion_accuracy'], self.global_step)
            self.writer.add_scalar(f"{phase}/emotion_precision/",  metrics['emotion_precision'], self.global_step)
            self.writer.add_scalar(f"{phase}/emotion_f1/",  metrics['emotion_f1'], self.global_step)
            self.writer.add_scalar(f"{phase}/sentiment_accuracy/",  metrics['sentiment_accuracy'], self.global_step)
            self.writer.add_scalar(f"{phase}/sentiment_precision/",  metrics['sentiment_precision'], self.global_step)
            self.writer.add_scalar(f"{phase}/sentiment_f1/",  metrics['sentiment_f1'], self.global_step)

    def train(self, epochs=10):
        self.model.train()
        training_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}

        for batch in self.train_loader:
            # Move batch to GPU
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_label = batch['emotion_label'].to(device)
            sentiment_label = batch['sentiment_label'].to(device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Compute Loss
            emotion_loss = self.emotion_loss(outputs['emotions'], emotion_label)
            sentiment_loss = self.sentiment_loss(outputs['sentiments'], sentiment_label)
            total_loss = emotion_loss + sentiment_loss

            # Backward pass and gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track losses
            training_loss['total'] += total_loss.item()
            training_loss['emotion'] += emotion_loss.item()
            training_loss['sentiment'] += sentiment_loss.item()
            self.global_step += 1

            # Log metrics per training step
            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
                }, phase="train"
            )

        return {k: v / len(self.train_loader) for k, v in training_loss.items()}
    
    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to GPU
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Compute and track losses
                emotion_loss = self.emotion_loss(outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_loss(outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

                all_emotion_preds.extend(outputs["emotions"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(outputs["sentiments"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}

        if phase == "val":
            self.scheduler.step(avg_loss['total'])

        # Calculate accuracy, precision and F1 score metrics
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='macro')

        # Log metrics
        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'emotion_f1': emotion_f1,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy,
            'sentiment_f1': sentiment_f1
            }, phase=phase
        )

        return avg_loss, {
                        'emotion_precision': emotion_precision,
                        'emotion_accuracy': emotion_accuracy,
                        'emotion_f1': emotion_f1,
                        'sentiment_precision': sentiment_precision,
                        'sentiment_accuracy': sentiment_accuracy,
                        'sentiment_f1': sentiment_f1
                        }


def main():
    Batch = namedtuple('Batch', ['text_inputs', 'video_frames', 'audio_features'])
    mock_batch = Batch(text_inputs={'input_ids': torch.randint(0, 768, (1, 128)), 'attention_mask': torch.randint(0, 1, (1, 128))},
                       video_frames = torch.randn(1, 30, 3, 224, 224),
                       audio_features = torch.randn(1, 1, 64, 300)
                       )
    mock_loader = DataLoader([mock_batch])

    model = MultiModalFusion()
    trainer = MultiModalTrainer(model, mock_loader, mock_loader)

    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
        }
    val_losses = {
        'total': 1.5,
        'emotion': 0.5,
        'sentiment': 1.0
    }
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'emotion_f1': 0.8,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95,
        'sentiment_f1': 0.95
    }

    trainer.log_metrics(train_losses, phase="train")
    trainer.log_metrics(val_losses, val_metrics, phase="val")
    trainer.writer.close()

if __name__ == '__main__':    
    main()


