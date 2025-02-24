import torch 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, accuracy_score, f1_score


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

        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
            ],
            weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=2
                                                                    )
        self.emotion_loss = torch.nn.CrossEntropyLoss(
            label_smoothing=0.05
        )
        self.sentiment_loss = torch.nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

    def train(self, epochs=10):
        self.model.train()
        training_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}

        for batch in self.train_loader:
            # MOve batch to GPU
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

            training_loss['total'] += total_loss.item()
            training_loss['emotion'] += emotion_loss.item()
            training_loss['sentiment'] += sentiment_loss.item()

        return {k: v / len(self.train_loader) for k, v in training_loss.items()}
    
    def validate(self):
        self.model.eval()
        val_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
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

                val_loss['total'] += total_loss.item()
                val_loss['emotion'] += emotion_loss.item()
                val_loss['sentiment'] += sentiment_loss.item()

                all_emotion_preds.extend(outputs["emotions"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(outputs["sentiments"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

        avg_loss = {k: v / len(self.val_loader) for k, v in val_loss.items()}

        self.scheduler.step(avg_loss['total'])

        # Calculate accuracy, precision and F1 score metrics
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='macro')
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='macro')

        return avg_loss, {
                        'emotion_precision': emotion_precision,
                        'emotion_accuracy': emotion_accuracy,
                        'emotion_f1': emotion_f1,
                        'sentiment_precision': sentiment_precision,
                        'sentiment_accuracy': sentiment_accuracy,
                        'sentiment_f1': sentiment_f1
                        }

def main():
    pass

if __name__ == '__main__':    
    main()


