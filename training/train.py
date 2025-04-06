import os
import sys
import json
import torch
import argparse
import torchaudio
from tqdm import tqdm

from models import MultiModalFusion
from trainer import MultiModalTrainer
from install_ffmpeg import install_ffmpeg
from meld_dataset import prepare_dataloaders

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Data Directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    
    return parser.parse_args()


def main():
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    # Load DataLoaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv = os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir = os.path.join(args.train_dir, 'train_splits'),
        dev_csv = os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir = os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv = os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir = os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size = args.batch_size
    )

    print(f"Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Training video directory: {os.path.join(args.train_dir, 'train_splits')}")

    # Load the model
    model = MultiModalFusion().to(device)
    trainer = MultiModalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": [],
    }

    for epoch in tqdm(range(args.epochs), desc=f'Epoch {epoch}/{args.epochs}'):
        train_losses = trainer.train()
        val_losses, val_metrics = trainer.evaluate(val_loader, phase="val")

        # Track metrics
        metrics_data["train_losses"].append(train_losses)
        metrics_data["val_losses"].append(val_losses)
        metrics_data["epochs"].append(epoch)
        
        # Display metrics in SageMaker Format
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_losses['total']},
                {"Name": "val:loss", "Value": val_losses['total']},
                {"Name": "val:emotion_precision", "Value": val_metrics['emotion_precision']},
                {"Name": "val:emotion_accuracy", "Value": val_metrics['emotion_accuracy']},
                {"Name": "val:emotion_f1", "Value": val_metrics['emotion_f1']},
                {"Name": "val:sentiment_precision", "Value": val_metrics['sentiment_precision']},
                {"Name": "val:sentiment_accuracy", "Value": val_metrics['sentiment_accuracy']},
                {"Name": "val:sentiment_f1", "Value": val_metrics['sentiment_f1']}
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Initial GPU memory used: {memory_used:.2f} GB")

        # Save model checkpoint
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))
            print("Model saved!")

    # After traning is complete, evaluate the model on the test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_losses"] = test_loss["total"]

    print(json.dumps({
            "metrics": [
                {"Name": "test:loss", "Value": test_loss['total']},
                {"Name": "test:emotion_precision", "Value": test_metrics['emotion_precision']},
                {"Name": "test:emotion_accuracy", "Value": test_metrics['emotion_accuracy']},
                {"Name": "test:emotion_f1", "Value": test_metrics['emotion_f1']},
                {"Name": "test:sentiment_precision", "Value": test_metrics['sentiment_precision']},
                {"Name": "test:sentiment_accuracy", "Value": test_metrics['sentiment_accuracy']},
                {"Name": "test:sentiment_f1", "Value": test_metrics['sentiment_f1']}
            ]
        }))

if __name__ == '__main__':
    main()

