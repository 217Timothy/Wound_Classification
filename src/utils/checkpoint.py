import os
import torch
import torch.nn as nn
import shutil


def save_checkpoint(state, is_best, checkpoint_dir, filename='last.pt'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        shutil.copy(filepath, best_path)
        print(f"[CheckPoint] ✅ New best model saved! Accuracy: {state.get('val_acc', 0):.4f}. Recall: {state.get('val_recall', 0): .4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint