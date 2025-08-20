import torch
from torch.utils.data import DataLoader
import argparse

from model import MyModel   # <-- adjust to your actual model class name
from pretrain import get_dataset  # <-- adjust if dataset loader is named differently

def check_dataset(args):
    # === Load dataset ===
    dataset = get_dataset(args)   # pretrain.py should have a function for this
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # === Build model to get n_classes ===
    model = MyModel(args)
    n_classes = model.output_dim if hasattr(model, "output_dim") else None
    if n_classes is None:
        raise ValueError("❌ Could not determine number of classes from model")

    print(f"✅ Loaded dataset with {len(dataset)} samples")
    print(f"✅ Model expects {n_classes} classes")

    # === Scan labels ===
    invalid_indices = []
    all_labels = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)

        # Track stats
        all_labels.append(labels)

        # Find invalids
        mask_invalid = (labels < 0) | (labels >= n_classes)
        if mask_invalid.any():
            bad = labels[mask_invalid]
            for i, val in zip(torch.nonzero(mask_invalid, as_tuple=True)[0], bad):
                dataset_idx = batch_idx * args.batch_size + i.item()
                invalid_indices.append((dataset_idx, val.item()))

    all_labels = torch.cat(all_labels)
    print("Label min:", all_labels.min().item(), "Label max:", all_labels.max().item())

    if invalid_indices:
        print("⚠️ Found invalid labels:")
        for idx, val in invalid_indices[:50]:
            print(f"  Sample {idx}: label={val}")
        if len(invalid_indices) > 50:
            print(f"... and {len(invalid_indices)-50} more.")
    else:
        print("✅ All labels are valid!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add the same args as pretrain.py (so dataset/model init works)
    parser.add_argument("--batch_size", type=int, default=32)
    # add other args here if needed (e.g., dataset path, vocab size, etc.)
    args = parser.parse_args()

    check_dataset(args)
