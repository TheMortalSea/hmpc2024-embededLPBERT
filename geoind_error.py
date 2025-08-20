# check_dataset.py
import torch
from torch.utils.data import DataLoader
import argparse

from dataset import TrainSet
from model import LPBERT
from pretrain import path_arr   # reuse same dataset paths

def check_dataset(args):
    # === Load dataset ===
    dataset = TrainSet(path_arr)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # === Build model to get n_classes ===
    model = LPBERT(
        args.layers_num,
        args.heads_num,
        args.embed_size,
        args.cityembed_size
    )
    n_classes = model.out_linear.out_features  # last layer output size

    print(f"✅ Loaded dataset with {len(dataset)} samples")
    print(f"✅ Model expects {n_classes} classes")

    # === Scan labels ===
    invalid_indices = []
    all_labels = []
    for batch_idx, batch in enumerate(loader):
        # batch format depends on TrainSet __getitem__, 
        # but labels usually come last (adjust if needed)
        *inputs, labels = batch
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)

        all_labels.append(labels)

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers_num", type=int, default=4)
    parser.add_argument("--heads_num", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--cityembed_size", type=int, default=64)
    args = parser.parse_args()

    check_dataset(args)
