from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "gzt_mnist_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    """Lightweight CNN that performs well on MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.MNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=get_transforms()
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=get_transforms()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


@dataclass
class TrainResult:
    train_loss: float
    test_acc: float
    epochs: int
    model_path: str


def train_model(epochs: int = 5, lr: float = 1e-3, batch_size: int = 128) -> TrainResult:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - test_acc: {test_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    return TrainResult(train_loss=avg_loss, test_acc=test_acc, epochs=epochs, model_path=str(MODEL_PATH))


def evaluate(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def load_trained_model(model_path: Path = MODEL_PATH) -> nn.Module:
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def normalize_student_id_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    np_img = np.array(gray).astype(np.float32)
    if np.mean(np_img) > 127:
        np_img = 255 - np_img
    np_img = np.clip(np_img, 0, 255)
    return Image.fromarray(np_img.astype(np.uint8))


def _find_digit_bounds(np_img: np.ndarray, min_width: int = 6) -> List[tuple[int, int]]:
    col_sums = np.sum(np_img > 20, axis=0)
    max_val = np.max(col_sums)
    active = col_sums > (0.18 * max_val)
    bounds: List[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(active):
        if val and start is None:
            start = idx
        if not val and start is not None:
            end = idx
            if end - start >= min_width:
                bounds.append((start, end))
            start = None
    if start is not None:
        end = len(active)
        if end - start >= min_width:
            bounds.append((start, end))
    return bounds


def _crop_and_pad(np_img: np.ndarray, col_bounds: tuple[int, int]) -> Image.Image:
    h, _ = np_img.shape
    x0, x1 = col_bounds
    digit = np_img[:, x0:x1]
    row_sums = np.sum(digit > 20, axis=1)
    rows = np.where(row_sums > 0)[0]
    if len(rows) == 0:
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    y0, y1 = rows[0], rows[-1] + 1
    digit = digit[y0:y1, :]
    digit_img = Image.fromarray(digit.astype(np.uint8))
    max_side = max(digit_img.size)
    canvas = Image.new("L", (max_side, max_side), color=0)
    offset = ((max_side - digit_img.size[0]) // 2, (max_side - digit_img.size[1]) // 2)
    canvas.paste(digit_img, offset)
    return canvas.resize((28, 28), Image.LANCZOS)


def segment_digits(image_path: Path, expected_len: int | None = None) -> List[Image.Image]:
    img = normalize_student_id_image(Image.open(image_path))
    np_img = np.array(img)
    bounds = _find_digit_bounds(np_img)
    if expected_len and len(bounds) != expected_len:
        print(f"Warning: expected {expected_len} digits but found {len(bounds)} segments")
    return [_crop_and_pad(np_img, b) for b in bounds]


def predict_digits(
    image_path: Path,
    model_path: Path = MODEL_PATH,
    expected_len: int | None = None,
) -> str:
    model = load_trained_model(model_path)
    digits = segment_digits(image_path, expected_len=expected_len)
    transform = get_transforms()

    preds: list[str] = []
    with torch.no_grad():
        for idx, digit_img in enumerate(digits):
            digit_img.save(f"debug_{idx}.png")
            tensor = transform(digit_img).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).item()
            preds.append(str(pred))
            print(f"Segment {idx + 1}: predicted {pred}")
    return "".join(preds)


def save_metrics(result: TrainResult) -> None:
    summary_path = MODEL_DIR / "training_summary.json"
    summary = {
        "train_loss": result.train_loss,
        "test_acc": result.test_acc,
        "epochs": result.epochs,
        "model_path": result.model_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run MNIST student ID recognizer.")
    parser.add_argument("--train", action="store_true", help="Train the model on MNIST.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--predict",
        type=Path,
        help="Path to student ID image for prediction. Runs prediction only.",
    )
    parser.add_argument(
        "--expected-len",
        type=int,
        help="Expected digit count in the student ID. Used to sanity check segmentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train:
        result = train_model(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        save_metrics(result)

    if args.predict:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train first with `python test3.py --train`."
            )
        student_id = predict_digits(args.predict, expected_len=args.expected_len)
        print(f"Predicted student ID: {student_id}")

    if not args.train and not args.predict:
        print("Nothing to do. Use --train to train and --predict path/to/id.png to infer.")


if __name__ == "__main__":
    main()
