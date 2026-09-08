from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class Config:
    data_dir: str
    output_dir: str
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_size: int = 5000
    num_workers: int = 4
    seed: int = 42
    embedding_dim: int = 64
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    force_cpu: bool = False


class MnistCNN(nn.Module):
    """CNN whose penultimate activation is the deep feature vector."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.projector(self.features(x))
        logits = self.classifier(embedding)
        if return_features:
            return logits, embedding
        return logits


def parse_args() -> Config:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train an MNIST CNN and analyze its deep features with PCA + K-Means."
    )
    parser.add_argument("--data-dir", default=str(project_dir / "data"))
    parser.add_argument("--output-dir", default=str(project_dir / "results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    return Config(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaders(config: Config, use_cuda: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_full = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)

    if config.max_train_samples is not None:
        limit = min(config.max_train_samples, len(train_full))
        train_full = Subset(train_full, range(limit))
    if config.max_test_samples is not None:
        limit = min(config.max_test_samples, len(test_set))
        test_set = Subset(test_set, range(limit))

    if config.val_size <= 0 or config.val_size >= len(train_full):
        raise ValueError("val_size must be greater than 0 and smaller than the training set")
    train_size = len(train_full) - config.val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set = random_split(
        train_full, [train_size, config.val_size], generator=generator
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if is_training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.inference_mode()
def extract_features(
    model: MnistCNN, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits, embeddings = model(images, return_features=True)
        all_features.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
    )


def match_clusters_to_digits(
    true_labels: np.ndarray, clusters: np.ndarray, digits: np.ndarray
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    counts = np.zeros((len(digits), len(digits)), dtype=np.int64)
    for cluster_id, digit in zip(clusters, true_labels):
        counts[int(cluster_id), int(np.where(digits == digit)[0][0])] += 1
    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping = {int(row): int(digits[col]) for row, col in zip(row_ind, col_ind)}
    mapped = np.array([mapping[int(cluster)] for cluster in clusters], dtype=np.int64)
    return mapped, mapping, counts


def save_training_curves(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [100 * row["val_accuracy"] for row in history], marker="o", label="validation")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pca_plot(
    points: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        points[:, 0], points[:, 1], c=values, cmap="tab10", s=9, alpha=0.58, linewidths=0
    )
    colorbar = fig.colorbar(scatter, ax=ax, ticks=np.unique(values))
    colorbar.set_label(colorbar_label)
    ax.set(
        title=title,
        xlabel=f"PC1 ({explained_variance[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({explained_variance[1] * 100:.1f}% variance)",
    )
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plot(matrix: np.ndarray, digits: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="K-Means clusters after Hungarian label matching",
        xlabel="Mapped cluster label",
        ylabel="True digit",
        xticks=np.arange(len(digits)),
        yticks=np.arange(len(digits)),
        xticklabels=digits,
        yticklabels=digits,
    )
    threshold = matrix.max() * 0.55
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def interpret_metrics(metrics: dict[str, float]) -> str:
    accuracy = metrics["cluster_accuracy_hungarian"]
    ari = metrics["adjusted_rand_index"]
    if accuracy >= 0.90 and ari >= 0.80:
        return "二维深层特征呈现很强的类别可分性；多数数字形成了与真实标签高度一致的簇。"
    if accuracy >= 0.80 and ari >= 0.65:
        return "二维深层特征总体可分，但仍有少数外形相近的数字在投影后重叠。"
    if accuracy >= 0.65 and ari >= 0.45:
        return "二维投影显示中等可分性；PCA 压缩和 K-Means 的球状簇假设造成了较明显的信息损失。"
    return "仅凭二维 PCA + K-Means 尚不能认为这些特征充分可分；应同时检查原始高维特征或使用非线性可视化。"


def save_report(
    config: Config,
    metrics: dict[str, float],
    mapping: dict[int, int],
    confusion: np.ndarray,
    digits: np.ndarray,
    device_name: str,
    output_dir: Path,
) -> None:
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    confusing_pairs: list[tuple[int, int, int]] = []
    for row in range(len(digits)):
        for col in range(len(digits)):
            if row != col and off_diagonal[row, col] > 0:
                confusing_pairs.append((int(off_diagonal[row, col]), int(digits[row]), int(digits[col])))
    confusing_pairs.sort(reverse=True)
    pair_text = "、".join(f"{a}→{b} ({count})" for count, a, b in confusing_pairs[:5]) or "无"
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    recall_text = "，".join(f"{digit}: {value:.1%}" for digit, value in zip(digits, recalls))
    mapping_text = "，".join(f"簇 {cluster}→数字 {digit}" for cluster, digit in sorted(mapping.items()))

    report = f"""# MNIST CNN 深层特征的 PCA + K-Means 分析

## 结论

{interpret_metrics(metrics)}

这里的“可分”专指：CNN 倒数第二层的 {config.embedding_dim} 维特征经过标准化、PCA 压缩到二维后，仍能被 9 类 K-Means 恢复到何种程度。聚类过程没有使用真实标签；真实标签只用于事后评估和着色。

## 核心指标

| 指标 | 结果 |
|---|---:|
| CNN 测试集准确率（0–9） | {metrics['cnn_test_accuracy_all_digits']:.2%} |
| CNN 测试集准确率（1–9） | {metrics['cnn_test_accuracy_digits_1_to_9']:.2%} |
| 匈牙利匹配后的聚类准确率 | {metrics['cluster_accuracy_hungarian']:.2%} |
| 聚类纯度（Purity） | {metrics['cluster_purity']:.2%} |
| Adjusted Rand Index | {metrics['adjusted_rand_index']:.4f} |
| Normalized Mutual Information | {metrics['normalized_mutual_info']:.4f} |
| Silhouette Score | {metrics['silhouette_score']:.4f} |
| PC1 + PC2 累计解释方差 | {metrics['pca_explained_variance_2d']:.2%} |

各数字经簇映射后的召回率：{recall_text}

最常见的映射错误（真实数字→映射簇标签）：{pair_text}

K-Means 簇与数字的最优匹配：{mapping_text}

## 如何阅读结果

- `pca_true_labels.png`：按真实数字着色，直接观察相同数字是否聚集、不同数字是否重叠。
- `pca_kmeans_clusters.png`：按无监督 K-Means 簇着色，对比聚类边界与真实类别结构。
- `cluster_confusion_matrix.png`：先用匈牙利算法把簇编号匹配到数字，再查看具体混淆来源。
- ARI/NMI 不依赖簇编号；匈牙利准确率和纯度更直观，但不要单独依赖某一个指标。
- 二维 PCA 是强压缩。如果二维结果不理想，并不等价于 {config.embedding_dim} 维特征本身不可分。

## 实验设置

- 训练类别：MNIST 0–9；聚类分析类别：1–9（排除 0）。
- 深层特征：分类器前的 {config.embedding_dim} 维激活。
- 训练轮数：{config.epochs}；随机种子：{config.seed}；设备：{device_name}。
- K-Means：`n_clusters=9, n_init=20`，输入为二维 PCA 坐标。
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def analyze_features(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    config: Config,
    full_test_accuracy: float,
    device_name: str,
) -> dict[str, float]:
    mask = (labels >= 1) & (labels <= 9)
    selected_features = features[mask]
    selected_labels = labels[mask]
    selected_predictions = predictions[mask]
    digits = np.arange(1, 10)

    standardized = StandardScaler().fit_transform(selected_features)
    pca = PCA(n_components=2, random_state=config.seed)
    points = pca.fit_transform(standardized)
    kmeans = KMeans(n_clusters=9, n_init=20, random_state=config.seed)
    clusters = kmeans.fit_predict(points)
    mapped_clusters, mapping, cluster_counts = match_clusters_to_digits(
        selected_labels, clusters, digits
    )

    conf = confusion_matrix(selected_labels, mapped_clusters, labels=digits)
    metrics = {
        "cnn_test_accuracy_all_digits": float(full_test_accuracy),
        "cnn_test_accuracy_digits_1_to_9": float(np.mean(selected_predictions == selected_labels)),
        "cluster_accuracy_hungarian": float(np.mean(mapped_clusters == selected_labels)),
        "cluster_purity": float(cluster_counts.max(axis=1).sum() / cluster_counts.sum()),
        "adjusted_rand_index": float(adjusted_rand_score(selected_labels, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(selected_labels, clusters)),
        "silhouette_score": float(silhouette_score(points, clusters)),
        "pca_explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
        "pca_explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
        "analyzed_sample_count": int(len(selected_labels)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "cluster_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "pca_features.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc1", "pc2", "true_digit", "kmeans_cluster", "mapped_digit", "cnn_prediction"])
        writer.writerows(
            zip(
                points[:, 0],
                points[:, 1],
                selected_labels,
                clusters,
                mapped_clusters,
                selected_predictions,
            )
        )
    with (output_dir / "cluster_label_crosstab.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster"] + [f"digit_{digit}" for digit in digits])
        for cluster_id, row in enumerate(cluster_counts):
            writer.writerow([cluster_id] + row.tolist())

    save_pca_plot(
        points,
        selected_labels,
        "Deep features in PCA space — colored by true digit",
        "True digit",
        output_dir / "pca_true_labels.png",
        pca.explained_variance_ratio_,
    )
    save_pca_plot(
        points,
        clusters,
        "Deep features in PCA space — colored by K-Means cluster",
        "Cluster ID",
        output_dir / "pca_kmeans_clusters.png",
        pca.explained_variance_ratio_,
    )
    save_confusion_plot(conf, digits, output_dir / "cluster_confusion_matrix.png")
    save_report(config, metrics, mapping, conf, digits, device_name, output_dir)
    return metrics


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    use_cuda = torch.cuda.is_available() and not config.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Using device: {device_name}")
    if use_cuda:
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    train_loader, val_loader, test_loader = build_loaders(config, use_cuda)
    model = MnistCNN(config.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    checkpoint_path = output_dir / "best_mnist_cnn.pt"
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{config.epochs}: "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.2%}; "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.2%}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_val_accuracy": best_val_accuracy,
                    "feature_layer": "projector output before classifier",
                },
                checkpoint_path,
            )

    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_training_curves(history, output_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Test loss={test_loss:.4f}, accuracy={test_accuracy:.2%}")

    features, labels, predictions = extract_features(model, test_loader, device)
    metrics = analyze_features(
        features, labels, predictions, output_dir, config, test_accuracy, device_name
    )
    print("Analysis complete:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}" if isinstance(value, float) else f"  {name}: {value}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Avoid excessive CPU thread contention when DataLoader workers are enabled on Windows.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
