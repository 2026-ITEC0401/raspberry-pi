"""Regenerate repository-friendly v2 evaluation figures from committed data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = RESULT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def configure_korean_font() -> None:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            name = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(metrics: dict) -> None:
    development = metrics["champion_development"]
    test = metrics["untouched_test"]
    names = ["표적 macro-F1", "전체 macro-F1", "Accuracy"]
    keys = ["target_macro_f1", "all_class_macro_f1", "accuracy"]
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_dev = ax.bar(x - width / 2, [development[key] for key in keys], width, label="Development OOF")
    bars_test = ax.bar(x + width / 2, [test[key] for key in keys], width, label="격리 test")
    for bars in (bars_dev, bars_test):
        ax.bar_label(bars, labels=[f"{bar.get_height():.3f}" for bar in bars], padding=3)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, names)
    ax.set_ylabel("점수")
    ax.set_title("Hearo YAMNet v2 핵심 성능")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    note = (
        f"비표적 false-alert: development {development['false_alert_rate']:.3%}, "
        f"test {test['false_alert_rate']:.3%}"
    )
    ax.text(0.5, -0.17, note, transform=ax.transAxes, ha="center")
    save(fig, "1_test_summary.png")


def plot_confusion_matrix() -> None:
    frame = pd.read_csv(RESULT_DIR / "confusion_matrix.csv")
    labels = frame.pop("true_label").tolist()
    matrix = frame.to_numpy(dtype=int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("예측 클래스")
    ax.set_ylabel("실제 클래스")
    ax.set_title("격리 test confusion matrix (행 기준 비율, 괄호 안 실제 개수)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ratio = normalized[row, column]
            ax.text(
                column,
                row,
                f"{ratio:.0%}\n({value})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if ratio > 0.5 else "black",
            )
    save(fig, "2_test_confusion_matrix.png")


def plot_per_class() -> None:
    frame = pd.read_csv(RESULT_DIR / "test_per_class_metrics.csv", encoding="utf-8-sig")
    x = np.arange(len(frame))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.bar(x - width, frame["precision"], width, label="Precision")
    ax.bar(x, frame["recall"], width, label="Recall")
    bars = ax.bar(x + width, frame["f1"], width, label="F1")
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in frame["f1"]], padding=2, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, frame["class"], rotation=40, ha="right")
    ax.set_ylabel("점수")
    ax.set_title("격리 test 클래스별 성능")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "3_test_per_class_metrics.png")


def plot_candidates() -> None:
    frame = pd.read_csv(RESULT_DIR / "experiment_results.csv", encoding="utf-8-sig")
    labels = frame["family"] + "\n" + frame["pooling"]
    colors = ["#168aad" if promoted else "#adb5bd" for promoted in frame["promoted"]]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    bars = ax.bar(np.arange(len(frame)), frame["target_macro_f1"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in frame["target_macro_f1"]], padding=3)
    ax.set_ylim(0, 0.82)
    ax.set_xticks(np.arange(len(frame)), labels, rotation=35, ha="right")
    ax.set_ylabel("Development target macro-F1")
    ax.set_title("1라운드 후보와 pooling 비교")
    ax.axhline(frame.loc[0, "target_macro_f1"], color="#d00000", linestyle="--", linewidth=1, label="incumbent")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "4_candidate_comparison.png")


def main() -> None:
    configure_korean_font()
    metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
    plot_summary(metrics)
    plot_confusion_matrix()
    plot_per_class()
    plot_candidates()
    print(f"generated figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
