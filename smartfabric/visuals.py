from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summary_chart(df: pd.DataFrame) -> BytesIO | None:
    if df.empty:
        return None

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    sns.countplot(data=df, x="label", order=df["label"].value_counts().index, ax=axes[0], color="#0EA5A5")
    axes[0].set_title("Defects by type")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    severity_order = ["Low", "Medium", "High"]
    sns.countplot(data=df, x="severity", order=severity_order, ax=axes[1], palette=["#22C55E", "#F59E0B", "#EF4444"], hue="severity", legend=False)
    axes[1].set_title("Severity mix")
    axes[1].set_xlabel("")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer
