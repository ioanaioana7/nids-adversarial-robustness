"""EDA: statistici descriptive, rapoarte de cardinalitate/asimetrie/corelatie si figuri."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


def numeric_feature_columns(train: pd.DataFrame) -> list[str]:
    """Returneaza numele coloanelor numerice (exclude categorice, id, target si label).

    Parameters:
        train (pd.DataFrame): cadrul din care se extrag coloanele.

    Returns:
        list[str]: numele coloanelor numerice, in ordinea din train.
    """
    exclude = config.CATEGORICAL_COLUMNS + [config.ID_COL, config.TARGET, config.LABEL_COL]
    return [c for c in train.columns if c not in exclude]


def compute_class_distribution(train: pd.DataFrame) -> pd.Series:
    """Calculeaza distributia claselor attack_cat, descrescator.

    Returns:
        pd.Series: numar de inregistrari per clasa.
    """
    return train[config.TARGET].value_counts()


def compute_categorical_report(train: pd.DataFrame) -> pd.DataFrame:
    """Calculeaza cardinalitatea si numarul de categorii rare pentru coloanele categorice.

    Returns:
        pd.DataFrame: coloane [column, nunique, rare_count, top5].
    """
    rows = []
    for col in config.CATEGORICAL_COLUMNS:
        vc = train[col].value_counts()
        rare = int((vc < config.MIN_CATEGORY_FREQUENCY).sum())
        top5 = ", ".join(f"{v}({n})" for v, n in vc.head(5).items())
        rows.append({"column": col, "nunique": int(vc.size), "rare_count": rare, "top5": top5})
    return pd.DataFrame(rows)


def compute_skew(train: pd.DataFrame, numeric_cols: list[str]) -> pd.Series:
    """Calculeaza |skew()| pentru fiecare coloana numerica, descrescator."""
    return train[numeric_cols].skew().abs().sort_values(ascending=False)


def compute_correlation_with_label(train: pd.DataFrame, numeric_cols: list[str]) -> pd.Series:
    """Calculeaza corelatia Pearson dintre fiecare coloana numerica si label (binar).

    Returns:
        pd.Series: corelatii semnate, ordonate dupa valoarea absoluta.
    """
    corr = train[numeric_cols + [config.LABEL_COL]].corr()[config.LABEL_COL].drop(config.LABEL_COL)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def save_eda_csvs(dist: pd.Series, cat_report: pd.DataFrame, skew: pd.Series, corr: pd.Series) -> None:
    """Salveaza statisticile EDA ca CSV in config.EDA_DIR."""
    dist.rename_axis(config.TARGET).reset_index(name="count") \
        .to_csv(config.EDA_DIR / "eda_class_distribution.csv", index=False)
    cat_report.to_csv(config.EDA_DIR / "eda_categorical_cardinality.csv", index=False)
    skew.rename_axis("feature").reset_index(name="abs_skew") \
        .to_csv(config.EDA_DIR / "eda_skew.csv", index=False)
    corr.rename_axis("feature").reset_index(name="correlation_with_label") \
        .to_csv(config.EDA_DIR / "eda_correlation.csv", index=False)


def plot_class_distribution(dist: pd.Series) -> None:
    """Salveaza graficul de distributie a claselor (bare, scara log)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    dist.sort_values(ascending=False).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Numar inregistrari (scara log)")
    ax.set_title("Distributia claselor de atac")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close()


def plot_correlation_with_label(corr: pd.Series) -> None:
    """Salveaza graficul cu top 15 features numerice dupa |corelatie cu label|."""
    top = corr.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.iloc[::-1].plot(kind="barh", ax=ax, color="darkorange")
    ax.set_xlabel("corelatie cu label")
    ax.set_title("Top 15 features numerice corelate cu label")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "eda_correlation_label.png", dpi=150)
    plt.close()


def plot_numeric_distributions(train: pd.DataFrame, cols: list[str]) -> None:
    """Salveaza histogramele (log1p) pentru cele mai asimetrice features numerice."""
    n = len(cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        np.log1p(train[col]).plot(kind="hist", bins=40, ax=ax, color="teal")
        ax.set_title(col)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Distributii numerice (log1p, top features asimetrice)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "eda_numeric_distributions.png", dpi=150)
    plt.close()


def run_eda(train: pd.DataFrame, logger) -> dict:
    """Ruleaza EDA completa: logheaza rezumatul si salveaza CSV-uri + figuri.

    Parameters:
        train (pd.DataFrame): setul de antrenare (cu attack_cat si label prezente).
        logger (logging.Logger): logger pentru mesaje (consola + training.log).

    Returns:
        dict: statisticile calculate (class_distribution, categorical_report, skew, correlation).
    """
    logger.info("=" * 60)
    logger.info("EDA RAPID")
    logger.info("=" * 60)
    logger.info(f"Dimensiune train: {train.shape}")

    dist = compute_class_distribution(train)
    logger.info(f"\nDistributia claselor ({config.TARGET}):")
    for cls, n in dist.items():
        logger.info(f"  {cls:<18} {n:>8}  ({100 * n / len(train):.1f}%)")
    logger.info(f"\nDezechilibru (max/min): {dist.max() / dist.min():.0f}x")

    present_ttl = [c for c in config.TTL_FEATURES if c in train.columns]
    logger.info(f"Features TTL disponibile (pentru perturbare): {present_ttl}")

    cat_report = compute_categorical_report(train)
    logger.info(f"\nCardinalitate categorii (prag grupare rare: {config.MIN_CATEGORY_FREQUENCY}):")
    for _, row in cat_report.iterrows():
        logger.info(f"  {row['column']:<10} nunique={row['nunique']:<4} "
                    f"rare(<{config.MIN_CATEGORY_FREQUENCY})={row['rare_count']:<4} top5: {row['top5']}")

    numeric_cols = numeric_feature_columns(train)
    skew = compute_skew(train, numeric_cols)
    logger.info("\nTop 10 features numerice dupa asimetrie (|skew|):")
    for col, val in skew.head(10).items():
        logger.info(f"  {col:<20} {val:.2f}")

    corr = compute_correlation_with_label(train, numeric_cols)

    save_eda_csvs(dist, cat_report, skew, corr)
    plot_class_distribution(dist)
    plot_correlation_with_label(corr)
    plot_numeric_distributions(train, skew.head(6).index.tolist())
    logger.info("=" * 60)
    logger.info("[salvat] eda_*.csv in results/eda/, eda_*.png in results/figures/")

    return {
        "class_distribution": dist,
        "categorical_report": cat_report,
        "skew": skew,
        "correlation": corr,
    }
