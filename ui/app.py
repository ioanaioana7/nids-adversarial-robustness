"""
O5 — interfata de vizualizare si testare interactiva.

    python -m streamlit run ui/app.py

Perturbarile se construiesc din primitivele validate in O2 (vezi
ui/live_perturbation.py), deci ce se vede aici e acelasi lucru cu ce s-a masurat
in O3 — nu o reimplementare paralela.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config, data_loader, inference
from src.perturbation import generators, schema
from ui import live_perturbation

OUTCOME_COLOR = {
    config.OUTCOME_CORRECT: "#0ca30c",
    config.OUTCOME_MISCLASSIFIED_ATTACK: "#fab219",
    config.OUTCOME_EVADED: "#d03b3b",
}
OUTCOME_LABEL = {
    config.OUTCOME_CORRECT: "detectat corect",
    config.OUTCOME_MISCLASSIFIED_ATTACK: "alt atac",
    config.OUTCOME_EVADED: "evaziune",
}

st.set_page_config(page_title="Robustete NIDS", layout="wide")


@st.cache_resource(show_spinner="Se incarca modelele...")
def load_models():
    return inference.load_frozen_models(include_transformer=True)


@st.cache_resource(show_spinner="Se incarca datele...")
def load_data():
    train, test = data_loader.load_data()
    attacks = test[test[config.TARGET] != "Normal"].reset_index(drop=True)
    return attacks, schema.feature_columns(test), generators.build_context(train, test)


@st.cache_data(show_spinner=False)
def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def controls():
    """Controalele din bara laterala.

    Sunt intr-un formular: valorile se scriu, iar nimic nu se recalculeaza pana
    la apasarea butonului. Ce e afisat in dreapta corespunde intotdeauna
    ultimei rulari, nu campurilor editate intre timp.
    """
    attacks, _, _ = load_data()
    classes = ["toate"] + sorted(attacks[config.TARGET].unique())

    with st.sidebar.form("parametri"):
        st.subheader("Trafic")
        attack_class = st.selectbox("Clasa", classes)
        n_flows = st.number_input("Fluxuri", 100, 45000, 2000, step=500)

        st.subheader("Modificari")
        use_ttl = st.checkbox("Modifica TTL sursa", value=True)
        sttl = st.number_input("sttl", config.MIN_REALISTIC_TTL, config.MAX_TTL,
                               config.NORMAL_TTL_TARGET, step=1)
        padding = st.number_input("Payload adaugat (%)", 0, 500, 0, step=5)
        duration = st.number_input("Durata (x)", 1.0, 100.0, 1.0, step=0.5, format="%.1f")
        conn = st.number_input("Contoare conexiuni (x)", 0.01, 1.0, 1.0,
                               step=0.05, format="%.2f")
        policy = st.selectbox("ct_state_ttl", config.CT_STATE_TTL_POLICIES, index=1)

        submitted = st.form_submit_button("Ruleaza", type="primary",
                                          use_container_width=True)

    settings = {
        "attack_class": attack_class, "n_flows": int(n_flows),
        "params": {"target_sttl": int(sttl) if use_ttl else None, "ct_policy": policy,
                   "padding_fraction": padding / 100.0, "duration_factor": duration,
                   "connection_rate_scale": conn},
    }
    if submitted or "settings" not in st.session_state:
        st.session_state.settings = settings
    return st.session_state.settings


def describe(settings) -> str:
    """Rezumatul intr-un rand al parametrilor efectiv rulati."""
    p = settings["params"]
    parts = [settings["attack_class"], f"{settings['n_flows']} fluxuri"]
    if p["target_sttl"] is not None:
        parts.append(f"sttl={p['target_sttl']} ({p['ct_policy']})")
    if p["padding_fraction"] > 0:
        parts.append(f"payload +{p['padding_fraction'] * 100:.0f}%")
    if p["duration_factor"] > 1:
        parts.append(f"durata x{p['duration_factor']:.1f}")
    if p["connection_rate_scale"] < 1:
        parts.append(f"conexiuni x{p['connection_rate_scale']:.2f}")
    return " · ".join(parts)


def measure(models, X, y_true, perturbed):
    """Rezultatele per model, pe fluxurile eligibile."""
    rows = []
    for name, model in models.items():
        clean = model.predict(X)
        eligible = clean != "Normal"
        if not eligible.any():
            continue
        outcomes = live_perturbation.classify(y_true[eligible],
                                              model.predict(perturbed)[eligible])
        total = len(outcomes)
        row = {"model": name, "eligible": total}
        for key in config.OUTCOME_ORDER:
            row[key] = 100.0 * float((outcomes == key).sum()) / total
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_chart(results: pd.DataFrame):
    """Bara orizontala stivuita: cum se distribuie cele trei rezultate."""
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(results) + 0.9))
    positions = np.arange(len(results))
    left = np.zeros(len(results))
    for key in config.OUTCOME_ORDER:
        values = results[key].to_numpy()
        ax.barh(positions, values, left=left, height=0.62, color=OUTCOME_COLOR[key],
                label=OUTCOME_LABEL[key], edgecolor="white", linewidth=1.5)
        left += values

    ax.set_yticks(positions)
    ax.set_yticklabels(results["model"], fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% din fluxurile eligibile", fontsize=9, color="#52514e")
    ax.invert_yaxis()
    ax.tick_params(length=0, colors="#52514e")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
              frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def view_result(models, data, settings):
    attacks, features, ctx = data
    subset = attacks if settings["attack_class"] == "toate" else \
        attacks[attacks[config.TARGET] == settings["attack_class"]]
    if len(subset) > settings["n_flows"]:
        subset = subset.sample(settings["n_flows"], random_state=config.RANDOM_SEED)

    X = subset[features]
    y_true = subset[config.TARGET].to_numpy()
    perturbed, _ = live_perturbation.apply_custom_perturbation(X, ctx, **settings["params"])

    valid, report = live_perturbation.check_validity(X, perturbed)
    if not valid:
        failed = report[report["violations"] > 0]["check"].tolist()
        st.error(f"Flux imposibil fizic — {', '.join(failed)}")

    results = measure(models, X, y_true, perturbed)
    if results.empty:
        st.warning("Niciun flux eligibil in selectie.")
        return

    columns = st.columns(len(results))
    for column, (_, row) in zip(columns, results.iterrows()):
        column.metric(row["model"], f"{row[config.OUTCOME_EVADED]:.0f}%",
                      f"{int(row['eligible']):,} fluxuri", delta_color="off")

    st.pyplot(outcome_chart(results), width="stretch")

    changed = [c for c in X.columns if not X[c].equals(perturbed[c])]
    st.caption(f"Modificat: {', '.join(changed) if changed else 'nimic'}")


def view_flow(models, data, settings):
    attacks, features, ctx = data
    pool = attacks if settings["attack_class"] == "toate" else \
        attacks[attacks[config.TARGET] == settings["attack_class"]]
    index = st.number_input("Flux", 0, max(len(pool) - 1, 0), 0, label_visibility="collapsed")

    row = pool.iloc[[int(index)]]
    X = row[features]
    true_label = row[config.TARGET].iloc[0]
    st.caption(f"id {int(row[config.ID_COL].iloc[0])} · {true_label}")
    perturbed, _ = live_perturbation.apply_custom_perturbation(X, ctx, **settings["params"])

    left, right = st.columns([1, 1])
    with left:
        before = X.iloc[0].astype(str)
        after = perturbed.iloc[0].astype(str)
        changed = pd.DataFrame({"inainte": before, "dupa": after})
        changed = changed[before != after]
        st.dataframe(changed if not changed.empty else pd.DataFrame({"": ["nicio modificare"]}),
                     width="stretch")

    with right:
        for name, model in models.items():
            after_label = model.predict(perturbed)[0]
            outcome = live_perturbation.classify([true_label], [after_label])[0]
            color = OUTCOME_COLOR[outcome]
            st.markdown(
                f"**{name}** &nbsp; `{model.predict(X)[0]}` → "
                f"<span style='color:{color};font-weight:600'>{after_label}</span>",
                unsafe_allow_html=True)


def view_measurements():
    evasion = read_csv(config.EVASION_DIR / "evasion_summary.csv")
    if evasion is None:
        st.warning("Ruleaza `python run_evasion.py`.")
        return

    realizable = evasion[(evasion["level"] > 0) & evasion["realizable"]]
    st.dataframe(
        realizable.pivot_table(index="type", columns=["model", "level"],
                               values="evasion_rate").round(1),
        width="stretch")

    for path in [config.FIGURES_DIR / "evasion_curves.png",
                 config.FIGURES_DIR / "sensitivity_importance.png"]:
        if path.exists():
            st.image(str(path), width="stretch")


def main():
    st.title("Robustetea unui NIDS la modificari adversariale")

    models = load_models()
    data = load_data()
    settings = controls()
    st.caption(describe(settings))

    result, flow, measurements = st.tabs(["Rezultat", "Un flux", "Masuratori"])
    with result:
        view_result(models, data, settings)
    with flow:
        view_flow(models, data, settings)
    with measurements:
        view_measurements()


if __name__ == "__main__":
    main()
