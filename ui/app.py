"""
O5 — Interfata de vizualizare si testare interactiva
====================================================
Se ruleaza din radacina proiectului:

    streamlit run ui/app.py

Modelele inghetate se incarca o singura data (~23s) si raman in cache. Perturbarile
se construiesc din ACELEASI primitive validate ca in O2, si trec prin aceleasi
verificari de validitate — deci nu se poate afisa o "evaziune" obtinuta cu un flux
imposibil fizic.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src import config, data_loader, inference
from src.perturbation import generators, schema
from ui import live_perturbation

OUTCOME_STYLE = {
    config.OUTCOME_CORRECT: ("#0ca30c", "detectat corect"),
    config.OUTCOME_MISCLASSIFIED_ATTACK: ("#fab219", "confundat cu alt atac"),
    config.OUTCOME_EVADED: ("#d03b3b", "EVAZIUNE"),
}

st.set_page_config(page_title="NIDS — robustete adversariala", layout="wide")


# --------------------------------------------------------------------------
# Incarcari cache-uite
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Se incarca modelele inghetate (~23s, o singura data)...")
def load_models():
    """Modelele inghetate, prin calea determinista din src.inference."""
    return inference.load_frozen_models(include_transformer=True)


@st.cache_resource(show_spinner="Se incarca datele...")
def load_data():
    """Setul de test, fluxurile de atac si contextul de perturbare."""
    train, test = data_loader.load_data()
    attack_rows = test[test[config.TARGET] != "Normal"].reset_index(drop=True)
    return {
        "test": test,
        "attack_rows": attack_rows,
        "features": schema.feature_columns(test),
        "ctx": generators.build_context(train, test),
    }


@st.cache_data(show_spinner=False)
def load_results(name: str):
    """Un CSV de rezultate din O3/O4, daca exista."""
    paths = {
        "evasion": config.EVASION_DIR / "evasion_summary.csv",
        "per_class": config.EVASION_DIR / "evasion_per_class.csv",
        "bounds": config.EVASION_DIR / "evasion_bounds.csv",
        "importance": config.SENSITIVITY_DIR / "permutation_importance.csv",
        "correlation": config.SENSITIVITY_DIR / "importance_evasion_correlation.csv",
        "attribution": config.SENSITIVITY_DIR / "importance_attribution.csv",
    }
    path = paths[name]
    return pd.read_csv(path) if path.exists() else None


def outcome_badge(outcome: str) -> str:
    """Eticheta colorata pentru un rezultat."""
    colour, label = OUTCOME_STYLE[outcome]
    return (f"<span style='background:{colour};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-size:0.85em'>{label}</span>")


# --------------------------------------------------------------------------
# Bara laterala — controalele de perturbare
# --------------------------------------------------------------------------
def perturbation_controls():
    """Controalele comune, in bara laterala.

    Returns:
        dict: parametrii de perturbare alesi.
    """
    st.sidebar.header("Perturbare")
    st.sidebar.caption("Toate modificarile respecta constrangerile fizice din O2: "
                       "padding-ul doar creste octetii, temporizarea doar creste durata, "
                       "contoarele de conexiuni doar scad.")

    change_ttl = st.sidebar.checkbox("Modifica TTL sursa (`sttl`)", value=True)
    target_sttl, ct_policy = None, "mimic"
    if change_ttl:
        target_sttl = st.sidebar.slider(
            "sttl tinta", config.MIN_REALISTIC_TTL, config.MAX_TTL,
            config.NORMAL_TTL_TARGET,
            help="Traficul normal are sttl=31 (70,5%); atacurile au 254 (86,7%). "
                 "Minimul e 16, ca pachetul sa ajunga la tinta.")
        ct_policy = st.sidebar.radio(
            "Politica `ct_state_ttl`", config.CT_STATE_TTL_POLICIES, index=1,
            help="ct_state_ttl e un contor pe fereastra glisanta, imposibil de "
                 "recalculat pentru un flux perturbat. 'hold' = neschimbat "
                 "(margine conservatoare); 'mimic' = semnatura tintei (optimista).")

    padding = st.sidebar.slider("Padding in payload (%)", 0, 200, 0, step=5) / 100.0
    duration = st.sidebar.slider("Dilatarea duratei (x)", 1.0, 10.0, 1.0, step=0.5)
    conn_scale = st.sidebar.slider("Contoare de conexiuni (x)", 0.25, 1.0, 1.0, step=0.05)

    return {"target_sttl": target_sttl, "ct_policy": ct_policy,
            "padding_fraction": padding, "duration_factor": duration,
            "connection_rate_scale": conn_scale}


# --------------------------------------------------------------------------
# Tab 1 — testare pe un grup de fluxuri
# --------------------------------------------------------------------------
def tab_batch(models, data, params):
    st.subheader("Testare interactiva pe un grup de fluxuri")
    st.caption("Perturbarea se aplica live si se masoara pe fluxurile ELIGIBILE — "
               "cele pe care modelul le semnaleaza ca malitioase pe traficul curat. "
               "Un flux nedetectat initial nu poate 'evada'.")

    attack_rows, features = data["attack_rows"], data["features"]
    classes = ["(toate)"] + sorted(attack_rows[config.TARGET].unique())

    col1, col2 = st.columns([2, 1])
    chosen = col1.selectbox("Clasa de atac", classes)
    max_rows = col2.slider("Fluxuri analizate", 200, 5000, 1500, step=100)

    subset = attack_rows if chosen == "(toate)" else attack_rows[attack_rows[config.TARGET] == chosen]
    # Esantion ALEATOR (cu seed fix), nu primele N randuri: setul de test e ordonat
    # dupa id, iar primele randuri nu sunt reprezentative pentru distributia claselor.
    if len(subset) > max_rows:
        subset = subset.sample(max_rows, random_state=config.RANDOM_SEED)
        st.caption(f"Esantion aleator de {max_rows:,} fluxuri (seed {config.RANDOM_SEED}); "
                   f"ratele pot diferi usor de cele masurate in O3 pe toate cele "
                   f"{len(attack_rows):,} fluxuri.")
    X = subset[features]
    y_true = subset[config.TARGET].to_numpy()

    perturbed, applied = live_perturbation.apply_custom_perturbation(X, data["ctx"], **params)
    valid, report = live_perturbation.check_validity(X, perturbed)

    if valid:
        st.success(f"Varianta trece toate cele {len(report)} verificari de validitate fizica "
                   f"din O2 — este un flux realizabil.")
    else:
        failed = report[report["violations"] > 0]["check"].tolist()
        st.error(f"Varianta INCALCA constrangerile: {', '.join(failed)}. "
                 f"Rezultatele de mai jos nu descriu un atac realizabil.")

    if not any([params["target_sttl"] is not None, params["padding_fraction"] > 0,
                params["duration_factor"] > 1.0, params["connection_rate_scale"] < 1.0]):
        st.info("Nicio perturbare selectata — foloseste controalele din stanga.")

    rows = []
    for name, model in models.items():
        clean_pred = model.predict(X)
        eligible = clean_pred != "Normal"
        if not eligible.any():
            continue
        pert_pred = model.predict(perturbed)
        outcomes = live_perturbation.classify(y_true[eligible], pert_pred[eligible])
        counts = {k: int((outcomes == k).sum()) for k in config.OUTCOME_ORDER}
        total = len(outcomes)
        rows.append({
            "Model": name,
            "Eligibile": total,
            "Evaziune (%)": round(100 * counts[config.OUTCOME_EVADED] / total, 2),
            "Confundate (%)": round(100 * counts[config.OUTCOME_MISCLASSIFIED_ATTACK] / total, 2),
            "Corecte (%)": round(100 * counts[config.OUTCOME_CORRECT] / total, 2),
        })

    if rows:
        summary = pd.DataFrame(rows)
        st.markdown("**Rata de evaziune** — fluxuri de atac devenite `Normal`")
        metric_cols = st.columns(len(summary))
        for col, (_, row) in zip(metric_cols, summary.iterrows()):
            col.metric(row["Model"], f"{row['Evaziune (%)']:.1f}%",
                       f"{row['Eligibile']:,} fluxuri eligibile", delta_color="off")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.bar_chart(summary.set_index("Model")[
            ["Corecte (%)", "Confundate (%)", "Evaziune (%)"]], height=280)

    with st.expander("Ce s-a modificat efectiv"):
        st.json(applied)
        changed = [c for c in X.columns if not X[c].equals(perturbed[c])]
        st.write(f"**Coloane modificate ({len(changed)}):** {', '.join(changed) or 'niciuna'}")


# --------------------------------------------------------------------------
# Tab 2 — un singur flux, in detaliu
# --------------------------------------------------------------------------
def tab_single(models, data, params):
    st.subheader("Inspectarea unui singur flux")
    attack_rows, features = data["attack_rows"], data["features"]

    col1, col2 = st.columns([2, 1])
    chosen = col1.selectbox("Clasa de atac", sorted(attack_rows[config.TARGET].unique()),
                            key="single_class")
    pool = attack_rows[attack_rows[config.TARGET] == chosen]
    position = col2.number_input("Indexul fluxului in clasa", 0, max(len(pool) - 1, 0), 0)

    row = pool.iloc[[int(position)]]
    X = row[features]
    st.caption(f"sample_id = {int(row[config.ID_COL].iloc[0])} · clasa reala: **{chosen}**")

    perturbed, _ = live_perturbation.apply_custom_perturbation(X, data["ctx"], **params)

    comparison = pd.DataFrame({
        "original": X.iloc[0].astype(str),
        "perturbat": perturbed.iloc[0].astype(str),
    })
    comparison["modificat"] = np.where(comparison["original"] != comparison["perturbat"], "da", "")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Caracteristici**")
        st.dataframe(comparison[comparison["modificat"] == "da"], width="stretch")
        with st.expander("Toate caracteristicile"):
            st.dataframe(comparison, width="stretch")

    with right:
        st.markdown("**Decizia fiecarui model**")
        for name, model in models.items():
            clean = model.predict(X)[0]
            after = model.predict(perturbed)[0]
            proba_after = model.predict_proba(perturbed)[0]
            outcome = live_perturbation.classify([chosen], [after])[0]

            st.markdown(f"### {name}")
            st.markdown(f"`{clean}` → `{after}` &nbsp; {outcome_badge(outcome)}",
                        unsafe_allow_html=True)
            top = pd.Series(proba_after, index=model.classes_).nlargest(4)
            st.bar_chart(top, height=180)


# --------------------------------------------------------------------------
# Tab 3 — rezultatele masurate (O3 / O4)
# --------------------------------------------------------------------------
def tab_results():
    st.subheader("Rezultatele masurate")
    evasion = load_results("evasion")
    if evasion is None:
        st.warning("Lipsesc rezultatele O3 — ruleaza `python run_evasion.py`.")
        return

    st.markdown("#### Rata de evaziune pe grila O2")
    realizable = evasion[(evasion["level"] > 0) & evasion["realizable"]]
    pivot = realizable.pivot_table(index="type", columns=["model", "level"],
                                   values="evasion_rate")
    st.dataframe(pivot.round(2), width="stretch")

    bounds = load_results("bounds")
    if bounds is not None:
        st.markdown("#### Intervale hold..mimic")
        st.caption("`ct_state_ttl` nu poate fi recalculat pentru un flux perturbat, deci "
                   "rezultatul corect e intervalul, nu una dintre margini.")
        st.dataframe(bounds, hide_index=True, width="stretch")

    figures = [
        ("Rata de evaziune pe niveluri", config.FIGURES_DIR / "evasion_curves.png"),
        ("Importanta prin permutare", config.FIGURES_DIR / "sensitivity_importance.png"),
        ("Importanta atinsa vs evaziune",
         config.FIGURES_DIR / "sensitivity_importance_vs_evasion.png"),
    ]
    for title, path in figures:
        if path.exists():
            st.markdown(f"#### {title}")
            st.image(str(path), width="stretch")


# --------------------------------------------------------------------------
# Tab 4 — de ce cedeaza modelele
# --------------------------------------------------------------------------
def tab_why():
    st.subheader("De ce cedeaza modelele")
    attribution = load_results("attribution")
    correlation = load_results("correlation")
    if attribution is None:
        st.warning("Lipsesc rezultatele O4 — ruleaza `python run_sensitivity.py`.")
        return

    st.markdown("#### Concentrarea importantei")
    st.caption("Cat din capacitatea de detectie sta intr-o singura caracteristica, si cat "
               "din ea poate fi atinsa de atacator.")
    st.dataframe(attribution, hide_index=True, width="stretch")
    st.info("XGBoost concentreaza peste jumatate din detectie in `sttl`, o caracteristica "
            "pe care atacatorul o schimba cu un apel `setsockopt`. Transformer-ul se "
            "sprijina cel mai mult pe `dttl` — TTL-ul **victimei**, inaccesibil "
            "atacatorului — de aceea evaziunea lui realizabila e marginita.")

    if correlation is not None:
        st.markdown("#### Corelatia dintre importanta atinsa si evaziunea produsa")
        st.dataframe(correlation, hide_index=True, width="stretch")

    importance = load_results("importance")
    if importance is not None:
        st.markdown("#### Importanta per caracteristica")
        model = st.selectbox("Model", sorted(importance["model"].unique()))
        top = (importance[importance["model"] == model]
               .nlargest(config.SENSITIVITY_TOP_FEATURES, "detection_drop")
               .set_index("feature")["detection_drop"])
        st.bar_chart(top, height=380)


# --------------------------------------------------------------------------
def main():
    st.title("Robustetea unui NIDS in fata modificarilor adversariale")
    st.caption("UNSW-NB15 · Random Forest, XGBoost si FT-Transformer, toate inghetate · "
               "perturbari validate fizic (O2), evaziune masurata (O3), explicata (O4)")

    models = load_models()
    data = load_data()
    params = perturbation_controls()

    st.sidebar.divider()
    st.sidebar.caption(f"Fluxuri de atac in test: {len(data['attack_rows']):,}\n\n"
                       f"Modele: {', '.join(models)}")

    tabs = st.tabs(["Testare pe grup", "Flux individual", "Rezultate masurate",
                    "De ce cedeaza modelele"])
    with tabs[0]:
        tab_batch(models, data, params)
    with tabs[1]:
        tab_single(models, data, params)
    with tabs[2]:
        tab_results()
    with tabs[3]:
        tab_why()


if __name__ == "__main__":
    main()
