"""
Interfata de vizualizare si testare interactiva.

    python -m streamlit run ui/app.py

Perturbarile se construiesc din primitivele validate (vezi ui/live_perturbation.py)

Tabul "Antrenare adversariala" citeste artefactele produse de train_adversarial.py
si lipseste, in mod controlat, daca etapa nu a fost rulata.
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
    config.OUTCOME_MISCLASSIFIED_ATTACK: "confundat cu alt atac",
    config.OUTCOME_EVADED: "evaziune",
}

st.set_page_config(page_title="Robustețea unui NIDS", layout="wide",
                   page_icon="🛡️")


@st.cache_resource(show_spinner="Se încarcă modelele...")
def load_models():
    return inference.load_frozen_models(include_transformer=True)


@st.cache_resource(show_spinner=False)
def load_defended_models():
    """Modelele reantrenate cu apărare (brațul O6), câte un singur seed per model.

    Seed-ul ales (primul din O6_TREE_SEEDS / O6_TRANSFORMER_SEEDS) e doar un
    reprezentant pentru demonstrația live; cifrele raportate în lucrare, în fila
    „Efectul apărării”, folosesc media pe toate cele trei repetări. Lipsește, în
    mod controlat, dacă `train_adversarial.py` nu a fost încă rulat.
    """
    from src.adversarial import trees as adv_trees
    from src.adversarial import transformer_arm
    from src.transformer.predictor import TransformerNIDS

    arm = config.O6_ARM_DEFENDED
    models = {}
    try:
        models["RandomForest"] = adv_trees.load_tree_model(
            "RandomForest", arm, config.O6_TREE_SEEDS[0])
        models["XGBoost"] = adv_trees.load_tree_model(
            "XGBoost", arm, config.O6_TREE_SEEDS[0])
        models["Transformer"] = TransformerNIDS.load(
            transformer_arm.model_path(arm, config.O6_TRANSFORMER_SEEDS[0]))
    except FileNotFoundError:
        return {}
    return models


@st.cache_resource(show_spinner=False)
def load_data():
    train, test = data_loader.load_data()
    attacks = test[test[config.TARGET] != "Normal"].reset_index(drop=True)
    return attacks, schema.feature_columns(test), generators.build_context(train, test)


@st.cache_data(show_spinner=False)
def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def controls():
    """Controalele din bara laterala.

    Sunt intr-un formular: valorile se scriu, iar nimic nu se recalculeaza pana
    la apasarea butonului. Ce e afisat in dreapta corespunde intotdeauna
    ultimei rulari, nu campurilor editate intre timp.
    """
    attacks, _, _ = load_data()
    classes = ["toate"] + sorted(attacks[config.TARGET].unique())

    with st.sidebar.form("parametri"):
        st.subheader("Ce trafic se atacă")
        attack_class = st.selectbox(
            "Categoria de atac (attack\\_cat)", classes,
            help="„toate” amestecă toate cele nouă categorii de atac din set.")
        n_flows = st.number_input(
            "Câte fluxuri", 100, 45000, 2000, step=500,
            help="Eșantion mai mare înseamnă estimare mai stabilă, dar calcul mai lung.")

        st.subheader("Ce modifică atacatorul")
        use_ttl = st.checkbox("Schimbă valoarea TTL (ttl)", value=True,
                              help="TTL-ul pachetelor trimise de atacator.")
        sttl = st.number_input("Valoarea TTL țintă (sttl)",
                               config.MIN_REALISTIC_TTL, config.MAX_TTL,
                               config.NORMAL_TTL_TARGET, step=1,
                               help="31 este valoarea dominantă a traficului legitim.")
        padding = st.number_input(
            "Octeți de umplutură (padding), %", 0, 500, 0, step=5,
            help="Crește volumul trimis, fără a schimba conținutul atacului.")
        duration = st.number_input(
            "Întinderea în timp (timing), ×", 1.0, 100.0, 1.0, step=0.5, format="%.1f",
            help="×2 înseamnă că atacul durează de două ori mai mult.")
        conn = st.number_input(
            "Ritmul conexiunilor (connection\\_rate), ×", 0.01, 1.0, 1.0,
            step=0.05, format="%.2f",
            help="Sub 1 înseamnă atac mai lent, cu mai puține conexiuni pe secundă.")
        policy = st.selectbox(
            "Ipoteza pentru contorul de context (ct\\_state\\_ttl)",
            config.CT_STATE_TTL_POLICIES, index=1,
            help="Contorul stare–TTL nu poate fi recalculat dintr-un flux izolat. "
                 "„hold” îl lasă neatins, „mimic” presupune că atacatorul "
                 "reușește să imite și semnătura traficului legitim.")

        submitted = st.form_submit_button("Rulează", type="primary",
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
        parts.append(f"TTL → {p['target_sttl']} ({p['ct_policy']})")
    if p["padding_fraction"] > 0:
        parts.append(f"umplutură +{p['padding_fraction'] * 100:.0f}%")
    if p["duration_factor"] > 1:
        parts.append(f"durată ×{p['duration_factor']:.1f}")
    if p["connection_rate_scale"] < 1:
        parts.append(f"conexiuni ×{p['connection_rate_scale']:.2f}")
    if len(parts) == 2:
        parts.append("nicio modificare")
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


def view_result(models, defended, data, settings):
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
        st.error("Combinația aceasta ar produce un flux imposibil fizic, deci rezultatul "
                 f"nu ar însemna nimic. Constrângeri încălcate: {', '.join(failed)}.")

    if defended:
        combined = {}
        for name, model in models.items():
            combined[f"{name} - referință"] = model
            if name in defended:
                combined[f"{name} - reantrenat"] = defended[name]
    else:
        combined = models
        st.caption("Modelele reantrenate (O6) nu sunt disponibile — rulează "
                   "`python train_adversarial.py` ca să apară aici, alături de "
                   "cele de referință.")

    results = measure(combined, X, y_true, perturbed)
    if results.empty:
        st.warning("Niciun flux din selecție nu era detectat înainte de modificare, "
                   "deci nu există pe ce măsura evaziunea. Încearcă alt eșantion.")
        return

    st.caption("Procentul atacurilor care scapă nedetectate după modificare, dintre cele "
               "pe care modelul le prindea înainte.")
    columns = st.columns(len(results))
    for column, (_, row) in zip(columns, results.iterrows()):
        column.metric(row["model"], f"{row[config.OUTCOME_EVADED]:.0f}% evaziune",
                      f"din {int(row['eligible']):,} fluxuri detectate inițial",
                      delta_color="off")

    st.pyplot(outcome_chart(results), width="stretch")

    changed = [c for c in X.columns if not X[c].equals(perturbed[c])]
    st.caption(f"Caracteristici modificate efectiv ({len(changed)}): "
               f"{', '.join(changed) if changed else 'niciuna'}")


def view_flow(models, defended, data, settings):
    attacks, features, ctx = data
    pool = attacks if settings["attack_class"] == "toate" else \
        attacks[attacks[config.TARGET] == settings["attack_class"]]
    st.caption("Alege un flux ca să vezi exact ce valori se schimbă în el și cum "
               "reacționează fiecare model.")
    index = st.number_input("Numărul fluxului din selecție", 0, max(len(pool) - 1, 0), 0)

    row = pool.iloc[[int(index)]]
    X = row[features]
    true_label = row[config.TARGET].iloc[0]
    st.caption(f"Identificator {int(row[config.ID_COL].iloc[0])} · atac de tip **{true_label}**")
    perturbed, _ = live_perturbation.apply_custom_perturbation(X, ctx, **settings["params"])

    if defended:
        combined = {}
        for name, model in models.items():
            combined[f"{name} - referință"] = model
            if name in defended:
                combined[f"{name} - reantrenat"] = defended[name]
    else:
        combined = models

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Ce s-a schimbat în flux**")
        before = X.iloc[0].astype(str)
        after = perturbed.iloc[0].astype(str)
        changed = pd.DataFrame({"înainte": before, "după": after})
        changed = changed[before != after]
        st.dataframe(changed if not changed.empty
                     else pd.DataFrame({"": ["nicio modificare"]}), width="stretch")
        if not changed.empty:
            st.caption("Doar o parte dintre ele au fost modificate direct, restul s-au "
                       "recalculat prin propagare, ca fluxul să rămână coerent.")
        if not defended:
            st.caption("Modelele reantrenate (O6) nu sunt disponibile — rulează "
                       "`python train_adversarial.py` ca să apară aici.")

    with right:
        st.markdown("**Cum răspunde fiecare model**")
        for name, model in combined.items():
            before_label = model.predict(X)[0]
            if before_label == "Normal":
                # Modelul rata deja fluxul pe varianta curata: nu exista nimic de
                # masurat, la fel ca eligibilitatea din view_result/O3. Altfel ar
                # aparea "evaziune" pentru o schimbare pe care perturbarea n-a produs-o.
                st.markdown(
                    f"**{name}** &nbsp; `{before_label}` "
                    "<span style='color:#807e7a'>(deja nedetectat pe fluxul curat, "
                    "înainte de orice modificare)</span>",
                    unsafe_allow_html=True)
                continue
            after_label = model.predict(perturbed)[0]
            outcome = live_perturbation.classify([true_label], [after_label])[0]
            color = OUTCOME_COLOR[outcome]
            st.markdown(
                f"**{name}** &nbsp; `{before_label}` → "
                f"<span style='color:{color};font-weight:600'>{after_label}</span> "
                f"<span style='color:#807e7a'>({OUTCOME_LABEL[outcome]})</span>",
                unsafe_allow_html=True)


def view_measurements():
    evasion = read_csv(config.EVASION_DIR / "evasion_summary.csv")
    if evasion is None:
        st.info("Rezultatele sistematice nu au fost încă generate. "
                "Rulează `python run_evasion.py`, apoi `python run_sensitivity.py`.")
        return

    st.markdown("Rezultatele măsurătorii sistematice, pe toată grila de transformări. "
                "Spre deosebire de fila **Explorare**, aici parametrii sunt cei "
                "predefiniți, iar cifrele sunt cele raportate în lucrare.")
    st.caption("Rata de evaziune (%) pentru fiecare tip de transformare și nivel de "
               "intensitate. Coloanele sunt grupate pe model.")
    realizable = evasion[(evasion["level"] > 0) & evasion["realizable"]]
    st.dataframe(
        realizable.pivot_table(index="type", columns=["model", "level"],
                               values="evasion_rate").round(1),
        width="stretch")

    for path in [config.FIGURES_DIR / "evasion_curves.png",
                 config.FIGURES_DIR / "sensitivity_importance.png"]:
        if path.exists():
            st.image(str(path), width="stretch")



ARM_LABEL = {"o1": "O1 - model de referință", "c1": "C1 - control (copii neperturbate)",
             "o6": "O6 - model reantrenat", "o6_loo": "O6-LOO - reantrenat, fără familia TTL"}
ARM_ORDER = ["o1", "c1", "o6", "o6_loo"]


def view_adversarial():
    """Rezultatele reantrenarii pe trafic perturbat, cu bratul de control."""
    worst = read_csv(config.ADVERSARIAL_DIR / "worst_case_global.csv")
    clean = read_csv(config.ADVERSARIAL_DIR / "clean_summary.csv")
    if worst is None or clean is None:
        st.info("Etapa de antrenare adversarială nu a fost încă rulată. "
                "Rulează `python train_adversarial.py`.")
        return

    cohorte = read_json(config.ADVERSARIAL_DIR / "cohort_sizes.json") or {}
    verdict = read_json(config.ADVERSARIAL_DIR / "o6_verdict.json") or {}

    st.markdown(
        "Fiecare model a fost antrenat de trei ori: pe setul original (**O1**), pe setul "
        "cu copii **neperturbate** (**C1**) și pe cel cu copii **perturbate** (**O6**). "
        "C1 primește exact același număr de rânduri în plus ca O6, deci singura diferență "
        "dintre ele este dacă rândurile adăugate au fost sau nu modificate.")

    # ---- tabelul principal: cost pe trafic curat vs cel mai rau caz de evaziune ----
    tabel = (worst.merge(clean, on=["model", "arm"], how="inner")
             [["model", "arm", "macro_f1_mean", "worst_case_mean", "worst_case_std",
               "worst_variants"]])
    tabel["arm"] = pd.Categorical(tabel["arm"], ARM_ORDER, ordered=True)
    tabel = tabel.sort_values(["model", "arm"])
    tabel["arm"] = tabel["arm"].map(ARM_LABEL).fillna(tabel["arm"].astype(str))
    tabel.columns = ["Model", "Grup de antrenare", "macro-F1 pe trafic nemodificat",
                     "Evaziune, cel mai rău caz (%)", "abatere între repetări",
                     "Varianta cea mai eficientă"]
    st.dataframe(tabel.round(4), width="stretch", hide_index=True)

    if cohorte:
        st.caption("Toate ratele se măsoară pe aceeași mulțime de fluxuri, comună tuturor "
                   "grupurilor, ca procentele să fie comparabile: "
                   + ",  ".join(f"{m} — {d['n']:,} fluxuri ({d['pct_of_attacks']:.2f}% "
                                "din atacuri)" for m, d in cohorte.items()))

    # ---- verdictul fata de criteriile fixate inaintea rularii ----
    verdicte = verdict.get("verdicts", {})
    if verdicte:
        st.subheader("Confruntarea cu criteriile stabilite înaintea rulării")
        st.caption("Pragurile au fost fixate și consemnate **înainte** de a cunoaște vreun "
                   "rezultat, tocmai ca să nu poată fi ajustate după aceea.")
        ales = st.selectbox("Pentru care model", sorted(verdicte), key="o6_model")
        v = verdicte[ales]
        rob, cost, sig = v["primary_robustness"], v["primary_clean_cost"], v["safety"]
        c1_, c2_, c3_ = st.columns(3)
        c1_.metric("Câștig de robustețe", f"{rob['improvement_over_c1_pp']:+.1f} pp",
                   "TRECUT" if rob["passed"] else "PICAT",
                   delta_color="normal" if rob["passed"] else "inverse",
                   help="Cât scade evaziunea față de grupul de control C1.")
        c2_.metric("Cost pe trafic nemodificat",
                   f"-{max(cost['drop_vs_c1'], cost['drop_vs_o1']):.4f}",
                   "TRECUT" if cost["passed"] else "PICAT",
                   delta_color="normal" if cost["passed"] else "inverse",
                   help="Cât macro-F1 se pierde pe traficul obișnuit.")
        c3_.metric(f"Clasa cea mai afectată: {sig['worst_class']}", f"{-sig['worst_drop']:+.3f}",
                   "TRECUT" if sig["passed"] else "PICAT",
                   delta_color="normal" if sig["passed"] else "inverse",
                   help="Criteriu de siguranță: nicio clasă nu are voie să piardă prea mult, "
                        "chiar dacă media globală arată bine.")
        st.caption(f"Praguri fixate în avans: câștig de cel puțin {rob['threshold_pp']:.0f} pp "
                   f"față de C1, scădere de macro-F1 de cel mult {cost['threshold']}, "
                   f"nicio clasă sub −{sig['threshold']}.")

    # ---- generalizarea la familia exclusa din antrenare ----
    loo = read_csv(config.ADVERSARIAL_DIR / "holdout_generalization.csv")
    if loo is not None and not loo.empty:
        st.subheader("Se transferă robustețea la transformări nevăzute?")
        st.markdown(
            "Grupul **O6-LOO** a fost antrenat **fără nicio variantă din familia TTL** și "
            "este apoi măsurat tocmai pe ea. Dacă rămâne aproape de control, modelul a "
            "memorat transformările văzute la antrenare, nu proprietatea care le face "
            "eficiente. Dacă scade mult, a învățat ceva ce se aplică și dincolo de ele.")
        loo = loo.copy()
        loo["arm"] = loo["arm"].map(ARM_LABEL).fillna(loo["arm"])
        loo.columns = ["Grup de antrenare", "Evaziune pe familia TTL (%)",
                       "minim", "maxim", "repetări"]
        st.dataframe(loo.round(2), width="stretch", hide_index=True)

    # ---- figuri ----
    for path in [config.FIGURES_DIR / "o6_cost_vs_robustness.png",
                 config.FIGURES_DIR / "o6_evasion_curves_transformer.png"]:
        if path.exists():
            st.image(str(path), width="stretch")


def main():
    st.title("Cât de ușor scapă un atac nedetectat?")
    st.markdown(
        "Trei modele de detecție a intruziunilor, antrenate pe UNSW-NB15, sunt puse în "
        "fața unui atacator care își modifică ușor traficul fără să-i schimbe scopul. "
        "Alege din stânga ce anume modifică și apasă **Rulează**.")

    models = load_models()
    defended = load_defended_models()
    data = load_data()
    settings = controls()
    st.caption(f"Se afișează rezultatul pentru: {describe(settings)}")

    result, flow, measurements, adversarial = st.tabs(
        ["Explorare", "Un singur flux", "Măsurători sistematice", "Efectul apărării"])
    with result:
        view_result(models, defended, data, settings)
    with flow:
        view_flow(models, defended, data, settings)
    with measurements:
        view_measurements()
    with adversarial:
        view_adversarial()


if __name__ == "__main__":
    main()
