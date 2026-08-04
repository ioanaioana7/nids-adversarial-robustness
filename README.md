# Evaluarea robusteței unui sistem de detecție a intruziunilor bazat pe ML în fața modificărilor adversariale

Lucrare de licență — Facultatea de Automatică și Calculatoare, UTCN

## Scop

Sistemele de detecție a intruziunilor (NIDS) bazate pe învățare automată ating rate
de detecție ridicate atunci când sunt evaluate pe trafic din aceeași distribuție cu
cea de antrenare. Această performanță nu garantează însă robustețea în fața traficului
malițios **modificat deliberat**. Un atacator poate ajusta caracteristici ale traficului
(la nivel de flux sau de pachet) astfel încât atacul să își păstreze funcționalitatea,
dar să nu mai corespundă tiparelor învățate de model.

Scopul proiectului este **evaluarea empirică a rezistenței** unui clasificator de trafic
la variante ușor modificate ale traficului malițios — cât de ușor poate fi indus în
eroare, prin ce tipuri de modificări, și cu ce consecințe asupra clasificării.

## Ideea

1. Se antrenează un **clasificator multi-clasă** care distinge traficul normal de mai
   multe tipuri de atac (dataset: UNSW-NB15; modele: Random Forest și XGBoost).
2. Se generează **variante perturbate** ale traficului malițios prin modificări
   controlate (ex. TTL, padding, temporizare), pe niveluri de intensitate diferite,
   respectând constrângerea ca atacul să rămână valid și funcțional.
3. Se măsoară **rata de evaziune** — ce procent din variante trec nedetectate — și se
   analizează **ce tipuri de modificări** influențează cel mai mult decizia modelului.
4. Se oferă o **interfață simplă** pentru vizualizarea rezultatelor și testarea
   interactivă a diferitelor niveluri de modificare.

Într-un cadru multi-clasă, o variantă perturbată poate: (a) rămâne clasificată corect,
(b) deveni „trafic normal” (evaziune propriu-zisă), sau (c) fi confundată cu un alt tip
de atac. Distincția (c) oferă informații despre fragilitatea granițelor de decizie dintre
clasele de atac.

## Ce se urmărește (obiective)

- **O1.** Antrenarea unui clasificator multi-clasă de referință, evaluat cu metrici
  adecvate cadrului dezechilibrat (precizie, recall, F1 per clasă; matrice de confuzie).
- **O2.** Un modul de generare a variantelor perturbate (tipuri × niveluri de intensitate).
- **O3.** Măsurarea ratei de evaziune, defalcată pe tip de modificare și nivel.
- **O4.** Analiza sensibilității: corelarea ratei de evaziune cu importanța caracteristicilor
  afectate; distincția între cele trei rezultate posibile ale perturbării.
- **O5.** Interfață de vizualizare și testare interactivă.

## De ce UNSW-NB15 și modele bazate pe arbori

- **Dataset:** UNSW-NB15 este modern, are 9 familii de atac + trafic normal, oferă atât
  PCAP brut (pentru perturbări la nivel de pachet) cât și features de flux, include
  coloanele TTL (`sttl`/`dttl`) vizate de perturbări, și vine cu split train/test
  predefinit. Este recomandat în literatură pentru evaluări generale.
- **Model:** ansamblurile de arbori (Random Forest, XGBoost) obțin constant cele mai
  bune rezultate pe date tabulare de trafic, gestionează dezechilibrul, sunt interpretabile
  (feature importance, SHAP — necesare pentru analiza de sensibilitate) și au proprietăți
  de robustețe adversarială diferite de cele ale rețelelor neuronale.

## Structura proiectului

```
main.py                          # orchestreaza: incarcare -> EDA -> antrenare -> evaluare
src/
    config.py                    # configurare centralizata (cai, coloane, hiperparametri, seed)
    utils.py                     # seed global, logging, versiuni librarii, salvare JSON
    data_loader.py                # incarcarea datelor brute
    preprocessing.py              # RareCategoryGrouper + ColumnTransformer
    eda.py                        # statistici descriptive + figuri
    models.py                     # pipeline-urile Random Forest / XGBoost
    evaluation.py                 # metrici, matrice de confuzie, feature importance,
                                   # export predictii/probabilitati, validare model salvat
    experiment.py                  # metadata experiment, mapare etichete, nume caracteristici
    shap_analysis.py               # analiza SHAP (optionala, pregateste O4)

Resources/CSV Files/              # datele UNSW-NB15 (train/test si fisierele brute)
models/                            # modelele antrenate (.joblib)
results/
    figures/                       # grafice EDA, matrici de confuzie, SHAP
    metrics/                       # metrici si matrici de confuzie (CSV)
    feature_importance/            # importanta caracteristicilor (RF, SHAP)
    eda/                           # statistici EDA (CSV)
    predictions/                   # predictii/probabilitati baseline — referinta pentru O3
    training.log
    experiment_metadata.json
    dataset_summary.json
    label_mapping.json
    feature_names.csv

perturbation/                     # modul de generare a variantelor perturbate (O2)  [in lucru]
ui/                                # interfata de vizualizare (O5)  [in lucru]
```

## Cum se rulează (baseline)

```bash
pip install pandas scikit-learn matplotlib joblib xgboost shap
python main.py
```

Caile catre CSV-urile UNSW-NB15 sunt configurate in `src/config.py`
(`TRAIN_PATH`/`TEST_PATH`, implicit `Resources/CSV Files/Training and Testing Sets/`).

Rezultate generate: modelele in `models/`, toate rapoartele/graficele/metadatele in
`results/` (vezi structura de mai sus); consola si `results/training.log` primesc
acelasi jurnal al rularii.

## Stadiu

- [x] Baseline multi-clasa (RF + XGBoost) — pasii 1-3
- [x] Refactorizare in modul `src/`, configurare centralizata, reproducibilitate
      (seed unic, metadata de experiment, validarea modelelor salvate, export
      predictii/probabilitati/SHAP) — pregatire pentru O2
- [ ] Modul de perturbare (O2)
- [ ] Masurare rata de evaziune (O3)
- [ ] Analiza de sensibilitate (O4)
- [ ] Interfata (O5)
