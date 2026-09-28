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
- **O6.** Antrenare adversarială: contrapartida defensivă a lui O3 — dacă reantrenăm
  pe trafic de atac perturbat, se închide diferența de evaziune și cu ce cost pe
  traficul curat? Cu braț de control (C1), care izolează efectul perturbării de
  efectul volumului de date.

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
    inference.py                   # PUNCT UNIC de predictie cu modelele inghetate
                                   # (determinist; obligatoriu pentru O2/O3 — vezi mai jos)

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

run_sensitivity.py                # O4: analiza de sensibilitate (importanta <-> evaziune)
src/sensitivity/
    importance.py                  # importanta prin permutare, comparabila intre modele
    linkage.py                     # leaga ce s-a perturbat de cat de important era
    outcomes_analysis.py           # destinatiile confuziei, proximitatea de granita
    figures.py                     # graficele O4
    runner.py                      # orchestrare + artefacte
results/sensitivity/               # importante, corelatii, destinatii, atribuire

run_evasion.py                    # O3: masoara rata de evaziune pe variantele O2
src/evasion/
    outcomes.py                    # taxonomia celor trei rezultate + metrici
    measurement.py                 # parcurgerea grilei variante x modele
    aggregation.py                 # tabele agregate, intervale, intensitate minima
    figures.py                     # graficele de evaziune
    runner.py                      # orchestrare + artefacte
results/evasion/                   # rate, intervale, defalcari, rezultate per flux

run_perturbation.py               # O2: genereaza variantele perturbate (modele inghetate)
src/perturbation/
    schema.py                      # taxonomia caracteristicilor + limite de domeniu
    dependencies.py                # propagare prin rapoarte + rezolvarea ct_state_ttl
    generators.py                  # tipurile de perturbare si grila de variante
    validators.py                  # verificarile de validitate fizica si de schema
    runner.py                      # orchestrare + scrierea artefactelor
results/perturbation/              # manifest, raport de validitate, delte, eșantioane

train_transformer.py              # al treilea clasificator: FT-Transformer (PyTorch)
src/transformer/
    preprocessing.py               # RareCategoryGrouper + StandardScaler + encodare categorii
    model.py                       # FT-Transformer scris de la zero (atentie multi-head)
    predictor.py                   # invelis cu API sklearn (predict / predict_proba / classes_)
    training.py                    # antrenare, ponderi de clasa, early stopping, checkpointing
    runner.py                      # orchestrare + scrierea artefactelor

ui/                               # O5: interfata de vizualizare si testare interactiva
    app.py                         # aplicatia Streamlit (4 taburi)
    live_perturbation.py           # perturbare cu parametri arbitrari, pe primitivele O2

train_adversarial.py              # O6: antrenare adversariala prin augmentare
src/adversarial/
    augment.py                     # esantionarea variantelor, seturile O6/C1, verificarile
    weights.py                     # ponderile de clasa INGHETATE, comune tuturor bratelor
    trees.py                       # reantrenarea RF/XGBoost per brat si seed
    transformer_arm.py             # augmentare dinamica per epoca pentru retea
    evaluation.py                  # cohorta comuna + matricile de rezultate per rand
    report.py                      # tabele, figuri, verdictul fata de criteriile fixate
    sensitivity_arms.py            # importanta prin permutare pe bratele o6 si c1
    runner.py                      # orchestrare + manifest
models/adversarial/                # modelele bratelor (nu se versioneaza, ~5GB)
results/adversarial/               # manifest, metrici, cohorte, verdict
```

## Cerinte

- **Python 3.12** (dezvoltat si testat pe 3.12.10)
- **~1,5 GB spatiu** pentru date + modele (`rf_baseline.joblib` singur are 633 MB)
- **Datele UNSW-NB15** — partitia oficiala train/test, in
  `Resources/CSV Files/Training and Testing Sets/`. Se descarca de la
  [research.unsw.edu.au](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
  (alternativ [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)).
  Caile sunt configurabile in `src/config.py` (`TRAIN_PATH` / `TEST_PATH`).
- **GPU NVIDIA — optional**, doar pentru FT-Transformer: 18 min cu GPU fata de ~9 h pe CPU.
  Restul pipeline-ului nu foloseste GPU.

### Instalare

```bash
pip install -r requirements.txt
```

Pentru GPU (CUDA 12.x), instaleaza `torch` SEPARAT, **inainte** de `requirements.txt`:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **Nu folosi `torch >= 2.6` pe Windows.** Wheel-ul depaseste limita `MAX_PATH` si
> instalarea esueaza cu `WinError 206`, lasand un pachet corupt fara fisier RECORD
> (pe care `pip uninstall` nu il mai poate sterge). Versiunea fixata, 2.5.1, se
> instaleaza curat.

Verifica daca GPU-ul e vazut:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

> **Foloseste `python`, nu `py`.** Daca ai mai multe versiuni de Python instalate,
> launcher-ul `py` alege implicit versiunea cea mai noua, care poate sa nu fie cea in
> care ai instalat pachetele. Simptomul e `ModuleNotFoundError: No module named 'numpy'`
> desi tocmai ai instalat totul. Verifica cu:
>
> ```bash
> py --list                                  # ce versiuni exista
> python -c "import sys; print(sys.executable)"   # care e folosita efectiv
> ```
>
> Toate comenzile din acest README presupun `python`.

## Cum se ruleaza

Etapele depind una de alta, deci ordinea conteaza. Fiecare comanda se ruleaza din
radacina proiectului.

| # | Comanda | Durata | Necesita | Produce |
|---|---|---|---|---|
| 1 | `python main.py` | ~15 min | datele | `models/*.joblib`, metrici, EDA |
| 2 | `python train_transformer.py` | ~18 min GPU | datele | `models/transformer_baseline.pt` |
| 3 | `python run_perturbation.py` | ~2 min | pasul 1 | `results/perturbation/` |
| 4 | `python run_evasion.py` | ~5 min | pasii 1-3 | `results/evasion/` |
| 5 | `python run_sensitivity.py` | ~8 min | pasii 1-4 | `results/sensitivity/` |
| 6 | `python -m streamlit run ui/app.py` | interactiv | pasii 1-5 | interfata |
| 7 | `python train_adversarial.py` | ~9 h | pasii 1-4 | `results/adversarial/` |

Pasul 2 e optional daca vrei doar arborii; pasii 4-6 il vor sari automat.
Pasii 3-6 **nu reantreneaza nimic** — incarca modelele inghetate. Pasul 7 antreneaza
30 de modele noi, dar in `models/adversarial/`: nu atinge niciun artefact O1-O5.

### Detalii utile

**Antrenarea Transformer-ului e reluabila.** Checkpoint-uri la fiecare 10 epoci si la
fiecare imbunatatire, deci o intrerupere costa cel mult cateva epoci:

```bash
python train_transformer.py               # antreneaza sau continua de unde a ramas
python train_transformer.py --fresh       # ignora checkpoint-ul, reia de la zero
python train_transformer.py --eval-only   # doar artefactele, din modelul deja salvat
python train_transformer.py --max-epochs 2  # rulare scurta de proba
```

## Interfata interactiva (O5)

```bash
python -m streamlit run ui/app.py
```

Se foloseste `python -m streamlit`, nu `streamlit` direct: pip instaleaza `streamlit.exe`
intr-un director `Scripts` care de obicei **nu e in PATH** pe Windows, iar comanda scurta
esueaza cu `'streamlit' is not recognized as an internal or external command`. Forma cu
`-m` ocoleste complet problema si foloseste exact interpretorul in care ai instalat
pachetele.

Se deschide pe `http://localhost:8501`. **Prima incarcare dureaza ~23 s** (cele trei
modele, dintre care unul de 633 MB); dupa aceea raman in cache si perturbarile se
aplica instantaneu.

| Tab | Ce face |
|---|---|
| **Testare pe grup** | Alegi o clasa de atac si misti controalele; rata de evaziune a celor trei modele se recalculeaza pe loc |
| **Flux individual** | Un singur flux: ce caracteristici s-au schimbat si cum se muta decizia si probabilitatile fiecarui model |
| **Rezultate masurate** | Tabelele si figurile din O3/O4 |
| **De ce cedeaza modelele** | Concentrarea importantei si corelatia din O4 |

Controalele din bara laterala accepta **valori arbitrare**, nu doar cele 28 de variante
din grila: orice `sttl` intre 16 si 255, orice procent de padding, orice factor de
temporizare, reducerea contoarelor de conexiuni, plus politica `ct_state_ttl`.

**Demo sugerat pentru sustinere:** lasa clasa pe „(toate)", muta `sttl` de la 254 la 31
si urmareste cele trei metrici. XGBoost trece de la 0% la peste 50%. Apoi pune `sttl=62`
si arata ca evaziunea **scade** — pentru ca 62 e o valoare asociata atacurilor, nu
traficului normal.

**Interfata nu reimplementeaza perturbarea.** `ui/live_perturbation.py` compune exact
aceleasi primitive validate din O2 (`dependencies.propagate`,
`dependencies.resolve_ct_state_ttl`, `schema.restore_dtypes`) si trece fiecare varianta
prin `validators.validate`. Doua consecinte:

- date aceleasi valori ca un nivel din grila, interfata reproduce **exact** varianta
  corespunzatoare din O2 (verificat pe 7 variante, identice bit cu bit);
- o combinatie arbitrara care ar incalca o constrangere fizica e semnalata ca invalida,
  deci nu se poate afisa o „evaziune" obtinuta cu un flux imposibil de produs in realitate.

### Daca ceva nu merge

| Simptom | Cauza / rezolvare |
|---|---|
| `FileNotFoundError` pe CSV-uri | Datele UNSW-NB15 lipsesc; vezi *Cerinte* |
| `lipseste feature_deltas.csv` | Ruleaza `python run_perturbation.py` intai |
| Interfata spune ca lipsesc rezultatele O3/O4 | Ruleaza `run_evasion.py`, apoi `run_sensitivity.py` |
| `modelul Transformer lipseste` | Ruleaza `python train_transformer.py`, sau ignora — restul merge fara el |
| `WinError 206` la instalarea torch | Foloseste `torch==2.5.1`, nu o versiune mai noua |
| Antrenarea Transformer dureaza ore | Rulezi pe CPU; instaleaza wheel-ul CUDA |
| `'streamlit' is not recognized...` | Foloseste `python -m streamlit run ui/app.py` |
| `ModuleNotFoundError` desi ai instalat pachetele | Rulezi cu alt interpretor (tipic `py` → 3.13 in loc de 3.12). Foloseste `python` |

## Modelul de amenintare (O2)

Trebuie enuntat explicit in lucrare, pentru ca fixeaza ce inseamna "evaziune" aici.

| Dimensiune | Alegere | Motiv |
|---|---|---|
| Acces la model | **Black-box, fara interogari** | Un atacator real nu poate interoga IDS-ul aparatorului si nu ii vede gradientii. |
| Strategie | **Neadaptiva, grila structurata** | Intrebarea de cercetare e *ce tipuri de modificare conteaza si cat*, nu *care e perturbarea adversariala minima*. |
| Capabilitate | **Doar partea sursa** | Atacatorul isi controleaza propriile pachete, nu si raspunsurile victimei. |
| Constrangere | **Atacul trebuie sa ramana functional** | Perturbarile au directie impusa si limite fizice. |

Deliberat **nu** este un atac prin optimizare (FGSM/PGD/ZOO). Acelea cer acces la gradient
sau la interogari repetate, pe care atacatorul din acest model de amenintare nu le are, si
produc vectori de caracteristici deseori imposibil de realizat ca trafic real. FT-Transformer-ul
fiind diferentiabil, atacurile pe gradient raman o extindere viitoare naturala — exista un
hook documentat in `src/transformer/predictor.py::logits_with_grad`, neimplementat intentionat,
ca toate cele trei modele sa fie comparate pe aceleasi perturbari realizabile.

**Directia e impusa, nu doar intentionata:** padding-ul poate doar sa creasca numarul de octeti
(nu poti "retrage" date deja trimise), temporizarea poate doar sa creasca durata (poti oricand
trimite mai lent, niciodata mai repede decat permite reteaua), iar contoarele de conexiuni pot
doar sa scada (atacatorul isi poate incetini oricand scanarea). Limitele se aplica *relativ la
randul original*, fiindca captura reala incalca deja unele dintre ele: 61 de fluxuri de atac au
`sttl` in {0,1} si o inregistrare are `smean=1504 > MTU`.

### Constatarea principala: `sttl` este un artefact al setului de date

| Trafic | `sttl` dominant |
|---|---|
| Normal | **31** (70.5%), 254 (20.1%), 62 (3.9%) |
| Fiecare clasa de atac | **254** (60–100%), 62 secundar |

Traficul de atac sta aproape universal la `sttl=254` pentru ca generatorul IXIA PerfectStorm
s-a aflat la o distanta fixa in hop-uri fata de senzor. `sttl` este caracteristica cea mai
corelata cu eticheta (r = 0.69). Un atacator o schimba cu un singur apel `setsockopt`, cu cost
zero si fara niciun efect asupra functionarii atacului. Daca detectia se prabuseste la aceasta
schimbare, modelul citea amprenta generatorului, nu atacul.

Traficul normal contine si el 11.230 de randuri la `sttl=254` (in setul de train; 26.279 in
train+test), deci modelul a invatat o corelatie, nu o regula. Aceasta e observatia care face
din perturbarea TTL piesa centrala a lucrarii.

### `ct_state_ttl`: de ce se raporteaza doua margini, nu o valoare

`ct_state_ttl` **nu** este o functie pe intervale de TTL, in ciuda descrierii din documentatia
setului de date. Contraexemple, verificate direct pe date: `sttl=62 -> 2` dar `sttl=63 -> 0`;
`sttl=254 -> 2` dar `sttl=252 -> 0`. Valori TTL adiacente dau iesiri diferite, deci nicio
partitionare pe intervale nu poate reproduce maparea. Se comporta exact ca numaratorul pe
fereastra glisanta de 100 de conexiuni pe care il sugereaza numele, si **nu poate fi recalculat
din inregistrari de flux izolate** — informatia necesara (ordinea si timpii tuturor conexiunilor
din captura) nu exista in date.

Asta conteaza: `ct_state_ttl` este a doua caracteristica dupa corelatia cu eticheta (r = 0.58),
iar `ct_state_ttl = 0` este semnatura traficului normal, in timp ce valorile nenule apartin
grupului de atac (ex. `(FIN, 31, 29) -> 0` apare in 40.524 de randuri, 0% atacuri, fata de
`(INT, 254, 0) -> 2` in 111.924 de randuri, 94% atacuri).

Cand `sttl` devine 31, **34,3% dintre randurile eligibile** (15.492, majoritatea `FIN` cu
`dttl=252`) ajung pe combinatii nevazute niciodata in date. Alegerea variantei de rezerva ar fi
determinat in tacere rezultatul principal, asa ca O2 emite **ambele margini**, ca variante
etichetate separat:

| Politica | Comportament | Directia erorii |
|---|---|---|
| `hold` | `ct_state_ttl` ramane neschimbat | conservator — **subestimeaza** evaziunea (lasa intact un semnal puternic de atac) |
| `mimic` | lookup exact -> modul traficului normal la acel `sttl` -> constanta 0 | optimist — **supraestimeaza** evaziunea (presupune ca atacatorul se camufleaza complet) |

`mimic` este empiric, nu "pune 0": la `sttl=62` lasa distributia practic neschimbata, pentru ca
fluxurile reale cu `sttl=62` din aceasta captura *sunt* dominate de atacuri. La `sttl=31` si 64
duce totul la 0.

Capabilitatea reala a atacatorului se afla intre cele doua. **O3 trebuie sa raporteze un
interval** ("evaziunea prin TTL este intre X% si Y%"), nu o singura cifra: distanta dintre
margini e ea insasi un rezultat, fiindca masoara cat din robustetea modelului depinde de o
caracteristica pe care nici atacatorul nu o controleaza direct, nici aparatorul nu o poate
recalcula. Dovezile complete si tabelul de lookup sunt in `results/perturbation/ct_state_ttl_rule.json`.

Varianta `ttl_both` perturba si `dttl`, care apartine victimei si e imposibil de atins de
atacator. Exista **doar** ca referinta de margine superioara, ca sa cuantifice cat din
dependenta modelului de TTL sta pe ceva inaccesibil, si nu trebuie raportata ca evaziune
realizabila.

## Rezultatele evaziunii (O3)

`python run_evasion.py` (~5 min) masoara, pentru fiecare (model, tip, nivel), ce se
intampla cu cele 45.332 de fluxuri de atac. Totul se raporteaza la fluxurile
**eligibile** — cele pe care modelul le-a semnalat ca malitioase pe traficul curat
(RF 45.116, XGBoost 44.258, Transformer 45.304). Un flux pe care modelul nu l-a prins
niciodata nu poate "evada"; includerea lui ar transforma esecuri preexistente ale
modelului in succese ale atacatorului.

**Invariant verificat:** la nivelul 0 (varianta-identitate) rata de evaziune este
exact 0,00% pentru toate cele trei modele — prin definitia setului eligibil.

### Rata de evaziune, cifre principale (% din fluxurile eligibile)

| Model | TTL -> sttl=31 | TTL + padding + timing (nivel max) | padding singur | timing singur | rata conexiuni |
|---|---|---|---|---|---|
| Random Forest | 4,4 – 18,4% | 31,3 – 75,9% | 0,1% | 0,7% | 0,3% |
| XGBoost | **50,2 – 51,1%** | **96,4 – 96,9%** | 1,2% | 1,1% | 0,3% |
| FT-Transformer | 4,0 – 43,2% | 17,4 – 71,0% | 0,1% | 0,0% | 0,0% |

Intervalele sunt marginile `hold`..`mimic` explicate mai sus. Cifrele nu includ
`ttl_both`, varianta nerealizabila.

**TTL este singurul vector care conteaza.** Padding-ul, temporizarea si reducerea
ratei de conexiuni, aplicate singure, lasa evaziunea sub ~1,2% la orice nivel de
intensitate — modelele sunt practic insensibile la ele. O singura modificare de TTL,
care costa un apel `setsockopt` si nu afecteaza deloc functionarea atacului, duce
XGBoost de la 0% la peste 50%. Combinat cu padding si temporizare, ajunge la 96%.
Aceasta este confirmarea empirica directa a constatarii ca `sttl` era un artefact
al generatorului, nu un semnal despre atac.

**Nivelurile TTL nu sunt o scara de intensitate.** Sunt valori tinta — sttl 64, 62,
31 — iar nivelul 2 (`sttl=62`) este o valoare asociata *traficului de atac*, nu celui
normal. De aceea evaziunea scade la nivelul 2 in loc sa creasca: la `sttl=62`
politica `mimic` nu curata deloc semnatura `ct_state_ttl` (0% dintre fluxuri ajung
la valoarea benigna 0, fata de 100% la nivelurile 1 si 3), deci cele doua margini
coincid si intervalul se inchide. Este un rezultat interpretabil, nu o anomalie.

**Cine e mai robust depinde de intrebare.** XGBoost are cele mai bune metrici pe
traficul curat (macro F1 0,51) dar cedeaza cel mai usor la perturbarea TTL; Random
Forest are marginea conservatoare cea mai mica. Robustetea si acuratetea nu merg in
aceeasi directie — exact ipoteza de la care porneste lucrarea.

Grafice: `results/figures/evasion_curves.png` (rata pe niveluri, cu benzile
hold-mimic), `evasion_outcomes_<model>.png` (compozitia celor trei rezultate) si
`evasion_per_class_<model>.png` (defalcare pe clase de atac).

## Analiza de sensibilitate (O4)

`python run_sensitivity.py` explica DE CE apar ratele din O3. Importantele native nu
se pot compara intre modele (RF are impuritate, XGBoost "gain", Transformer atentie),
deci se calculeaza importanta prin permutare — model-agnostica, aceeasi definitie
pentru toate trei. Metrica e scaderea ratei de detectie pe fluxurile de atac, adica
exact marimea complementara evaziunii, deci cele doua sunt comensurabile.

| Model | Caracteristica dominanta | Cota ei | Din importanta totala, cat poate atinge atacatorul |
|---|---|---|---|
| Random Forest | `sttl` | 19,8% | 53,5% |
| XGBoost | `sttl` | **54,6%** | **84,0%** |
| FT-Transformer | `dttl` | 57,0% | 26,0% |

Tabelul explica rezultatele din O3. XGBoost concentreaza peste jumatate din capacitatea
de detectie intr-o singura caracteristica pe care atacatorul o schimba cu un apel
`setsockopt` — de aici 50% evaziune. Random Forest o distribuie, deci cedeaza mai putin.
Transformer-ul se sprijina cel mai mult pe `dttl`, TTL-ul *victimei*, pe care atacatorul
nu il poate atinge — de aceea evaziunea lui realizabila e marginita, si de aceea varianta
nerealizabila `ttl_both` (singura care atinge `dttl`) ajunge la 96,8% pentru el.

**Corelatia ceruta de obiectiv:** perturbarile care ating mai multa importanta produc
mai multa evaziune — Spearman rho = 0,95 (RF), 0,88 (XGBoost), 0,83 (Transformer),
toate cu p < 0,0001 peste cele 26 de variante realizabile.

**Rezultatul (c) — fiecare model are o clasa "atractor".** Fluxurile care nu evadeaza
dar isi schimba eticheta ajung covarsitor intr-o singura clasa: `Backdoor` la Random
Forest, `Analysis` la XGBoost, `DoS` la Transformer. Nu e intamplator: `Analysis` si
`Backdoor` sunt exact clasele cu cel mai slab F1 de baza (0,02-0,11). Atractorul e
regiunea in care modelul nu are un angajament ferm.

**Atributia pe o singura caracteristica subestimeaza vulnerabilitatea.** Pe clasa
`Generic`, substituirea unei singure caracteristici (chiar si a celei dominante) misca
detectia cu cel mult cateva puncte, in timp ce perturbarea coerenta din O2 — `sttl` si
`ct_state_ttl` schimbate impreuna, consistent — o prabuseste. Nicio analiza care schimba
o caracteristica pe rand nu ar fi detectat asta. Justifica retroactiv decizia centrala
din O2: propagarea consistenta a caracteristicilor derivate, nu perturbarea lor izolata.

Detalii complete in `docs/o4_sensitivity_analysis.md` si `docs/pipeline_complet_O1_O5.md`
(locale — `docs/` nu e versionat).

## Predictie determinista (obligatoriu pentru O3)

Toate predictiile cu modelele inghetate trebuie sa treaca prin `src/inference.py`,
niciodata prin `joblib.load` direct:

```python
from src import inference
models = inference.load_frozen_models(include_transformer=True)
preds = inference.predict(models, X)   # dict: nume model -> etichete string
```

Motivul: Random Forest-ul antrenat are 44 de randuri pe muchie de cutit in setul de
atac (43 la egalitate exacta, unul cu diferenta de 1 ULP — `sample_id=49676`, unde
Exploits=0.392499999999999905 si Reconnaissance=0.392500000000000016, adica 78,5
voturi din 200 in ambele cazuri). Cu `n_jobs=-1`, ordinea de insumare a voturilor
variaza intre fire de executie si `argmax` oscileaza: cinci apeluri identice au dat
0, 1, 0, 1, 0 nepotriviri fata de baseline.

Daca verificarea si masurarea ar folosi cai diferite, un astfel de flux ar fi numarat
drept schimbare de clasa provocata de perturbare, cand de fapt e doar planificarea
firelor. `src/inference.py` forteaza `n_jobs=1` la incarcare — nu modifica ponderile
si nu schimba niciun rezultat salvat (reproduce exact `baseline_predictions.csv`),
la un cost de 1.2s in loc de 0.4s pentru 45.332 de randuri. Modulul expune si aceeasi
interfata pentru toate cele trei modele (`.predict`, `.predict_proba`, `.classes_`,
cu aceeasi ordine a claselor), deci O3 nu are nevoie de ramificatii per model.

## Rezultate (test oficial, aceleasi features si acelasi split la toate modelele)

| Model | Accuracy | Balanced acc. | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Random Forest | 0.6913 | 0.5944 | 0.5010 | 0.7416 |
| XGBoost | 0.7659 | 0.5443 | **0.5098** | **0.7816** |
| FT-Transformer | 0.6594 | **0.6288** | 0.4354 | 0.7168 |

Selectia checkpoint-ului se face pe media mobila a macro-F1-ului de validare pe 3
epoci, nu pe valoarea unei singure epoci: prima rulare salvase o epoca aflata la 2,3
abateri standard peste media locala, adica zgomot. Corectia nu a imbunatatit macro-F1
pe test (0,4538 -> 0,4354, diferenta sub zgomotul de 0,030 dintre epoci), dar a dat un
model mai robust (evaziune TTL 4,0-43,2% fata de 10,7-72,1%) si o procedura de selectie
defensabila.

Arborii raman rezultatul principal: pe date tabelare, ansamblurile de arbori sunt
frecvent competitive sau mai bune decat retelele (Grinsztajn et al., 2022), iar
masuratoarea de aici confirma asta. Transformer-ul obtine insa cea mai buna
*balanced accuracy* si cel mai bun F1 pe DoS si Backdoor, deci nu e uniform mai slab.

Diferenta de preprocesare e deliberata si trebuie mentionata in lucrare: arborii
primesc numericele nescalate (folosesc doar ordinea valorilor), reteaua le primeste
standardizate (medie 0, varianta 1), cu scalerul ajustat exclusiv pe train. Features,
randuri, split si etichete sunt identice la toate trei modelele.

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
- [x] Modul de perturbare (O2) — 28 de variante validate, `python run_perturbation.py`;
      varianta-identitate reproduce exact baseline-ul O1
- [x] Al treilea clasificator: FT-Transformer (PyTorch) — `python train_transformer.py`;
      aceeasi interfata ca modelele sklearn, deci O3 nu are nevoie de ramificatii per model
- [x] Cale de predictie determinista (`src/inference.py`) — elimina variatia intre
      rulari data de egalitatile din Random Forest, altfel O3 ar numara evaziuni fantoma
- [x] Masurare rata de evaziune (O3) — `python run_evasion.py`; 28 de variante x 3 modele,
      raportata ca interval hold..mimic; varianta-identitate da exact 0% evaziune
- [x] Analiza de sensibilitate (O4) — `python run_sensitivity.py`; importanta prin
      permutare comparabila intre modele, corelata cu evaziunea (Spearman rho 0,83-0,95)
- [x] Interfata de vizualizare si testare interactiva (O5) — `python -m streamlit run ui/app.py`;
      perturbari live cu valori arbitrare, validate cu aceleasi verificari ca in O2
- [x] Antrenare adversariala (O6) — `python train_adversarial.py`; 30 de modele in patru
      grupuri (referinta, control, aparat, aparat fara familia TTL), ponderi de clasa
      inghetate, multime comuna de fluxuri si criterii fixate inaintea rularii.
      Evaziunea in cel mai rau caz scade sub 0,5% la toate trei modelele, dar
      generalizarea la o familie nevazuta se produce doar la unul din trei
- [x] Importanta prin permutare pe modelele aparate (`src/adversarial/sensitivity_arms.py`)
      — arata *prin ce* s-a obtinut robustetea: fractiunea accesibila scade de la
      38-76% (control) la 7-19% (aparat), iar `sttl` cade de pe primele pozitii pe
      locurile 38-41 din 42
