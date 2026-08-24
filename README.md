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

ui/                                # interfata de vizualizare (O5)  [neinceput]
```

## Cum se rulează (baseline)

```bash
pip install pandas scikit-learn matplotlib joblib xgboost shap
python main.py
```

Pentru modulul de perturbare (O2 — foloseste modelele inghetate, nu reantreneaza):

```bash
python run_perturbation.py
```

Pentru al treilea clasificator (FT-Transformer, PyTorch):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # sau /cpu
python train_transformer.py
```

Antrenarea salveaza checkpoint-uri periodic, deci o intrerupere nu pierde progresul:
relansarea aceleiasi comenzi continua de unde a ramas. `--fresh` reia de la zero,
`--eval-only` regenereaza artefactele din modelul deja salvat.

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
| FT-Transformer | 0.6654 | **0.6038** | 0.4538 | 0.7209 |

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
- [ ] Masurare rata de evaziune (O3)
- [ ] Analiza de sensibilitate (O4)
- [ ] Interfata (O5)
