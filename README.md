# nested-gravity-llm

Kleiner reproduzierbarer PyTorch-POC für ein kausales Sequenzmodell mit lokalen gravitationsartigen Interaktionen und verschachtelten latenten Zentren sowie GRU- und Transformer-Baselines.

## Projektstruktur

```text
nested_gravity_poc/
├── README.md
├── requirements.txt
├── config.py
├── train.py
├── evaluate.py
├── visualize.py
├── models/
├── data/
├── tests/
├── outputs/
└── configs/
```

## Ausführung in fünf Befehlen

```bash
python -m pip install -r requirements.txt
python train.py --model nested_gravity --task copy --gap-length 64 --epochs 20
python train.py --model gru --task copy --gap-length 64 --epochs 20
python evaluate.py --checkpoint outputs/checkpoints/nested_gravity_copy_gap64_best.pt --task copy
python visualize.py --checkpoint outputs/checkpoints/nested_gravity_copy_gap64_best.pt --task copy
```

## Enthaltene Aufgaben

- Copy Task mit auswertbarer Zielmaske für den zweiten Auftakt der Erinnerungssequenz
- Associative Recall mit zufälligen Schlüssel-Wert-Paaren
- Hierarchical Brackets mit mehreren Klammertypen
- Character-Level-Textdaten mit eingebautem deutschem Fallback-Korpus oder `data/text.txt`

## Reproduzierbarkeit

- Standard-Seed: `42`
- CPU-kompatible PyTorch-Implementierung
- Checkpoints unter `outputs/checkpoints/`
- CSV/JSON-Metriken unter `outputs/metrics/`
- PNG-Plots unter `outputs/plots/`

## Tests

```bash
python -m unittest discover -s tests -v
```
