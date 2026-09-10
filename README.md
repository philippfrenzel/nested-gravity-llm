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

## GitHub Codespaces / Devcontainer

Die Konfiguration in `.devcontainer/devcontainer.json` stellt Python 3.12 und die
vorhandenen Abhängigkeiten in einer virtuellen Umgebung (`.venv`) bereit. PyTorch
wird als CPU-Version installiert; eine GPU ist nicht erforderlich. Matplotlib
erzeugt PNG-Dateien ohne grafische Oberfläche.

1. Auf GitHub den gewünschten Branch auswählen (z. B. `dev`, sofern vorhanden).
   Dieser Branch muss die Devcontainer-Konfiguration enthalten.
2. **Code → Codespaces → Create codespace on …** auswählen.
3. Die automatische Installation abwarten und anschließend ein neues Terminal öffnen.
   `python` und die Python-Erweiterung verwenden dort bereits `.venv`.

**Hinweis zu github.dev:** Der mit der Taste `.` geöffnete Webeditor hat keine
Python-Laufzeit und kein ausführbares Terminal. Über **Continue Working in
Codespaces** zu einem Codespace wechseln, um Training und Tests auszuführen.
Codespaces benötigt ein verfügbares Kontingent bzw. eine entsprechende
Abrechnungseinstellung im GitHub-Konto.

Für einen kurzen CPU-Testlauf im Terminal des Codespaces:

```bash
python -m unittest discover -s tests -v
python train.py --model nested_gravity --task copy --gap-length 8 --epochs 1 --batch-size 4 --steps-per-epoch 2 --val-steps 1
python evaluate.py --checkpoint outputs/checkpoints/nested_gravity_copy_gap8_best.pt --task copy
python visualize.py --checkpoint outputs/checkpoints/nested_gravity_copy_gap8_best.pt --task copy
```

Dieser kurze Lauf prüft nur die Ausführung, nicht die Modellqualität. Checkpoints,
Metriken und Plots liegen unter `outputs/`; PNG-Dateien lassen sich direkt im
Editor öffnen. Ergebnisse vor dem Löschen eines Codespaces herunterladen.

Lokal funktioniert dieselbe Konfiguration mit Docker und der VS-Code-Erweiterung
**Dev Containers** über **Dev Containers: Reopen in Container**. Nach Änderungen
an der Container-Konfiguration oder den Abhängigkeiten **Rebuild Container**
ausführen.

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
- Statischer HTML-Report mit Vergleichstabelle und Artefakt-Galerie über `python report.py`; Ausgabe: `outputs/report.html`
- `configs/default.yaml` verwendet bewusst nur ein flaches YAML-kompatibles `key: value`-Format, damit keine zusätzliche YAML-Bibliothek nötig ist

## Tests

```bash
python -m unittest discover -s tests -v
```
