# MATVEC — Aerospace Materials Feasibility Analysis

**MATVEC** is an open-source screening tool that tells you which structural materials are physically compatible with a given flight envelope — in under a second.

You define the vehicle (Mach number, altitude, mass, g-load, mission duration) and MATVEC runs an integrated physics analysis — aerodynamic heating, structural loads, lifecycle creep — then ranks all ~60 candidate materials by how much thermal and structural margin they carry. Results export as a professional PDF report with equations, tables, and per-zone recommendations.

> **Scope:** MATVEC is a concept-phase screening tool, not a detailed design tool. It tells you which material families are viable and which are ruled out. It does not replace FEM analysis, fatigue testing, or a materials engineer.

---

## Live demo

https://matvec.streamlit.app/

---

## Bundled presets

| Preset | Mach | Altitude | Design lifetime |
|---|---|---|---|
| SR-71 Blackbird | 3.2 | 25 km | 3,000 h |
| Concorde | 2.04 | 18 km | 25,000 h |
| Collegiate Sounding Rocket | 2.0 | 9 km | 1 h |

Load any of these from the **Load Bundled Example** dropdown — no typing required.

---

## What MATVEC computes

**Thermal analysis** — Aerodynamic heating via the recovery temperature model (Mach < 5) or the Sutton–Graves convective model with Tauber–Sutton radiation (Mach ≥ 5). ISA atmosphere, 0–86 km.

**Structural analysis** — Combined inertial, dynamic-pressure, and thermal stress under MIL-HDBK-5 safety factors.

**Lifecycle / creep evaluation** — Larson–Miller parameter screening for long-duration vehicles (≥ 1,000 h design lifetime). Materials that pass the instantaneous check but creep to failure over thousands of hours are flagged separately.

**Transient heat soak** — 1D finite-difference solver through the panel thickness for short-duration boost-coast trajectories (flight duration < 300 s). The back-face temperature, not the surface flash temperature, drives the screening.

**Electromagnetic signature** — Peak emission wavelength and radiated power via Wien's law and Stefan–Boltzmann.

**Multi-objective trade-off** — Pareto front across weight, thermal margin, structural margin, and material availability.

**Materials property space** — k-nearest-neighbor surrogate in 7-dimensional property space to surface candidates the margin-based ranker might miss.

---

## Getting started

### Requirements

- Python 3.10+
- pdflatex (optional — required for PDF report export; `.tex` source always available)

### Install

```bash
git clone https://github.com/ikersimarro/matvec.git
cd matvec
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run tests

```bash
pytest
```

492 tests, ~90 seconds on a modern laptop.

---

## Project structure

```
app.py                  Streamlit UI
physics_engine.py       ISA atmosphere, thermal & structural physics
matching_engine.py      3-stage filter + rank pipeline
materials_db.py         ~60-entry materials database with creep data
latex_export.py         LaTeX/PDF report generator
core/
  api.py                Single-function entry point: run_session()
  session.py            SessionSchema dataclass + JSON round-trip
  presets.py            Canonical preset definitions
  pareto.py             Multi-objective Pareto front
  surrogate.py          k-NN property-space surrogate
  transient_heat.py     1D transient heat solver
  sensitivity.py        Uncertainty sweep (±Δ per input)
  creep.py              Larson–Miller creep evaluation
  component_zones.py    Per-zone thermal/structural multipliers
presets/                Bundled example JSON files
scripts/                Utility scripts (generate presets, run validation)
VALIDATION.md           Physics calibration and validation cases
```

---

## PDF report export

The **Export Report (PDF)** button requires `pdflatex`. If it is not installed:

- On macOS: `brew install --cask mactex-no-gui`
- On Ubuntu/Debian: `sudo apt install texlive-latex-extra texlive-fonts-recommended`
- On Windows: install [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

If pdflatex is absent the button falls back to a `.tex` source download you can compile locally or via [Overleaf](https://overleaf.com).

---

## Limitations

- **~60 materials** in the database. Granta MI has thousands. MATVEC covers common aerospace structural materials (aluminum alloys, titanium alloys, nickel superalloys, CMCs, CFRPs) but not every alloy.
- **Simplified thermal model.** No ablation, no active cooling, no TPS modelling. Ablative heat shields and cooled structures would pass in practice but may not appear in the viable list.
- **1D structural model.** Inertial + dynamic-pressure + thermal stress combined under MIL-HDBK-5. Not a substitute for FEM.
- **Screening, not sizing.** MATVEC tells you Ti-6Al-4V is viable at Mach 3.2 / 25 km. It does not size the wall thickness, compute fatigue life, or select fasteners.

---

## License

[Business Source License 1.1](LICENSE) — free for non-commercial use (personal,
academic, student projects). Commercial deployments require a license.
Converts to MIT automatically on 2035-05-25.
