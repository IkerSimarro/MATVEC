# MATVEC — Bundled Example Envelopes

Each `.json` file in this folder is a complete `SessionSchema` payload describing one calibration envelope. They are the same artefact the CLI consumes:

```bash
python -m matvec run presets/sr_71_blackbird.json --out report.pdf
```

…and the same files the Streamlit UI loads via the **Session I/O → Load bundled example** dropdown. UI and CLI consume identical bytes.

## How to add your own example

1. Save your envelope through the UI's **Session I/O → Download session JSON** button (or hand-author one — the schema is `core/session.py:SessionSchema`).
2. Drop the file in this folder.
3. Restart Streamlit. The UI dropdown picks it up automatically — no code change required.

The `system_label` field inside the JSON becomes the dropdown label. The optional `notes` field becomes the "About this preset" caption that appears after load.

## How the bundled set is maintained

The 3 files in this folder are **mechanically generated** from `core.presets.CANONICAL_PRESETS` by the helper script:

```bash
python scripts/generate_example_presets.py
```

The script picks a curated subset of the canonical dict, embeds the per-preset description in the `notes` field, and writes one JSON per entry. Re-run after editing `CANONICAL_PRESETS`; commit the resulting diff.

If you hand-edit a bundled JSON file, `test_session.py::TestPresetParity` will fail until you either revert the edit or update `CANONICAL_PRESETS` and re-run the generator. That's intentional — the file system and the canonical dict can't silently drift.

## Why only three?

Earlier rounds shipped 10 bundled examples covering hobbyist through reference-aerospace scales. The smaller speculative envelopes (consumer drones, amateur rockets, balloon payloads, eVTOL, small commercial UAV, reentry capsules) were dropped in 2026-05 because none had a published primary source the recommendation could be checked against — they were plausible-looking numbers, not validated ones.

The three remaining presets are:

- **Two validation-anchored cases (SR-71, Concorde)** — each corresponds to a row in [`VALIDATION.md`](../VALIDATION.md) with a citation-grade primary source for the historical material choice. These are calibration targets with known-good answers.
- **One audience-relevant case (Collegiate Sounding Rocket)** — IREC / Spaceport America Cup competition class. There is no published "official material choice" for collegiate rockets the way there is for SR-71 + Goodall, so this preset is included for relevance to the university-team audience rather than as a validation target.

The lifecycle / creep feature (added in the Phase 0–6 rollout) means single-design-point analysis works correctly for boost-coast vehicles like the sounding rocket: with `flight_duration_s = 25` s and `design_lifetime_hours = 1.0` h, the worst-case envelope IS the right thing to evaluate, and the creep stage stays a silent no-op.

---

## The bundled envelopes

### `sr_71_blackbird.json` — Lockheed SR-71A
- **Mach 3.2 / 25 km / 30,600 kg / 2.5 g**
- **Flight duration:** 5,400 s (90-min sustained supersonic-cruise segment)
- **Design lifetime:** 3,000 h (typical SR-71 airframe service life)
- **Panel thickness:** 1.5 mm Ti skin (per Goodall 1995)
- Aircraft category
- Cruise Mach 3.2 above 80,000 ft (24.4 km), OEW 30,617 kg, length 32.74 m, envelope-cleared 3.0 g (preset uses 2.5 g, conservative). R_n 0.15 m approximates the inlet-spike stagnation reference; actual leading edges are sharper.
- **Validation row:** viable hit on `Ti-5Al-2.5Sn` (titanium family). Titanium creep at ~600 K × 3,000 h is comfortably within the alloy's envelope, so the lifecycle screening doesn't change the recommendation.
- **Source:** SR-71A flight manual; Goodall, *SR-71 Blackbird* (1995).

### `collegiate_sounding_rocket.json` — IREC / Spaceport America Cup 30k-ft class
- **Mach 2.0 / 9 km / 30 kg / 15 g**
- **Flight duration:** 25 s (boost ~10 s + coast ~15 s to apogee)
- **Design lifetime:** 1 h (single-flight, rebuilt between competition flights)
- **Panel thickness:** 3.0 mm wall (typical 4-inch L-motor competition tube)
- General category
- IREC mass cap is 50 kg (preset 30 kg is realistic for a 4-inch L-motor competition rocket). Apogee 9 km matches the 30,000 ft category target. Boost peak g ~15 from a Class L motor; M=2.0 at burnout is the realistic competition median (the most aggressive teams hit ~M=2.5).
- **Recommendation context:** at this envelope MATVEC computes T_wall ≈ 380 K (recovery model, no TPS unlock) and surfaces aluminum (6061-T6, 7075-T6, 2024-T3) and CFRP composites — exactly what real Spaceport America Cup teams build with.
- **Not a validation-anchored case.** Unlike SR-71 and Concorde, there is no published "official material choice" for collegiate rockets — different teams use different alloys based on machinist availability and motor-vendor compatibility. This preset is included for audience relevance, not historical calibration.
- **Source:** ESRA / IREC competition rules; typical Tripoli L2-L3 high-power rocket specifications.

### `concorde.json` — Aérospatiale-BAC Concorde
- **Mach 2.04 / 18 km / 78,000 kg / 2.0 g**
- **Flight duration:** 10,800 s (3-h trans-Atlantic supersonic cruise)
- **Design lifetime:** 25,000 h (Concorde retirement-fleet accrual; BA G-BOAB and AF F-BTSC each accumulated ~22-23k flight hours over 27 years)
- **Panel thickness:** 2.0 mm Al-Cu skin (per Aérospatiale-BAe materials reports)
- Aircraft category
- Cruise Mach 2.04 at 18 km (60,000 ft), MTOW 78,000 kg, fuselage length 61.66 m, civil load limit ~2.0 g.
- **Validation row:** marginal hit on `2219-T87` (heat-resistant Al-Cu, Saturn V tank alloy — closest in-DB analogue to Concorde's actual Al 2618 / RR58). Under the lifecycle-aware screening, **`2024-T3` is correctly rejected on creep** at 100 °C × 25,000 h — exactly the historical reason Hiduminium RR58 was developed in the 1960s.
- **Caveat:** Concorde actually flew on Aluminum 2618 (RR58), a heat-resistant Al-Cu-Mg alloy specifically developed for sustained M=2 cruise. Al 2618 is not in the database; 2219-T87 is the closest analogue. Documented limitation.
- **Source:** Owen, *Concorde and the Americans* (1997); Aérospatiale-BAe materials reports.

---

## Lifecycle / creep screening

All three bundled presets ship realistic `flight_duration_s` and `design_lifetime_hours` values. The matching engine's Larson-Miller creep stage (added in the lifecycle rollout) screens materials against rupture stress at (T_wall, lifetime), so:

- **SR-71 + Ti** continues to pass at 3,000 h × 600 K — titanium creeps slowly enough that 3,000 h is still well within margin.
- **Concorde + Al 2024-T3 fails creep** at 25,000 h × 100 °C, while Al 2219-T87 marginally passes — historically correct (this is exactly why Hiduminium RR58 was developed in the 1960s).
- **Collegiate Sounding Rocket** at 1 h lifetime stays below the creep-screening threshold; the lifecycle stage is a silent no-op and the materials list looks identical to pre-creep-feature behaviour. Aluminum + CFRP recommendations match what real teams build with.

A custom envelope with a short lifetime (≤1,000 h, including the default 1.0 h) silently skips the creep stage; the materials list looks identical to pre-lifecycle behaviour. A long lifetime triggers a banner on the Results tab and a new section in the PDF report.

## Transient heat / soak screening

In addition to creep, the matching engine runs a **1D transient heat solver** (Phase 7 rollout) when the flight is short enough that the airframe never reaches thermal equilibrium with its surroundings (`flight_duration_s < 300 s`). The solver integrates the heat equation through the panel thickness with a convective + radiative surface boundary condition and an insulated back face (worst-case internal soak), reporting the peak back-face temperature reached during the flight.

- **Collegiate Sounding Rocket** (25 s, 3 mm panel) → transient solver fires. Peak internal soak ≈ 70 °C (343 K) for the entire 25-s burn-coast, versus the static T_wall recovery temperature of 113 °C. Aluminum and CFRP recommendations are physically defensible.
- **SR-71 and Concorde** (5,400 s and 10,800 s) → transient solver does **not** fire — sustained-flight envelopes reach equilibrium with the recovery temperature, and the static T_wall check is the correct screening anyway.

Custom envelopes with `flight_duration_s` below 300 s, or with an explicit `flight_profile` time-series, trigger the transient solver. The Results tab shows a "Transient screening active" banner + new "Soak@Life" column; the PDF report grows a "Transient Heat / Soak Evaluation" section. For sustained flights, none of these are surfaced.
