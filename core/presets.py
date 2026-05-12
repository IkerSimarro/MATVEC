"""
MATVEC — Canonical preset SessionSchemas for CLI / CI consumption.

These entries mirror ``app.py`` 's ``EXAMPLES`` dict but expressed as
full SessionSchema records — the form the headless CLI and regression
harness consume.

Why two copies?
  * ``app.py`` 's EXAMPLES is wired into the Streamlit preset-loader
    staging mechanism (``_pending_vehicle_category``, widget keys)
    using its historical short-form field names. Refactoring it to
    use SessionSchema directly would touch every widget bind.
  * The canonical form here lets the CLI, tests, and future
    Metric-Standard split import preset envelopes WITHOUT importing
    Streamlit.

If you add a preset, add it **both** places. The test
``test_presets.py::test_preset_parity`` enforces that the two sets
stay aligned — if they drift, CI tells you exactly which key is
missing and from where.
"""

from core.session import SessionSchema


# Insertion order matches app.py's EXAMPLES (after the placeholder).
# Key = short human name; system_label = canonical title used in the
# PDF header and the slugified filename.
CANONICAL_PRESETS: dict[str, SessionSchema] = {
    "SR-71 Blackbird": SessionSchema(
        mach=3.2, alt_km=25.0, mass_kg=30600.0, R_n_m=0.15,
        g_load=2.5, char_len_m=32.7,
        # 5400 s = 90-min sustained supersonic-cruise segment.
        # Reconnaissance missions ran 3-4 h total but most of that
        # was subsonic positioning; the 90-min figure captures the
        # worst-case sustained thermal soak. 3000 h is the typical
        # SR-71 airframe lifetime (Lockheed reported 1959-1999 fleet
        # accrual). 1.5 mm Ti skin is the documented gauge on
        # production SR-71A panels (Goodall 1995).
        flight_duration_s=5400.0,
        design_lifetime_hours=3000.0,
        panel_thickness_m=0.0015,
        vehicle_category="aircraft",
        system_label="SR-71 Blackbird",
    ),
    "X-15": SessionSchema(
        mach=6.7, alt_km=30.0, mass_kg=15195.0, R_n_m=0.30,
        g_load=5.0, char_len_m=15.5,
        # X-15 powered phase ~90-120 s per flight; 199 missions
        # × ~10 min total mission = ~33 h fleet total. Per-mission
        # 600 s is conservative; lifetime 50 h covers the program.
        flight_duration_s=600.0,
        design_lifetime_hours=50.0,
        vehicle_category="aircraft",
        system_label="X-15",
    ),
    "Mach 4 Tactical Missile": SessionSchema(
        mach=4.0, alt_km=20.0, mass_kg=900.0, R_n_m=0.08,
        g_load=10.0, char_len_m=4.5,
        # Tactical missiles are single-flight munitions: ~120 s
        # boost-cruise-terminal. Lifetime equals mission duration.
        flight_duration_s=120.0,
        design_lifetime_hours=1.0,
        vehicle_category="hypersonic_missile",
        system_label="Mach 4 Tactical Missile",
    ),
    "Supersonic Cruise": SessionSchema(
        mach=2.5, alt_km=18.0, mass_kg=45000.0, R_n_m=0.40,
        g_load=2.0, char_len_m=28.0,
        # Generic SST envelope. 7200 s (2 h) cruise, 10000 h fleet life.
        flight_duration_s=7200.0,
        design_lifetime_hours=10000.0,
        vehicle_category="aircraft",
        system_label="Supersonic Cruise",
    ),
    "Small Reentry Capsule": SessionSchema(
        mach=20.0, alt_km=70.0, mass_kg=500.0, R_n_m=1.50,
        g_load=8.0, char_len_m=2.2,
        # Single-mission ballistic entry: peak heating phase ~600 s.
        # Capsule is single-use, so lifetime == mission.
        flight_duration_s=600.0,
        design_lifetime_hours=1.0,
        vehicle_category="reentry",
        system_label="Small Reentry Capsule",
    ),
    "Turbine HPT Blade": SessionSchema(
        # R_n_m=0.005 (5 mm): real HPT-blade leading-edge radius per
        # Cohen/Rogers/Saravanamuttoo "Gas Turbine Theory" 6e, Ch. 7.
        # Previously 0.05 m (50 mm), wrong for a single blade. mass and
        # char_len remain test-pinned scaled placeholders — see the
        # _PRESET_NOTES entry in app.py for the honest framing.
        mach=0.5, alt_km=0.0, mass_kg=50.0, R_n_m=0.005,
        g_load=1.0, char_len_m=0.1,
        # Single mission cycle ~2 h. Overhaul interval 25,000 h is
        # the relevant lifetime for blade creep evaluation
        # (CFM56 / GE90 / Trent class shop visits). 1 mm airfoil leaf
        # is the wall thickness of a typical HPT-blade pressure-side
        # face (Reed, The Superalloys, Ch. 1).
        flight_duration_s=7200.0,
        design_lifetime_hours=25000.0,
        panel_thickness_m=0.0010,
        vehicle_category="turbine",
        system_label="Turbine HPT Blade",
        options={"hot_section_temp_K": 1400.0},
    ),
    "General Structure Panel": SessionSchema(
        mach=0.3, alt_km=5.0, mass_kg=500.0, R_n_m=0.5,
        g_load=2.0, char_len_m=2.0,
        # Generic test panel — short mission, modest fatigue life.
        flight_duration_s=600.0,
        design_lifetime_hours=1000.0,
        vehicle_category="general",
        system_label="General Structure Panel",
    ),
    "Generic Hypersonic Aircraft": SessionSchema(
        mach=6.0, alt_km=30.0, mass_kg=12000.0, R_n_m=0.25,
        g_load=4.0, char_len_m=20.0,
        # Hypothetical sustained hypersonic cruiser: 1 h cruise,
        # ~500 h research-fleet lifetime.
        flight_duration_s=3600.0,
        design_lifetime_hours=500.0,
        vehicle_category="hypersonic_aircraft",
        system_label="Generic Hypersonic Aircraft",
    ),
    # ── New audience-appropriate presets (mirrored from app.EXAMPLES) ──
    # The UI dropdown shows these to startup / hobbyist / commercial
    # users. They live here too so:
    #   1. The CLI `validate` command sweeps over them.
    #   2. test_session.py:TestPresetParity stays green (UI ⊂ canonical).
    #   3. matvec sweep / sensitivity can reference them by name.
    #
    # The historical 8 entries above remain untouched so test_api.py and
    # test_sensitivity.py keep finding SR-71 / X-15 / Turbine HPT.
    "Consumer Quadcopter": SessionSchema(
        mach=0.06, alt_km=0.1, mass_kg=1.0, R_n_m=0.05,
        g_load=5.0, char_len_m=0.30,
        # Consumer drone: ~30 min flight, ~500 h consumer lifecycle.
        flight_duration_s=1800.0,
        design_lifetime_hours=500.0,
        vehicle_category="general",
        system_label="Consumer Quadcopter",
    ),
    "Consumer FPV Drone": SessionSchema(
        mach=0.10, alt_km=0.2, mass_kg=0.7, R_n_m=0.03,
        g_load=12.0, char_len_m=0.20,
        # FPV / racing: ~5 min battery, ~100 h race-frame life.
        flight_duration_s=300.0,
        design_lifetime_hours=100.0,
        vehicle_category="general",
        system_label="Consumer FPV Drone",
    ),
    "High-Power Amateur Rocket": SessionSchema(
        mach=1.5, alt_km=5.0, mass_kg=5.0, R_n_m=0.05,
        g_load=25.0, char_len_m=2.0,
        # Boost-coast: ~30 s powered + glide. Rebuilt-between-flights
        # so lifetime ~10 h.
        flight_duration_s=30.0,
        design_lifetime_hours=10.0,
        vehicle_category="general",
        system_label="High-Power Amateur Rocket",
    ),
    "High-Altitude Balloon Payload": SessionSchema(
        mach=0.05, alt_km=30.0, mass_kg=3.0, R_n_m=0.10,
        g_load=1.5, char_len_m=0.30,
        # Balloon ascent ~3 h to apogee. Payloads are reusable-once.
        flight_duration_s=10800.0,
        design_lifetime_hours=10.0,
        vehicle_category="general",
        system_label="High-Altitude Balloon Payload",
    ),
    "Collegiate Sounding Rocket": SessionSchema(
        # IREC / Spaceport America Cup 30k-ft category competition rocket.
        # M=2.0 is typical L3-motor peak velocity at burnout; the most
        # competitive teams hit ~M=2.5, but M=2.0 is the realistic
        # competition median. Apogee ~9 km matches the IREC 30,000 ft
        # category target. Mass 30 kg sits below the IREC 50 kg cap;
        # R_n 0.05 m corresponds to a typical 4-inch (102 mm) ogive
        # nose. Boost peak g ~15 from a Class L motor.
        mach=2.0, alt_km=9.0, mass_kg=30.0, R_n_m=0.05,
        g_load=15.0, char_len_m=3.0,
        # Single-flight burn-coast trajectory: ~10 s boost + ~15 s
        # coast to apogee. Airframes are typically rebuilt between
        # competition flights, so design lifetime is single-flight.
        # 3 mm wall is typical for a 4-inch L-motor competition tube
        # (aluminium / fiberglass / phenolic vendors).
        flight_duration_s=25.0,
        design_lifetime_hours=1.0,
        panel_thickness_m=0.0030,
        # General category: NOT hypersonic_missile. The missile category
        # applies a specific-strength weighting + slender-body plasma
        # threshold tuned for sustained-flight tactical missiles. A
        # boost-coast student rocket has none of those failure modes;
        # general gives the permissive bucket that surfaces aluminum
        # and CFRP --- exactly what real teams build with.
        vehicle_category="general",
        system_label="Collegiate Sounding Rocket",
    ),
    "Small Commercial UAV": SessionSchema(
        mach=0.15, alt_km=1.0, mass_kg=10.0, R_n_m=0.05,
        g_load=4.0, char_len_m=0.7,
        # Inspection / mapping mission ~30 min; ~2000 h commercial life.
        flight_duration_s=1800.0,
        design_lifetime_hours=2000.0,
        vehicle_category="aircraft",
        system_label="Small Commercial UAV",
    ),
    "eVTOL Air Taxi Prototype": SessionSchema(
        mach=0.3, alt_km=3.0, mass_kg=2200.0, R_n_m=0.15,
        g_load=3.0, char_len_m=12.0,
        # Hop ~30 min; certification target ~25,000 h fleet life.
        flight_duration_s=1800.0,
        design_lifetime_hours=25000.0,
        vehicle_category="aircraft",
        system_label="eVTOL Air Taxi Prototype",
    ),
    "Small Sat Reentry Capsule": SessionSchema(
        mach=22.0, alt_km=70.0, mass_kg=30.0, R_n_m=0.30,
        g_load=12.0, char_len_m=1.0,
        # Single-use ballistic capsule.
        flight_duration_s=600.0,
        design_lifetime_hours=1.0,
        vehicle_category="reentry",
        system_label="Small Sat Reentry Capsule",
    ),
    "Small Turbojet Engine": SessionSchema(
        mach=0.5, alt_km=0.0, mass_kg=25.0, R_n_m=0.005,
        g_load=1.0, char_len_m=0.15,
        # Small uncooled turbojet: 1 h mission, ~5000 h between
        # overhauls (target-drone / model-jet duty cycle).
        flight_duration_s=3600.0,
        design_lifetime_hours=5000.0,
        vehicle_category="turbine",
        system_label="Small Turbojet Engine",
        options={"hot_section_temp_K": 1100.0},
    ),
    # Concorde reference. Numbers match VALIDATION_CASES in
    # scripts/run_validation.py so the bundled preset and the
    # validation row stay byte-identical (M=2.04, 18 km cruise,
    # 78,000 kg MTOW, 61.66 m fuselage). Al 2618 (RR58) is not in
    # materials_db.py — the closest analogues (2024-T3, 2219-T87)
    # land in the marginal list, which is the documented limitation
    # called out in the validation prose.
    "Concorde": SessionSchema(
        mach=2.04, alt_km=18.0, mass_kg=78000.0, R_n_m=0.40,
        g_load=2.0, char_len_m=61.66,
        # 10800 s = 3-hour trans-Atlantic supersonic cruise.
        # 25,000 h is the typical Concorde retirement-fleet
        # accrual (BA G-BOAB and AF F-BTSC each accumulated
        # ~22-23k flight hours over 27 years of service). 2 mm
        # skin matches published Hiduminium RR58 / Al 2618
        # airframe-panel gauges (Aerospatiale-BAe materials
        # reports).
        flight_duration_s=10800.0,
        design_lifetime_hours=25000.0,
        panel_thickness_m=0.002,
        vehicle_category="aircraft",
        system_label="Concorde",
    ),
}
