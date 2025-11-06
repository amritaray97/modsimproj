# RQ1: Vaccination Timing Optimization Research Guide

## Research Question

**How does the timing of vaccination campaigns affect epidemic outcomes across different R₀ regimes?**

## Scientific Motivation

Vaccination campaigns are rarely instantaneous. Understanding when to deploy limited vaccine supplies (early vs. at epidemic peak) and how this decision interacts with disease transmissibility is crucial for public health policy.

## Specific Sub-Questions

1. For diseases with R₀ = 1.5, 3.0, and 5.0, what is the optimal start time for vaccination campaigns that minimize:
   - Final attack rate (total infections)
   - Peak hospitalization burden
   - Time to epidemic extinction

2. How does vaccine efficacy (50%, 70%, 90%) modify the optimal timing strategy?

3. Is there a critical threshold in vaccination coverage rate (doses per day) below which timing becomes irrelevant?

## Methodology

### Phase 1: Baseline Characterization

**Objective:** Establish epidemic dynamics without intervention

**Approach:**
- Simulate SEIR dynamics for R₀ ∈ {1.5, 2.5, 4.0}
- Measure baseline metrics:
  - Epidemic peak time (t_peak)
  - Peak infectious fraction (I_peak)
  - Final attack rate (R_∞)

**Output:** Baseline curves and reference metrics for comparison

### Phase 2: Vaccination Timing Sweep

**Objective:** Systematically test different vaccination start times

**Parameters:**
- Vaccination start time: t_start ∈ [0, t_peak - 10, t_peak, t_peak + 10, ..., 2×t_peak]
- Vaccination rate: ν = 0.01 population/day (1% daily)
- Vaccine efficacy: ε = 0.80 (80%)

**Simulations:**
- Deterministic: Full timing sweep
- Stochastic: 30 replicates per condition (to capture uncertainty)

**Output:** Attack rate, peak infections, and epidemic duration as functions of start time

### Phase 3: Parameter Sensitivity

**Objective:** Test robustness to vaccine characteristics

**Parameters to vary:**
- Vaccine efficacy: ε ∈ {0.5, 0.7, 0.9}
- Vaccination rate: ν ∈ {0.005, 0.01, 0.02}

**Output:** Optimal timing windows for different parameter combinations

### Phase 4: Analysis and Visualization

**Key Analyses:**
1. **Attack Rate vs. Start Time:** Main result showing optimal windows
2. **Peak Infections vs. Start Time:** Healthcare burden implications
3. **Stochastic Variability:** Uncertainty quantification
4. **Parameter Sensitivity Heatmaps:** How efficacy and rate modify timing

**Quantitative Metrics:**
- Optimal vaccination start time for each R₀
- Percent reduction in attack rate: (AR_baseline - AR_optimal) / AR_baseline × 100
- Width of "effective window" (times within 5% of optimal)

## Expected Outcomes

### Hypothesis

**High R₀ diseases (R₀ ≥ 4):**
- Early vaccination is critical
- Narrow optimal window
- Delayed vaccination loses effectiveness rapidly

**Low R₀ diseases (R₀ ≈ 1.5-2):**
- More flexibility in timing
- Broader optimal window
- Even delayed vaccination provides substantial benefit

**Mechanism:** High R₀ means rapid initial spread, consuming susceptibles quickly. Once the epidemic peaks, most susceptibles are already infected, limiting vaccination impact.

## Running the Analysis

### Quick Version (Recommended for initial exploration)

```bash
python experiments/rq1_vaccination_timing_quick.py
```

**Runtime:** ~2-5 minutes
**Output:** Basic visualization showing key patterns

### Full Analysis

```bash
python experiments/rq1_vaccination_timing.py
```

**Runtime:** ~30-60 minutes (depending on system)
**Output:**
- `results/rq1_vaccination_timing/phase1_baseline.pkl` - Baseline data
- `results/rq1_vaccination_timing/phase2_timing_sweep.pkl` - Timing sweep data
- `results/rq1_vaccination_timing/phase3_sensitivity.pkl` - Sensitivity data
- `results/rq1_vaccination_timing/figure1_attack_rate_vs_timing.png` - Main result
- `results/rq1_vaccination_timing/figure2_peak_infections_vs_timing.png`
- `results/rq1_vaccination_timing/figure3_stochastic_variability.png`
- `results/rq1_vaccination_timing/figure4_parameter_sensitivity.png`
- `results/rq1_vaccination_timing/quantitative_analysis.csv` - Summary table

## Interpreting Results

### Figure 1: Attack Rate vs. Vaccination Timing

**What to look for:**
- **Optimal point (gold star):** Best time to start vaccination
- **Distance from peak:** How far before/after epidemic peak is optimal?
- **Curve steepness:** How quickly does delayed vaccination lose effectiveness?

**Key insight:** Compare curve shapes across R₀ values

### Figure 2: Peak Infections vs. Timing

**What to look for:**
- Reduction in peak (healthcare burden)
- Whether early vaccination prevents the peak entirely

**Public health implication:** Hospital capacity planning

### Figure 3: Stochastic Variability

**What to look for:**
- Error bars (uncertainty in outcomes)
- Whether optimal timing is robust to stochastic effects

**Key insight:** Narrow error bars = predictable outcomes

### Figure 4: Parameter Sensitivity

**What to look for:**
- How curves shift with different efficacy/rates
- Whether there's a "threshold effect" (e.g., below ν = 0.5%, timing doesn't matter)

## Model Details

### SEIRV Model

**Compartments:**
- S: Susceptible
- E: Exposed (infected but not yet infectious)
- I: Infectious
- R: Recovered (includes successful vaccinations)
- V: Vaccinated (failed vaccinations, no immunity)

**Equations:**
```
dS/dt = -β·S·I - ν(t)·S
dE/dt = β·S·I - σ·E
dI/dt = σ·E - γ·I
dR/dt = γ·I + ε·ν(t)·S
dV/dt = (1-ε)·ν(t)·S
```

Where:
- β: transmission rate
- σ: 1/incubation_period
- γ: 1/infectious_period
- ν(t): time-dependent vaccination rate
- ε: vaccine efficacy

### Parameter Values

**Fixed (based on COVID-19-like disease):**
- Incubation period: 5 days (σ = 0.2)
- Infectious period: 10 days (γ = 0.1)

**Varied:**
- β: Calculated from R₀ = β/γ
- ε: 50%, 70%, 90%
- ν: 0.5%, 1%, 2% per day

## Extensions and Future Work

1. **Age-structured models:** Different vaccination priorities for age groups
2. **Spatial heterogeneity:** Vaccination campaigns in different regions
3. **Waning immunity:** Re-vaccination strategies
4. **Multiple doses:** Prime-boost vaccination schedules
5. **Behavioral responses:** Vaccine hesitancy and acceptance dynamics

## References

1. Keeling, M. J., & Rohani, P. (2008). *Modeling infectious diseases in humans and animals.* Princeton University Press.

2. Anderson, R. M., & May, R. M. (1991). *Infectious diseases of humans: dynamics and control.* Oxford University Press.

3. Diekmann, O., & Heesterbeek, J. A. P. (2000). *Mathematical epidemiology of infectious diseases: model building, analysis and interpretation.* John Wiley & Sons.

4. Bubar, K. M., et al. (2021). "Model-informed COVID-19 vaccine prioritization strategies by age and serostatus." *Science*, 371(6532), 916-921.

5. Matrajt, L., et al. (2021). "Vaccine optimization for COVID-19: Who to vaccinate first?" *Science Advances*, 7(6), eabf1374.

## Contact and Support

For questions about this research code:
- Check the main README.md
- Review model implementations in `models/seirv_model.py`
- Examine analysis functions in the research script

## Citation

If you use this code for research, please cite:

```
[Your Name/Institution] (2025). Epidemic Simulator: Vaccination Timing
Optimization Research. GitHub: [repository-url]
```
