# Visualization Guide for RQ1 Research

## Quick Start: Generate All Report Figures

```bash
# First, run the research analysis (if not already done)
python experiments/rq1_vaccination_timing_quick.py

# Then generate all publication-quality figures
python experiments/generate_report_figures.py
```

Output location: `results/rq1_vaccination_timing/report_figures/`

---

## Available Figures

### Figure 1: Comprehensive Timing Analysis
**File:** `figure1_comprehensive_timing_analysis.png`

**What it shows:**
- **Row 1:** Attack rate vs vaccination start time (3 panels for each R₀)
- **Row 2:** Peak infections vs vaccination start time
- **Row 3:** Epidemic duration vs vaccination start time

**Key features:**
- Gold stars mark optimal timing
- Shaded regions show "optimal window" (within 5% of optimal)
- Dashed red lines show baseline (no vaccination)
- Vertical dotted lines mark epidemic peaks
- Text boxes display key metrics

**Use in report:** Main results section - this is your primary figure

**Interpretation:**
- Steep curves = timing is critical
- Flat curves = more flexibility in timing
- Compare curve shapes across R₀ values to draw conclusions

---

### Figure 2: Baseline Epidemic Dynamics
**File:** `figure2_baseline_comparison.png`

**What it shows:**
- **Row 1:** SEIR time series for each R₀
- **Row 2:** Phase portraits (S vs I) with time coloring
- **Row 3:** R_effective over time

**Key features:**
- All compartments (S, E, I, R) with proper colors
- Annotated peak times and key metrics
- R_eff crossing threshold (R_eff = 1) marked
- Start/end points clearly labeled on phase portraits

**Use in report:** Methods section - shows baseline characterization

**Interpretation:**
- Shows epidemic speed increases with R₀
- Phase portraits show epidemic trajectory in state space
- R_eff plots show when epidemic transitions from growth to decline

---

### Figure 3: Stochastic Variability Analysis
**File:** `figure3_stochastic_analysis.png`

**What it shows:**
- **Row 1:** Box plots of attack rates at different timing
- **Row 2:** Mean ± standard deviation with 95% confidence intervals
- **Row 3:** Coefficient of variation (relative uncertainty)

**Key features:**
- Shows uncertainty in outcomes
- Whiskers and outliers visible in box plots
- Shaded confidence bands
- CV% indicates relative uncertainty

**Use in report:** Uncertainty quantification section

**Interpretation:**
- Wide boxes/error bars = high uncertainty
- CV% measures predictability (lower = more predictable)
- Compare uncertainty levels across R₀ values

**Note:** Only available if stochastic simulations were run (not in quick version)

---

### Figure 4: Parameter Sensitivity Heatmaps
**File:** `figure4_sensitivity_heatmaps.png`

**What it shows:**
- Heatmaps showing optimal attack rates for combinations of:
  - Vaccine efficacy (rows): 50%, 70%, 90%
  - Vaccination rate (columns): 0.5%, 1%, 2% per day
- One heatmap per R₀ value

**Key features:**
- Color scale: green = good (low attack rate), red = bad (high attack rate)
- Numerical values annotated in each cell
- Colorbar with baseline reference line

**Use in report:** Sensitivity analysis section

**Interpretation:**
- Identify which parameter matters more (look at gradient direction)
- Check for threshold effects or linear relationships
- Compare sensitivity patterns across R₀ values

**Note:** Only available if sensitivity analysis was run (not in quick version)

---

### Figure 5: Vaccination Benefit Waterfall Chart
**File:** `figure5_benefit_waterfall.png`

**What it shows:**
- Side-by-side bar comparison:
  - Red bars: Baseline (no vaccination)
  - Blue bars: Optimal vaccination
- Green arrows showing reduction
- Percentage reduction labeled

**Key features:**
- Clear visual comparison of benefit
- Absolute values labeled on bars
- Relative reduction (%) prominently displayed
- Summary interpretation text box

**Use in report:** Key findings / Discussion section

**Interpretation:**
- Directly see vaccination impact for each R₀
- Compare absolute vs relative benefits
- Supports quantitative claims about effectiveness

---

### Figure 6: Optimal Timing Windows Analysis
**File:** `figure6_timing_windows.png`

**What it shows:**
- **Panel 1:** Optimal timing vs R₀ (with peak timing for reference)
- **Panel 2:** Window width vs R₀ (how flexible timing is)
- **Panel 3:** Relative timing (optimal time / peak time)
- **Panel 4:** Summary table with all metrics

**Key features:**
- Annotations showing days before/after peak
- Color-coded interpretation (green = before peak, red = after)
- Shaded regions in panel 3 (before/after peak zones)
- Professional table with alternating row colors

**Use in report:** Results interpretation section

**Interpretation:**
- Shows relationship between R₀ and optimal timing
- Quantifies "timing flexibility" across regimes
- Table provides precise values for text/discussion

---

### Figure 7: Vaccination Dynamics Examples
**File:** `figure7_vaccination_dynamics.png`

**What it shows:**
- SEIRV time series for three scenarios:
  - **Left column:** Early vaccination
  - **Middle column:** Optimal vaccination
  - **Right column:** Late vaccination
- One row per R₀ value

**Key features:**
- All 5 compartments (S, E, I, R, V) plotted
- Vertical dashed line marks vaccination start
- Shaded region shows vaccination period
- Attack rates annotated in titles
- Color-coded titles (blue=early, green=optimal, red=late)

**Use in report:** Methods / Illustration section

**Interpretation:**
- Visual comparison of how timing affects dynamics
- Shows mechanism: early vaccination prevents infections,
  late vaccination arrives too late
- V compartment shows vaccination coverage achieved

---

## How to Customize Figures

### Change Figure Size or DPI

Edit in the individual visualization functions:

```python
# In analysis/rq1_visualizations.py

# Find the function, e.g., create_comprehensive_timing_analysis
# Modify figsize parameter
fig = plt.figure(figsize=(20, 14))  # Make larger

# When saving, modify DPI
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # Higher resolution
```

### Change Colors

```python
# In analysis/rq1_visualizations.py

# Find the color definitions, e.g.
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Change these hex codes

# Or for specific elements
ax.plot(..., color='#YOUR_HEX_CODE')
```

### Add Your Own Annotations

```python
# After generating a figure, add annotations:
ax.annotate('Your text here',
           xy=(x_position, y_position),
           xytext=(text_x, text_y),
           arrowprops=dict(arrowstyle='->', color='black', lw=2),
           fontsize=12, fontweight='bold')
```

### Remove/Add Elements

Comment out sections in the visualization functions:

```python
# To remove the optimal window shading:
# ax.axvspan(effective_times[0], effective_times[-1], ...)  # Comment this line

# To add a horizontal line at a specific value:
ax.axhline(y=your_value, color='purple', linestyle=':', linewidth=2,
          label='Your label')
```

---

## Figure Quality Guidelines for Reports

### Recommended Settings

**For digital reports (PDFs):**
- DPI: 300
- Format: PNG or PDF
- Font size: 10-12 pt

**For print publications:**
- DPI: 600
- Format: PDF (vector) or high-res PNG
- Font size: 8-10 pt
- Consider color-blind friendly palettes

### Current Settings

All figures are generated with:
- **DPI:** 300 (publication quality)
- **Format:** PNG with white background
- **Font sizes:**
  - Main title: 14-16 pt
  - Subplot titles: 12-13 pt
  - Axis labels: 11-12 pt
  - Tick labels: 9 pt
  - Legend: 9-11 pt
- **Colors:** Professionally chosen, color-blind considerate
- **Bounding box:** Tight (no extra whitespace)

---

## Usage Tips

### For Your Report

1. **Introduction:** Use Figure 7 (vaccination dynamics) to illustrate the model

2. **Methods:**
   - Use Figure 2 (baseline dynamics) to show model validation
   - Use Figure 7 (one panel) to explain SEIRV model

3. **Results:**
   - Lead with Figure 1 (comprehensive timing) - this is your main result
   - Follow with Figure 5 (benefit waterfall) for impact summary
   - Use Figure 6 (timing windows) for detailed optimal timing analysis

4. **Sensitivity Analysis:**
   - Use Figure 4 (heatmaps) for parameter sensitivity
   - Use Figure 3 (stochastic) for uncertainty quantification

5. **Discussion:**
   - Reference all figures to support your conclusions
   - Figure 6's table provides exact values for the text

### Recommended Figure Order

**In a typical research paper:**
1. Figure 7 or Figure 2 - Model illustration
2. Figure 1 - Main results
3. Figure 6 - Optimal timing analysis
4. Figure 5 - Benefit summary
5. Figure 3 - Stochastic analysis (supplementary)
6. Figure 4 - Sensitivity (supplementary)

### Captions

Here are suggested captions for each figure:

**Figure 1:**
> "Comprehensive analysis of vaccination timing effects across R₀ regimes. (Top row) Final attack rate vs. vaccination start time. Gold stars indicate optimal timing, and shaded regions show timing windows within 5% of optimal. (Middle row) Peak infectious fraction vs. timing. (Bottom row) Time to epidemic extinction vs. timing. Red dashed lines show baseline (no vaccination), and vertical dotted lines mark epidemic peaks."

**Figure 2:**
> "Baseline epidemic dynamics without vaccination. (Top row) SEIR compartment dynamics over time, with epidemic peaks marked. (Middle row) Phase portraits showing epidemic trajectory in S-I space, colored by time. (Bottom row) Effective reproduction number R_eff(t) over time, showing transition through epidemic threshold."

**Figure 3:**
> "Stochastic variability in vaccination outcomes. (Top row) Box plots showing distribution of attack rates across 30 replicates for each timing. (Middle row) Mean attack rates with standard deviation (error bars) and 95% confidence intervals (shaded regions). (Bottom row) Coefficient of variation showing relative uncertainty as a function of timing."

**Figure 4:**
> "Parameter sensitivity heatmaps showing optimal attack rates for different vaccine efficacy and vaccination rate combinations. Green indicates low attack rates (good outcomes) and red indicates high attack rates. Numbers in cells show exact optimal attack rate values. Colorbar includes baseline reference line."

**Figure 5:**
> "Vaccination benefit across R₀ regimes. Red bars show baseline attack rates without vaccination, blue bars show optimal vaccination outcomes. Green arrows indicate absolute reduction, with percentage reduction labeled. Higher R₀ diseases show [interpret your specific pattern]."

**Figure 6:**
> "Analysis of optimal vaccination timing windows. (A) Optimal start times compared to epidemic peak times. Annotations show timing relative to peak. (B) Width of optimal timing windows (within 5% of optimal). (C) Relative timing (optimal time normalized by peak time). (D) Summary table of key metrics."

**Figure 7:**
> "Example vaccination dynamics for early, optimal, and late timing strategies. Each row shows a different R₀ regime. Vertical dashed lines indicate vaccination campaign start, and shaded regions show vaccination periods. All five compartments (S, E, I, R, V) are shown, with attack rates in titles."

---

## Troubleshooting

### "No module named 'seaborn'"

Install seaborn:
```bash
pip install seaborn
```

### "No module named 'tqdm'"

Install tqdm:
```bash
pip install tqdm
```

### "FileNotFoundError: phase1_baseline.pkl"

You need to run the research first:
```bash
python experiments/rq1_vaccination_timing_quick.py
# or
python experiments/rq1_vaccination_timing.py
```

### "Sensitivity results not found"

This is normal for the quick version. Run the full version to get sensitivity analysis:
```bash
python experiments/rq1_vaccination_timing.py
```

### Figures look different than expected

Check your matplotlib backend:
```python
import matplotlib
print(matplotlib.get_backend())
```

If issues persist, try:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### Font warnings

If you see font warnings, you can suppress them:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
```

Or install additional fonts on your system.

---

## Advanced: Creating Custom Visualizations

### Using the Visualization Functions Directly

```python
from analysis.rq1_visualizations import create_comprehensive_timing_analysis
import pickle

# Load your results
with open('results/rq1_vaccination_timing/phase1_baseline.pkl', 'rb') as f:
    baseline_results = pickle.load(f)

with open('results/rq1_vaccination_timing/phase2_timing_sweep.pkl', 'rb') as f:
    timing_results = pickle.load(f)

# Generate figure
R0_VALUES = [1.5, 2.5, 4.0]
fig = create_comprehensive_timing_analysis(
    baseline_results,
    timing_results,
    R0_VALUES,
    save_path='my_custom_figure.png'
)

# Modify figure before saving (optional)
fig.suptitle('My Custom Title', fontsize=18)
plt.savefig('my_modified_figure.png', dpi=300)
```

### Combining Multiple Figures

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create custom layout
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig)

# Generate sub-figures in specific positions
ax1 = fig.add_subplot(gs[0, :])  # Top row, full width
ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
ax3 = fig.add_subplot(gs[1, 1])  # Bottom right

# Plot your data in each axis
# ... your plotting code ...

plt.savefig('combined_figure.png', dpi=300, bbox_inches='tight')
```

---

## Getting Help

1. **Check the code:** All visualization functions are in `analysis/rq1_visualizations.py`
2. **Read docstrings:** Each function has detailed documentation
3. **Run examples:** The generate_report_figures.py script shows usage
4. **Modify incrementally:** Start with existing figures and tweak parameters

---

## Summary Checklist

- [ ] Run research analysis (quick or full version)
- [ ] Generate all figures using generate_report_figures.py
- [ ] Review FIGURE_INDEX.txt for descriptions
- [ ] Select figures for your report sections
- [ ] Write appropriate captions
- [ ] Check figure quality (DPI, fonts, colors)
- [ ] Save figures in appropriate format for your report
- [ ] Cite figures in text with proper numbering
- [ ] Include figure legends/keys where needed
- [ ] Proofread all labels and annotations

---

**All set! Your publication-quality visualizations are ready for your report.**
