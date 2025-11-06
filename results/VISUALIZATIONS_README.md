# Epidemic Simulator Visualizations

## Quick Start: Generate All Figures

```bash
# Step 1: Run the research (choose one)
python experiments/rq1_vaccination_timing_quick.py          # Fast (2-5 min)
python experiments/rq1_vaccination_timing.py               # Full (30-60 min)

# Step 2: Generate all report-quality figures
python experiments/generate_report_figures.py
```

## Output

All figures will be saved to: `results/rq1_vaccination_timing/report_figures/`

## Available Figures

| Figure | File | Description | Best For |
|--------|------|-------------|----------|
| **1** | `figure1_comprehensive_timing_analysis.png` | 9-panel analysis: attack rate, peak infections, duration vs timing | Main results |
| **2** | `figure2_baseline_comparison.png` | SEIR dynamics, phase portraits, R_eff plots | Methods/baseline |
| **3** | `figure3_stochastic_analysis.png` | Box plots, confidence intervals, variability | Uncertainty |
| **4** | `figure4_sensitivity_heatmaps.png` | Parameter sensitivity heatmaps | Sensitivity analysis |
| **5** | `figure5_benefit_waterfall.png` | Vaccination benefit comparison | Key findings |
| **6** | `figure6_timing_windows.png` | Optimal windows and flexibility analysis | Interpretation |
| **7** | `figure7_vaccination_dynamics.png` | Early/optimal/late vaccination examples | Illustration |

## Features

✅ **Publication Quality**
- 300 DPI resolution
- Professional color schemes
- Proper labels, legends, titles
- Annotated with key metrics

✅ **Report-Ready**
- Consistent styling across all figures
- Clear visual hierarchy
- Informative annotations
- Statistical summaries included

✅ **Comprehensive Coverage**
- Baseline characterization
- Timing optimization results
- Sensitivity analysis
- Uncertainty quantification
- Benefit analysis
- Window characterization

## Documentation

- **Full Guide:** `experiments/VISUALIZATION_GUIDE.md`
  - Detailed descriptions of each figure
  - Customization instructions
  - Suggested captions
  - Troubleshooting tips

- **Figure Index:** `results/rq1_vaccination_timing/report_figures/FIGURE_INDEX.txt`
  - Generated after running generate_report_figures.py
  - Lists all available figures
  - Suggests report structure

## Code Location

- **RQ1 Visualizations:** `analysis/rq1_visualizations.py`
- **General Utilities:** `analysis/visualization.py`
- **Generation Script:** `experiments/generate_report_figures.py`

## Requirements

```bash
pip install numpy matplotlib seaborn pandas
```

## Customization

All visualization functions accept standard matplotlib parameters and can be customized:

```python
from analysis.rq1_visualizations import create_comprehensive_timing_analysis

# Generate with custom styling
fig = create_comprehensive_timing_analysis(
    baseline_results,
    timing_results,
    R0_VALUES,
    save_path='custom_figure.png'
)

# Further customize
fig.suptitle('My Custom Title', fontsize=20)
plt.savefig('my_figure.png', dpi=600)  # Higher resolution
```

See `VISUALIZATION_GUIDE.md` for detailed customization examples.

## Troubleshooting

**Problem:** "FileNotFoundError: phase1_baseline.pkl"
**Solution:** Run the research script first (step 1 above)

**Problem:** Sensitivity figures missing
**Solution:** Run the full version (`rq1_vaccination_timing.py`), not the quick version

**Problem:** Import errors
**Solution:** `pip install -r requirements.txt`

## Support

For questions or issues with visualizations:
1. Check `VISUALIZATION_GUIDE.md` for detailed help
2. Review code in `analysis/rq1_visualizations.py`
3. See examples in `experiments/generate_report_figures.py`

---

**Ready to create publication-quality figures for your epidemic modeling research!**
