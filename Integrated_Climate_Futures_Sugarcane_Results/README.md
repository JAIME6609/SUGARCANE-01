# Integrated Climate Futures and Decision Support for Resilient Sugarcane

This reproducibility package contains the complete synthetic worked case used in the chapter.
It is intended to demonstrate auditable integration, not to prescribe management in a real region.
Local calibration, independent field validation, and stakeholder review are required before use.

## Output structure

- `5.1_Climate_and_System_Response`: scenario envelope, process-model response, and resource trade-offs.
- `5.2_Predictive_Performance_and_Uncertainty`: surrogate validation, SHAP explanations, conformal intervals, and Jansen sensitivity indices.
- `5.3_Robust_Decisions_and_Adaptive_Pathways`: multicriteria robustness, regret, Pareto trade-offs, preference uncertainty, and adaptive triggers.
- `data`: synthetic scenarios and scenario-strategy performance records.

## Reproduction

```bash
python integrated_climate_futures_sugarcane.py --output-dir "PATH_TO_RESULTS"
```

Use `--auto-install` only when package installation is permitted. The deterministic seed is
4901; the ensemble contains 1800 futures; the Sobol base size is
512; and the measured runtime for this run was 17.77 seconds.
