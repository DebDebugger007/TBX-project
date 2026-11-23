# Convenience targets. Override CFG=configs/your.yaml as needed.
CFG ?= configs/default.yaml
RUN = python scripts/05_run_experiment.py --config

.PHONY: data eda masks features within cross regions gradcam compare corrections table smoke all week8 \
        features-corrections corrections-classical corrections-sz2mc gradcam-clahe week9

data:      ; python scripts/01_prepare_data.py --config $(CFG)
eda:       ; python scripts/02_run_eda.py --config $(CFG)
masks:     ; python scripts/03_train_unet_masks.py --config $(CFG)
features:  ; python scripts/04_build_features.py --config $(CFG)

within:
	$(RUN) configs/E0_classical_within.yaml
	$(RUN) configs/E1_montgomery_within.yaml
	$(RUN) configs/E2_shenzhen_within.yaml

# Phase 5 (weeks 7-8)
cross:
	$(RUN) configs/E3_mc_to_sz_cross.yaml
	$(RUN) configs/E4_sz_to_mc_cross.yaml
	$(RUN) configs/E3_classical_mc_to_sz_cross.yaml
	$(RUN) configs/E4_classical_sz_to_mc_cross.yaml

regions:
	$(RUN) configs/E3_mc_to_sz_cross_lungmask.yaml
	$(RUN) configs/E3_mc_to_sz_cross_lungcrop.yaml
	$(RUN) configs/E4_sz_to_mc_cross_lungmask.yaml
	$(RUN) configs/E4_sz_to_mc_cross_lungcrop.yaml

gradcam:   ; python scripts/07_gradcam.py --config configs/E3_mc_to_sz_cross.yaml
compare:   ; python scripts/08_compare_auc.py

# Phase 6 (week 8, in progress): CNN Montgomery -> Shenzhen, one run per correction
corrections:
	$(RUN) configs/E3_mc_to_sz_cross_histeq.yaml
	$(RUN) configs/E3_mc_to_sz_cross_clahe.yaml
	$(RUN) configs/E3_mc_to_sz_cross_zscore.yaml

table:     ; python scripts/06_make_results_table.py --config $(CFG)
smoke:     ; python tests/smoke_test.py

# Full week 2-7 pipeline (data prep through within-set results)
all: data eda masks features within table

# Week 8 work on top of `all`
week8: cross regions gradcam corrections compare table

# Phase 6 (week 9): finish the correction family
features-corrections: ; python scripts/04_build_features.py --config $(CFG) --corrections histeq clahe zscore

corrections-classical: features-corrections
	$(RUN) configs/E3_classical_mc_to_sz_cross_histeq.yaml
	$(RUN) configs/E3_classical_mc_to_sz_cross_clahe.yaml
	$(RUN) configs/E3_classical_mc_to_sz_cross_zscore.yaml
	$(RUN) configs/E4_classical_sz_to_mc_cross_histeq.yaml
	$(RUN) configs/E4_classical_sz_to_mc_cross_clahe.yaml
	$(RUN) configs/E4_classical_sz_to_mc_cross_zscore.yaml

corrections-sz2mc:
	$(RUN) configs/E4_sz_to_mc_cross_histeq.yaml
	$(RUN) configs/E4_sz_to_mc_cross_clahe.yaml
	$(RUN) configs/E4_sz_to_mc_cross_zscore.yaml

gradcam-clahe: ; python scripts/07_gradcam.py --config configs/E3_mc_to_sz_cross_clahe.yaml

# Week 9 work on top of `week8`
week9: corrections-classical corrections-sz2mc gradcam-clahe compare table
