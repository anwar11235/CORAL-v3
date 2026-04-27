# CORAL v3 — repo-root Makefile
#
# Targets
# -------
#   canonical-dataset   Build the canonical Sudoku-Extreme-1K dataset and verify it.
#
# Usage
# -----
#   make canonical-dataset DATA_DIR=/workspace/data/sudoku-extreme-1k-aug-1000

.PHONY: canonical-dataset

# Override on the command line: make canonical-dataset DATA_DIR=/your/path
DATA_DIR ?= /workspace/data/sudoku-extreme-1k-aug-1000

canonical-dataset:
	@echo "[CORAL] Building canonical Sudoku-Extreme-1K dataset -> $(DATA_DIR)"
	python coral/data/build_sudoku_dataset.py \
		--output-dir "$(DATA_DIR)" \
		--subsample-size 1000 \
		--num-aug 1000 \
		--seed 0
	@echo "[CORAL] Verifying dataset metadata..."
	python scripts/verify_canonical_dataset.py "$(DATA_DIR)"
