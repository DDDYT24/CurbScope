# Run from WSL (or Linux). `make help` lists the targets.
UV ?= uv
# Keep the venv on the Linux disk even when the repo sits on /mnt/c.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/curbscope
KITTI_DIR ?= $(HOME)/data/kitti
RUN = $(UV) run --frozen

.PHONY: help setup lint test info

help:
	@echo "setup               install pinned dependencies with uv"
	@echo "lint                ruff check"
	@echo "test                unit tests (no KITTI data needed)"
	@echo "info DRIVE=<id>     frames, duration and distance of one KITTI drive"

setup:
	$(UV) sync --frozen

lint:
	$(RUN) ruff check .

test:
	$(RUN) pytest -q

info:
	@test -n "$(DRIVE)" || (echo "usage: make info DRIVE=2011_09_26_drive_0019" && exit 2)
	$(RUN) python scripts/info.py --kitti-dir $(KITTI_DIR) --drive $(DRIVE)
