# Run from WSL (or Linux). `make help` lists the targets.
UV ?= uv
# Keep the venv on the Linux disk even when the repo sits on /mnt/c.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/curbscope
KITTI_DIR ?= $(HOME)/data/kitti
RUN = $(UV) run --frozen
# Ultralytics: no network (no usage analytics), no automatic pip installs.
YOLO_ENV = YOLO_OFFLINE=1 YOLO_AUTOINSTALL=False

.PHONY: help setup lint test info weights track geo

help:
	@echo "setup               install pinned dependencies with uv"
	@echo "lint                ruff check"
	@echo "test                unit tests (no KITTI data needed)"
	@echo "info DRIVE=<id>     frames, duration and distance of one KITTI drive"
	@echo "weights             download YOLO weights and check their SHA-256"
	@echo "track [DRIVE=<id>]  YOLO + ByteTrack on all drives in the config, or one"
	@echo "geo [DRIVE=<id>]    LiDAR position, lat/lon and error against KITTI labels"

setup:
	$(UV) sync --frozen

lint:
	$(RUN) ruff check .

test:
	$(RUN) pytest -q

info:
	@test -n "$(DRIVE)" || (echo "usage: make info DRIVE=2011_09_26_drive_0022" && exit 2)
	$(RUN) python scripts/info.py --kitti-dir $(KITTI_DIR) --drive $(DRIVE)

weights:
	$(RUN) python scripts/get_weights.py

track: weights
	$(YOLO_ENV) $(RUN) python scripts/run_pipeline.py --stage track --kitti-dir $(KITTI_DIR) $(if $(DRIVE),--drive $(DRIVE))

geo:
	$(RUN) python scripts/run_pipeline.py --stage geo --kitti-dir $(KITTI_DIR) $(if $(DRIVE),--drive $(DRIVE))
