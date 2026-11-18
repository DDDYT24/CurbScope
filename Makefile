# Run from WSL (or Linux). `make help` lists the targets.
UV ?= uv
# Keep the venv on the Linux disk even when the repo sits on /mnt/c.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/curbscope
KITTI_DIR ?= $(HOME)/data/kitti
RUN = $(UV) run --frozen
# Ultralytics: no network (no usage analytics), no automatic pip installs.
YOLO_ENV = YOLO_OFFLINE=1 YOLO_AUTOINSTALL=False
# Hugging Face: no usage telemetry when TrOCR is downloaded.
HF_ENV = HF_HUB_DISABLE_TELEMETRY=1

.PHONY: help setup lint test info weights track geo motion label tune evaluate segments strip-exif ocr report dashboard all

help:
	@echo "setup               install pinned dependencies with uv"
	@echo "lint                ruff check"
	@echo "test                unit tests (no KITTI data needed)"
	@echo "info DRIVE=<id>     frames, duration and distance of one KITTI drive"
	@echo "weights             download YOLO weights and check their SHA-256"
	@echo "track [DRIVE=<id>]  YOLO + ByteTrack on all drives in the config, or one"
	@echo "geo [DRIVE=<id>]    LiDAR position, lat/lon and error against KITTI labels"
	@echo "motion              per-track features for the parked/moving rules"
	@echo "label               open the track labelling page (writes labels/)"
	@echo "tune                rule accuracy over thresholds, dev drives only"
	@echo "evaluate SPLIT=dev|test  score both rules with configs/frozen_thresholds.yaml"
	@echo "segments [DRIVE=<id>]  curb segments and GeoJSON (needs frozen thresholds)"
	@echo "strip-exif          copy data/signs/ to data/signs_clean/ without EXIF"
	@echo "ocr                 compare three OCR engines on the sign photos"
	@echo "report              results/report.md and the results block of README.md"
	@echo "dashboard           Streamlit dashboard"
	@echo "all                 every step whose inputs exist, then the report"

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

motion:
	$(RUN) python scripts/run_pipeline.py --stage motion $(if $(DRIVE),--drive $(DRIVE))

label:
	$(RUN) streamlit run apps/label_tracks.py --server.headless true -- --kitti-dir $(KITTI_DIR)

tune:
	$(RUN) python scripts/tune.py --split dev

evaluate:
	@test -n "$(SPLIT)" || (echo "usage: make evaluate SPLIT=dev  (or SPLIT=test)" && exit 2)
	$(RUN) python scripts/evaluate.py --split $(SPLIT)

segments:
	$(RUN) python scripts/run_pipeline.py --stage segments --kitti-dir $(KITTI_DIR) $(if $(DRIVE),--drive $(DRIVE))

strip-exif: weights
	$(YOLO_ENV) $(RUN) python scripts/strip_exif.py

ocr:
	$(HF_ENV) $(RUN) python scripts/ocr_benchmark.py

report:
	$(RUN) python -m curbscope.report

dashboard:
	$(RUN) streamlit run apps/dashboard.py --server.headless true -- --kitti-dir $(KITTI_DIR)

# Steps that need the owner's input run only once that input exists; the report says what is
# still pending.
all: track geo motion
	@if [ -f configs/frozen_thresholds.yaml ]; then 		$(MAKE) evaluate SPLIT=dev && $(MAKE) evaluate SPLIT=test && $(MAKE) segments; 	else echo "skipping evaluate and segments: configs/frozen_thresholds.yaml does not exist yet"; fi
	@if [ -f labels/signs_groundtruth.csv ]; then $(MAKE) strip-exif ocr; 	else echo "skipping OCR: labels/signs_groundtruth.csv does not exist yet"; fi
	$(MAKE) report
