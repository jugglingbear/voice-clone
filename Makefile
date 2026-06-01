# ── voice-clone ───────────────────────────────────────────
# Synthesize speech in a cloned voice from a short reference
# audio/video clip (built on Chatterbox TTS).
# Type "make" (or "make help") to see available targets.
# ──────────────────────────────────────────────────────────

.DEFAULT_GOAL := help

PYTHON      ?= python3
POETRY      ?= poetry
MIN_PYTHON  := 3.10

.PHONY: help
help:  ## Show this help message
	@printf "\n\033[1mvoice-clone — available targets:\033[0m\n"
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@ / { printf "\n\033[1;38;5;208m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_-]+:.*?## / { printf "  \033[97m%-16s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)
	@printf "\n"

.PHONY: check
check:  ## Verify required tools (python3, poetry, ffmpeg) are installed
	@printf "\033[1mChecking environment...\033[0m\n"
	@ok=true; \
	command -v $(PYTHON) >/dev/null 2>&1 \
		&& printf "  ✅ python3   %s\n" "$$($(PYTHON) --version 2>&1 | awk '{print $$2}')" \
		|| { printf "  ❌ python3   not found (need >= $(MIN_PYTHON))\n"; ok=false; }; \
	command -v $(POETRY) >/dev/null 2>&1 \
		&& printf "  ✅ poetry    %s\n" "$$($(POETRY) --version 2>&1 | sed 's/[^0-9.]//g')" \
		|| { printf "  ❌ poetry    not found — install from https://python-poetry.org\n"; ok=false; }; \
	command -v ffmpeg >/dev/null 2>&1 \
		&& printf "  ✅ ffmpeg    %s\n" "$$(ffmpeg -version 2>&1 | head -1 | awk '{print $$3}')" \
		|| { printf "  ❌ ffmpeg    not found — 'brew install ffmpeg' (macOS) or 'apt-get install -y ffmpeg'\n"; ok=false; }; \
	$$ok || { printf "\n\033[31mEnvironment check failed.\033[0m\n"; exit 1; }; \
	printf "\n\033[32mAll checks passed.\033[0m\n"

.PHONY: install
install: check  ## Verify tools then install all dependencies via poetry
	@printf "\033[1mInstalling dependencies...\033[0m\n"
	$(POETRY) install

.PHONY: lock
lock:  ## Re-resolve and rewrite poetry.lock from pyproject.toml
	$(POETRY) lock

.PHONY: update
update:  ## Update dependencies to the latest allowed versions
	$(POETRY) update

.PHONY: run
run:  ## Show the CLI help (usage: poetry run voice-clone REFERENCE "TEXT" [-o out.wav])
	$(POETRY) run voice-clone --help

.PHONY: clean
clean:  ## Remove caches and build artifacts
	@printf "\033[1mCleaning up...\033[0m\n"
	rm -rf dist .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean  ## clean + remove the poetry virtualenv
	@printf "\033[1mRemoving virtualenv...\033[0m\n"
	$(POETRY) env remove --all 2>/dev/null || true
