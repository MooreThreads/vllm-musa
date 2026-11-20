# Build the recipe site from the cookbook markdown.
#
#   make site                         # docs/cookbook from this checkout
#   make site REF=codex/v0240-cookbook # cookbook from another git ref
#   make serve PORT=8000              # build, then serve _site locally
#   make clean                        # remove generated output

PYTHON ?= python3
OUT    ?= _site
PORT   ?= 8000
REF    ?=
SRC    ?=

ifeq ($(strip $(SRC)),)
  ifeq ($(strip $(REF)),)
    BUILD = $(PYTHON) build_site.py --out $(OUT)
  else
    BUILD = $(PYTHON) build_site.py --git-repo ../.. --git-ref $(REF) --ref $(REF) --out $(OUT)
  endif
else
  BUILD = $(PYTHON) build_site.py --source $(SRC) --out $(OUT)
endif

.PHONY: site serve clean

site:
	$(BUILD)

serve: site
	$(PYTHON) -m http.server -d $(OUT) $(PORT)

clean:
	rm -rf $(OUT)
