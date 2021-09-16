# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

LIB_NAME = markovflow
SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)

PANDOC_DEB = https://github.com/jgm/pandoc/releases/download/2.10.1/pandoc-2.10.1-1-amd64.deb

# the --per-file-ignores are to ignore "unused import" warnings in __init__.py files (F401)
# the F403 ignore in gpflux/__init__.py allows the `from .<submodule> import *`
LINT_FILE_IGNORES = "$(LIB_NAME)/__init__.py:F401,F403 \
                     $(LIB_NAME)/architectures/__init__.py:F401 \
                     $(LIB_NAME)/encoders/__init__.py:F401 \
                     $(LIB_NAME)/experiment_support/__init__.py:F401 \
                     $(LIB_NAME)/initializers/__init__.py:F401 \
                     $(LIB_NAME)/layers/__init__.py:F401 \
                     $(LIB_NAME)/layers/basis_functions/__init__.py:F401 \
                     $(LIB_NAME)/models/__init__.py:F401 \
                     $(LIB_NAME)/optimization/__init__.py:F401 \
                     $(LIB_NAME)/sampling/__init__.py:F401 \
                     $(LIB_NAME)/utils/__init__.py:F401"




.PHONY: help docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).



docs:  ## Build the documentation
	@echo "\n=== pip install doc requirements =============="
	pip install -r docs_requirements.txt
	@echo "\n=== install pandoc =============="
ifeq ("$(UNAME_S)", "Linux")
	$(eval TEMP_DEB=$(shell mktemp))
	@echo "Checking for pandoc installation..."
	@(which pandoc) || ( echo "\nPandoc not found." \
	  && echo "Trying to install automatically...\n" \
	  && wget -O "$(TEMP_DEB)" $(PANDOC_DEB) \
	  && echo "\nInstalling pandoc using dpkg -i from $(PANDOC_DEB)" \
	  && echo "(If this step does not work, manually install pandoc, see http://pandoc.org/)\n" \
	  && sudo dpkg -i "$(TEMP_DEB)" \
	)
	@rm -f "$(TEMP_DEB)"
endif
ifeq ($(UNAME_S),Darwin)
	brew install pandoc
endif
	# Create a .py (percent format with multiline comments) from an .ipynb in the first place:
	# jupytext --update-metadata '{"jupytext": {"cell_markers": "\"\"\""}}' --to py:percent <name-of-notebook>.ipynb
	# Convert .py to .ipynb (don't seem to need to --execute? perhaps sphinx-build does it...), then build html:
	rm -rf _build
	rm -rf _autosummary
	find . -name 'notebooks/*.py' -not -iname '__init__.py' | xargs jupytext --to notebook
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
