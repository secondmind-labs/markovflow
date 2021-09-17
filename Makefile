.PHONY: help install docs

SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)

PANDOC_DEB = https://github.com/jgm/pandoc/releases/download/2.10.1/pandoc-2.10.1-1-amd64.deb

help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with dev requirements =============="
	pip install --upgrade --upgrade-strategy eager \
        -r requirements.txt \
		-r notebook_requirements.txt \
		-r tests_requirements.txt \
		-e .


docs:  ## Build the documentation
	@echo "\n=== pip install doc requirements =============="
	pip install -r docs/docs_requirements.txt
	pip install --upgrade "Jinja2<3"
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
		@echo "\n=== build docs =============="
		(cd docs ; make html)
		@echo "\n${SUCCESS}=== Docs are available at docs/_build/html/index.html ============== ${SUCCESS}"