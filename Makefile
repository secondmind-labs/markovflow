.PHONY: docs

SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)

PANDOC_DEB = https://github.com/jgm/pandoc/releases/download/2.10.1/pandoc-2.10.1-1-amd64.deb



docs:  ## Build the documentation
	@echo "\n=== pip install doc requirements =============="
	pip install -r docs/docs_requirements.txt
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
