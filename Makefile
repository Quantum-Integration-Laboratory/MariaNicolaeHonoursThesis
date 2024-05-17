# uses GNU extensions

LATEX_SRC_DIR = latex-src
LATEX_BUILD_DIR = latex-build
FIGURES_DIR = figures
CODE_DIR = code
LATEX_BUILD_CMD = cd $(LATEX_BUILD_DIR); pdflatex
BIBTEX_BUILD_CMD = cd $(LATEX_BUILD_DIR); biber

LATEX_SRC_MAIN = thesis
LATEX_SRC_SUBS = titlepage \
                 frontmatter \
                 introduction \
                 prior-transduction \
                 four-level-transduction \
                 biphoton-generation \
                 conclusion \
                 barnett-longdell-reverse \
                 efficiency-fit \
                 printed-code
LATEX_SRC_FILES = $(LATEX_SRC_MAIN) $(LATEX_SRC_SUBS)
LATEX_SRC_PATHS_BUILD = $(LATEX_SRC_FILES:%=$(LATEX_BUILD_DIR)/%.tex)
LATEX_SRC_BIB = $(LATEX_SRC_DIR)/$(LATEX_SRC_MAIN).bib
LATEX_BUILD_BIB = $(LATEX_BUILD_DIR)/$(LATEX_SRC_MAIN).bib

GENERATED_FIGURES = inhomogeneous-broadening.pgf \
                    3lt-phase.png \
                    4lt-example-scan.png \
                    trench-duplication.pgf \
                    4lt-pixel-intersections.png \
                    4lt-model-scan.png \
                    4lt-scan-hyperbolas.png \
                    4lt-hybridisation-ratio.png \
                    biphoton-results-small.png \
                    biphoton-results-large.png \
                    3lt-replication.png \
                    4lt-power-calibration.png
FIGURES = quantum-networking.tikz \
          chi-2.tikz optomechanics.tikz \
          hybrid-atomic-system.tikz \
          lambda-and-v-systems.tikz \
          atoms-in-cavity.tikz \
          4lt-device.png \
          four-level-diagram.tikz \
          feature-finding.tikz \
          neighbourhood-integration.tikz
GENERATED_FIGURES_PATHS_BUILD = $(GENERATED_FIGURES:%=$(LATEX_BUILD_DIR)/%)
FIGURES_PATHS_BUILD = $(FIGURES:%=$(LATEX_BUILD_DIR)/%)

PRINTED_CODE = biphoton_steady_state.py \
               biphoton_super_atom.cu \
               flt_model.py \
               tlt_double_cavity.py \
               tlt_single_cavity.py
PRINTED_CODE_PATHS_BUILD = $(PRINTED_CODE:%=$(LATEX_BUILD_DIR)/%)

all: thesis.pdf
clean:
	rm -f thesis.pdf
	rm -f $(LATEX_BUILD_DIR)/*

thesis.pdf: $(LATEX_BUILD_DIR)/thesis.pdf
	cp $< $@

$(LATEX_BUILD_DIR)/thesis.pdf: $(LATEX_SRC_PATHS_BUILD) $(LATEX_BUILD_BIB) \
        $(FIGURES_PATHS_BUILD) $(GENERATED_FIGURES_PATHS_BUILD) \
        $(PRINTED_CODE_PATHS_BUILD) $(LATEX_BUILD_DIR)/signature.png
	$(LATEX_BUILD_CMD) $(LATEX_SRC_MAIN)
	$(BIBTEX_BUILD_CMD) $(LATEX_SRC_MAIN)
	$(LATEX_BUILD_CMD) $(LATEX_SRC_MAIN)
	$(LATEX_BUILD_CMD) $(LATEX_SRC_MAIN)

$(LATEX_SRC_PATHS_BUILD): $(LATEX_BUILD_DIR)/%: $(LATEX_SRC_DIR)/%
	cp $< $@

$(LATEX_BUILD_BIB): $(LATEX_SRC_BIB)
	cp $< $@

$(FIGURES_PATHS_BUILD): $(LATEX_BUILD_DIR)/%: $(FIGURES_DIR)/%
	cp $< $@

$(PRINTED_CODE_PATHS_BUILD): $(LATEX_BUILD_DIR)/%: $(CODE_DIR)/%
	cp $< $@

$(LATEX_BUILD_DIR)/inhomogeneous-broadening.pgf: \
        $(CODE_DIR)/inhomogeneous_figure.py
	python $<

$(LATEX_BUILD_DIR)/3lt-phase.png: \
        $(CODE_DIR)/tlt_phase_plot.py
	python $<

$(LATEX_BUILD_DIR)/4lt-example-scan.png: \
        $(CODE_DIR)/flt_example_scan.py
	python $<

$(LATEX_BUILD_DIR)/4lt-pixel-intersections.png: \
       $(CODE_DIR)/flt_pixel_intersections.py
	python $<

$(LATEX_BUILD_DIR)/trench-duplication.pgf: \
        $(CODE_DIR)/trench_duplication_plot.py
	python $<

$(LATEX_BUILD_DIR)/4lt-power-calibration.png: \
        $(CODE_DIR)/flt_power_calibration.py
	python $<

$(CODE_DIR)/flt_pixel_intersections.py: $(CODE_DIR)/flt_model.py
	touch $@

$(LATEX_BUILD_DIR)/biphoton-results-small.png \
$(LATEX_BUILD_DIR)/biphoton-results-large.png &: \
        $(CODE_DIR)/biphoton_results_plots.py
	python $<

$(LATEX_BUILD_DIR)/4lt-model-scan.png: $(CODE_DIR)/flt_model_scan.py
	python $<

$(CODE_DIR)/flt_model_scan.py: $(CODE_DIR)/flt_model.py \
        $(CODE_DIR)/flt_experiment_parameters.py
	touch $@

$(CODE_DIR)/flt_experiment_parameters.py: \
        $(CODE_DIR)/YbYVO_spin_hamiltonian.py \
        $(CODE_DIR)/flt_power_calibration.py
	touch $@

$(LATEX_BUILD_DIR)/4lt-scan-hyperbolas.png: $(CODE_DIR)/flt_scan_hyperbolas.py
	python $<

$(LATEX_BUILD_DIR)/4lt-hybridisation-ratio.png: \
        $(CODE_DIR)/YbYVO_hybridisation_ratio.py
	python $<

$(CODE_DIR)/YbYVO_hybridisation_ratio.py: $(CODE_DIR)/YbYVO_spin_hamiltonian.py
	touch $@

$(LATEX_BUILD_DIR)/3lt-replication.png: $(CODE_DIR)/tlt_plot_replication.py
	python $<

$(CODE_DIR)/tlt_plot_replication.py: $(CODE_DIR)/tlt_double_cavity.py
	touch $@

$(LATEX_BUILD_DIR)/signature.png: signature.png
	cp $< $@
