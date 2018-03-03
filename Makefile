PYTHON ?= python
CYTHON ?= cython

cython:
	find deepdmr -name "*.pyx" -exec $(CYTHON) {} \;
