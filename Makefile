.PHONY: all test clean
all:
	PYTHONPATH=src python -m spatialrl.cli all
test:
	PYTHONPATH=src python -m pytest -q tests
clean:
	rm -rf results/*
