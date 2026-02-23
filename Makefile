.PHONY: install test lint clean build publish all

install:
	pip install --upgrade pip
	pip install -e .[dev]

test:
	pytest tests/ -v --cov=chronos

lint:
	flake8 chronos tests

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

build: clean
	python -m build

publish: build
	twine upload dist/*

all: install lint test
