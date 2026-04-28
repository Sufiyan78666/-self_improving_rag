# Makefile for Self-Improving RAG system

.PHONY: install ui train test clean

install:
	pip install -r requirements.txt

ui:
	python run.py ui

train:
	python run.py train

test:
	python -m pytest tests/

clean:
	rm -rf data/chunks/ data/uploads/ rag_system.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
