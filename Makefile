.PHONY: help install train play dashboard benchmark test

help:
	@echo "Small Models Matter"
	@echo ""
	@echo "  make install     install Python dependencies"
	@echo "  make train       train for 500 episodes with the pygame window"
	@echo "  make play        play the best saved model"
	@echo "  make dashboard   open the live network dashboard"
	@echo "  make benchmark   reproduce the latency numbers"
	@echo "  make test        run the smoke tests"

install:
	python3 -m pip install -r requirements.txt

train:
	python3 src/snake_ai.py train --episodes 500 --render

play:
	python3 src/snake_ai.py play

dashboard:
	python3 src/neuron_watch.py --port 8765

benchmark:
	python3 scripts/benchmark.py

test:
	python3 -m pytest -q
