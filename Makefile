.PHONY: setup prepare run diagnose demo doctor sound image fetch collect label dataset build-dataset train trained check format

setup:
	sh scripts/setup.sh

prepare:
	.venv/bin/dmotion prepare

run:
	.venv/bin/dmotion run

diagnose:
	.venv/bin/dmotion run --mode check

fetch:
	.venv/bin/dmotion fetch examples/money-spread-sources.json

collect:
	.venv/bin/dmotion collect

label:
	.venv/bin/dmotion label

dataset:
	.venv/bin/dmotion dataset

build-dataset:
	.venv/bin/dmotion build-dataset

train:
	.venv/bin/dmotion train

trained:
	.venv/bin/dmotion run --mode trained

demo:
	.venv/bin/dmotion run --demo

doctor:
	.venv/bin/dmotion doctor

sound:
	.venv/bin/dmotion sound

image:
	@test -n "$(IMAGE)" || (echo 'Usage: make image IMAGE="data/your-photo.jpg"'; exit 1)
	.venv/bin/dmotion image "$(IMAGE)"

check:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/pytest

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .
