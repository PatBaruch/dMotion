.PHONY: setup setup-labeling prepare run diagnose test-money demo doctor sound image fetch collect auto-label label dataset build-dataset train trained check format

setup:
	sh scripts/setup.sh

setup-labeling:
	.tools/bin/uv sync --locked --extra vision --extra labeling --cache-dir "$(CURDIR)/.cache/uv"

prepare:
	.venv/bin/dmotion prepare

run:
	.venv/bin/dmotion run

diagnose:
	.venv/bin/dmotion run --mode check

test-money:
	.venv/bin/dmotion run --prompt "paper money" --confidence 0.1 --image-size 640 --device cpu

fetch:
	.venv/bin/dmotion fetch examples/money-spread-sources.json

collect:
	.venv/bin/dmotion collect

auto-label:
	.venv/bin/dmotion auto-label --engine grounding --confidence 0.2 --prompt banknotes --prompt "dollar bills" --prompt "cash money"

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
	@mkdir -p reports
	.venv/bin/bandit -r src scripts --severity-level medium -f json -o reports/bandit.json
	.venv/bin/pytest --cov --cov-report=term:skip-covered --cov-report=xml:reports/coverage.xml --cov-report=json:reports/coverage.json --junitxml=reports/junit.xml

.PHONY: audit package-check

audit:
	@mkdir -p reports
	.venv/bin/python scripts/locked_requirements.py --output reports/locked-requirements.txt --audit

package-check:
	uv build --no-build-isolation
	.venv/bin/python scripts/package_smoke.py --uv uv

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

.PHONY: setup-workflow finish-feature

setup-workflow:
	sh scripts/setup-workflow.sh

finish-feature:
	.venv/bin/python scripts/finish_feature.py $(ARGS)
