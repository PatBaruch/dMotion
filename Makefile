.PHONY: setup prepare run demo doctor sound image fetch collect auto-label label dataset build-dataset train evaluate trained check format audit package-check organize-media setup-workflow finish-feature

setup:
	sh scripts/setup.sh

prepare:
	.venv/bin/dmotion prepare $(ARGS)

run trained:
	.venv/bin/dmotion run $(ARGS)

demo:
	.venv/bin/dmotion run --demo $(ARGS)

doctor:
	.venv/bin/dmotion doctor

sound:
	.venv/bin/dmotion sound

image:
	@test -n "$(IMAGE)" || (echo 'Usage: make image IMAGE="data/photos/cash.jpg"'; exit 1)
	.venv/bin/dmotion image "$(IMAGE)" $(ARGS)

fetch:
	.venv/bin/dmotion fetch examples/money-spread-sources.json

collect:
	.venv/bin/dmotion collect $(ARGS)

auto-label:
	.venv/bin/dmotion auto-label --confidence 0.2 $(ARGS)

label:
	.venv/bin/dmotion label $(ARGS)

dataset:
	.venv/bin/dmotion dataset $(ARGS)

build-dataset:
	.venv/bin/dmotion build-dataset $(ARGS)

train:
	.venv/bin/dmotion train $(ARGS)

evaluate:
	.venv/bin/dmotion evaluate $(ARGS)

check:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	@mkdir -p reports
	.venv/bin/bandit -r src scripts --severity-level medium -f json -o reports/bandit.json
	.venv/bin/pytest --cov --cov-report=term:skip-covered --cov-report=xml:reports/coverage.xml --cov-report=json:reports/coverage.json --junitxml=reports/junit.xml

audit:
	@mkdir -p reports
	.venv/bin/python scripts/locked_requirements.py --output reports/locked-requirements.txt --audit

package-check:
	uv build --no-build-isolation
	.venv/bin/python scripts/package_smoke.py --uv uv

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

organize-media:
	.venv/bin/python scripts/organize_media.py $(ARGS)

setup-workflow:
	sh scripts/setup-workflow.sh

finish-feature:
	.venv/bin/python scripts/finish_feature.py $(ARGS)
