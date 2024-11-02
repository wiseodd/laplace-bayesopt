test:
	uv run pytest --cov

ruff:
	-uv run ruff format
	@uv run ruff check --fix

mypy:
	uv run mypy --config-file mypy.ini src

lint:
	make ruff
	make mypy
