PYTHON ?= python

up:
	docker compose up -d

down:
	docker compose down

ingest:
	$(PYTHON) -m app.ingest

retrieve:
	$(PYTHON) -m app.retrieve --query "$(QUERY)"

chat:
	$(PYTHON) -m app.chat --question "$(QUESTION)"
