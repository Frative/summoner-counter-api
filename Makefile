build:
	docker build -t summoner-counter-api .

run:
	docker run --rm -p 8000:8000 summoner-counter-api

shell:
	docker run --rm -it -p 8000:8000 summoner-counter-api /bin/bash
