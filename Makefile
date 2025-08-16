build:
	docker build -t summoner-counter-api .

run:
	docker run -d -p 8000:8000 --name summoner-counter-api --restart unless-stopped summoner-counter-api

shell:
	docker run --rm -it -p 8000:8000 summoner-counter-api /bin/bash

stop:
	docker stop summoner-counter-api