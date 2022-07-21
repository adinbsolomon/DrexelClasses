
:: clean up thoroughly
docker stop cs500
docker container rm cs500
docker image prune --force --filter "label=prune=cs500"

:: build the container
docker build -t cs500 .

:: start the container in the background
docker run --name cs500 -e POSTGRES_PASSWORD=password -d -p 5432:5432 --label "prune=cs500" cs500:latest

:: start an interactive bash terminal in the container
docker exec -it cs500 bash

:: clean up thoroughly
docker stop cs500
docker container rm cs500
docker image prune --force --filter "label=prune=cs500"
