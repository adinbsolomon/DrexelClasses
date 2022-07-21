# CS500 - Fundamentals of Databases

## How to Use This Repo (Only for Windows)

Assuming you have docker installed, you can simple run the `go.bat` batch script:
```shell
go
```

This will build the docker container, open it in interactive mode, then, once closed, will clean up after itself.

To initialize your project, see the `start` directory; in there are scripts for intiializing databases. To start your project within the container, simply use:
```bash
start/a2 # a2 is a placeholder for a script in the start directory
```

The example scripts initialize the database, then open the database in interactive mode with psql.

