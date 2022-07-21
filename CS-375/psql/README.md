
## PSQL is Annoying :/

### overwrite the pg_hba.conf to allow for permissions

sudo cp psql/pg_hba.conf /etc/postgresql/12/main/pg_hba.conf

### restart the server

sudo /etc/init.d/postgresql restart

### access psql

psql --user postgres
