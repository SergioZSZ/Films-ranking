# pgAdmin service design

## Objective

Add pgAdmin to the existing Docker Compose development environment so the
PostgreSQL database can be inspected from a local web browser.

## Compose architecture

- Add a `pgadmin` service to `docker-compose.yml` using the official
  `dpage/pgadmin4` image.
- Publish the web interface as `127.0.0.1:5050:80`, keeping it inaccessible
  from other machines by default.
- Configure the login through `PGADMIN_EMAIL` and `PGADMIN_PASSWORD`, with
  development-friendly fallback values in Compose.
- Make pgAdmin depend on the existing `postgres` service reaching its healthy
  state.
- Persist pgAdmin state in a named volume so registered servers and settings
  survive container recreation.

## Database connection

Both services use the default Compose network. From pgAdmin, PostgreSQL is
available at host `postgres`, port `5432`. The published host port `5433` is
only needed by clients running outside Docker.

The PostgreSQL username, password, and database remain controlled by the
existing `DATABASE_USER`, `DATABASE_PASSWORD`, and `DATABASE_NAME` variables.
The pgAdmin service does not duplicate or alter those database settings.

## Validation

- Run `docker compose config` to validate the merged configuration.
- Confirm the rendered pgAdmin port is bound to `127.0.0.1:5050`.
- Confirm the service dependency targets the healthy `postgres` service and
  the pgAdmin data volume is declared.

Starting the containers or logging into pgAdmin is outside this configuration
change unless explicitly requested.
