# pgAdmin Compose Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent, locally exposed pgAdmin service to the existing PostgreSQL Docker Compose environment.

**Architecture:** pgAdmin runs as a second service on Compose's default network and reaches PostgreSQL at `postgres:5432`. Its web interface is bound only to `127.0.0.1:5050`, while a named volume preserves pgAdmin settings across container recreation.

**Tech Stack:** Docker Compose, PostgreSQL 16, pgAdmin 4

---

### Task 1: Add and validate the pgAdmin service

**Files:**
- Modify: `docker-compose.yml`
- Reference: `docs/superpowers/specs/2026-10-29-pgadmin-compose-design.md`

- [x] **Step 1: Verify that the current Compose model has no pgAdmin service**

```powershell
$compose = docker compose config --format json | ConvertFrom-Json
if ($compose.services.PSObject.Properties.Name -contains 'pgadmin') { throw 'pgadmin unexpectedly exists' }
```

Expected: the assertion emits no output and exits successfully.

- [x] **Step 2: Add the pgAdmin service and persistent volume**

Add this service immediately after the existing `postgres` service in `docker-compose.yml`:

```yaml
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: api-proxy-pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL:-admin@example.com}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}
    ports:
      - "127.0.0.1:5050:80"
    volumes:
      - pgadmin-data:/var/lib/pgadmin
    depends_on:
      postgres:
        condition: service_healthy
```

Declare the named volume alongside the existing volume declaration:

```yaml
volumes:
  postgres-data:
  pgadmin-data:
```

- [x] **Step 3: Validate the rendered Compose model**

```powershell
docker compose config --quiet
$compose = docker compose config --format json | ConvertFrom-Json
if (-not ($compose.services.PSObject.Properties.Name -contains 'pgadmin')) { throw 'pgadmin service missing' }
if ($compose.services.pgadmin.ports[0].host_ip -ne '127.0.0.1') { throw 'pgAdmin is not restricted to localhost' }
if ([string]$compose.services.pgadmin.ports[0].published -ne '5050') { throw 'pgAdmin host port is not 5050' }
if ($compose.services.pgadmin.depends_on.postgres.condition -ne 'service_healthy') { throw 'pgAdmin does not wait for healthy PostgreSQL' }
if (-not ($compose.volumes.PSObject.Properties.Name -contains 'pgadmin-data')) { throw 'pgAdmin volume missing' }
```

Expected: both commands exit successfully. The assertions emit no output.

- [x] **Step 4: Check the patch for formatting errors and unintended changes**

```powershell
git diff --check -- docker-compose.yml
git diff -- docker-compose.yml
```

Expected: `git diff --check` emits no output. The diff contains only the new `pgadmin` service and `pgadmin-data` volume.

- [x] **Step 5: Commit the Compose change**

```powershell
git add -- docker-compose.yml docs/superpowers/plans/2026-10-29-pgadmin-compose.md
git commit -m "feat: add pgAdmin compose service"
```

Expected: Git creates one commit containing the Compose service and this implementation plan without staging unrelated working-tree changes.
