#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a

PSQL="docker compose exec -T postgres psql -U postgres -q -v ON_ERROR_STOP=1"

$PSQL -v app_pw="$AGENT_APP_PASSWORD" -v ro_pw="$AGENT_READONLY_PASSWORD" -f /db/01_databases.sql
$PSQL -d northwind -f /db/02_northwind.sql
$PSQL -d northwind -f /db/04_shift_dates.sql
$PSQL -d northwind -f /db/03_grants.sql
echo "Database setup complete."

