"""Set up the Aiven Postgres service: roles, databases, Northwind data and grants.

Run from the project root (needs the POSTGRES_* / AGENT_* values in .env):

    uv run python scripts/init_db.py

Safe to re-run:
- roles and databases are created only if missing, and the role passwords are reset to .env;
- the Northwind dump drops and recreates its tables, and the date shift runs right after it,
  so the dates are shifted exactly once per load.

Aiven's `avnadmin` is not a superuser (G17). It can create roles, but `CREATE DATABASE ...
OWNER agent_app` needs avnadmin to be a member of agent_app, so that grant comes first.
"""

from pathlib import Path

import psycopg
from psycopg import sql

from multi_agenti_conversation_system.config import settings

DB_DIR = Path(__file__).resolve().parent.parent / "db"
ROLES = {
    "agent_app": settings.AGENT_APP_PASSWORD,
    "agent_readonly": settings.AGENT_READONLY_PASSWORD,
}


def _exists(conn: psycopg.Connection, query: str, name: str) -> bool:
    return conn.execute(query, (name,)).fetchone() is not None


def create_roles_and_databases() -> None:
    # CREATE DATABASE can't run inside a transaction, hence autocommit.
    with psycopg.connect(settings.admin_conninfo(), autocommit=True) as conn:
        for role, password in ROLES.items():
            if not password:
                raise SystemExit(f"Set the password for {role} in .env first.")
            verb = (
                "ALTER"
                if _exists(conn, "SELECT 1 FROM pg_roles WHERE rolname = %s", role)
                else "CREATE"
            )
            conn.execute(
                sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
            print(f"role {role}: {'updated' if verb == 'ALTER' else 'created'}")

        admin = sql.Identifier(settings.POSTGRES_ADMIN_USER)
        conn.execute(sql.SQL("GRANT agent_app TO {}").format(admin))

        databases = {settings.NORTHWIND_DB: None, settings.MEMORY_DB: "agent_app"}
        for db, owner in databases.items():
            if _exists(conn, "SELECT 1 FROM pg_database WHERE datname = %s", db):
                print(f"database {db}: exists")
                continue
            statement = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db))
            if owner:
                statement += sql.SQL(" OWNER {}").format(sql.Identifier(owner))
            conn.execute(statement)
            print(f"database {db}: created")

        # Only agent_app may connect to the memory DB (03_grants.sql does the same for northwind).
        memory = sql.Identifier(settings.MEMORY_DB)
        conn.execute(
            sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(memory)
        )
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO agent_app").format(memory)
        )


def load_northwind() -> None:
    with psycopg.connect(settings.admin_conninfo(settings.NORTHWIND_DB)) as conn:
        # One transaction: a failure part-way leaves the previous data in place.
        for name in ("02_northwind.sql", "04_shift_dates.sql", "03_grants.sql"):
            conn.execute((DB_DIR / name).read_text())
            print(f"ran {name}")


def check() -> None:
    with psycopg.connect(settings.northwind_conninfo) as conn:
        orders, first, last = conn.execute(
            "SELECT count(*), min(order_date), max(order_date) FROM orders"
        ).fetchone()
        read_only = conn.execute("SHOW default_transaction_read_only").fetchone()[0]
    print(
        f"agent_readonly sees {orders} orders, {first} .. {last}, read_only={read_only}"
    )
    with psycopg.connect(settings.memory_conninfo) as conn:
        conn.execute("SELECT 1")
    print("agent_app can connect to the memory database")


def main() -> None:
    create_roles_and_databases()
    load_northwind()
    check()
    print("Database setup complete.")


if __name__ == "__main__":
    main()
