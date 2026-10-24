import pytest

from multi_agenti_conversation_system.guardrails.sql_validator import (
    find_pii_columns,
    validate_select,
)
from multi_agenti_conversation_system.mcp.servers.db_server import catalog

ALLOWED = {"orders", "customers", "products", "order_details"}
MAX_ROWS = 100


def check(sql: str):
    return validate_select(sql, MAX_ROWS, ALLOWED)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "select * from ORDERS",
        'SELECT * FROM "Orders"',
        "SELECT * FROM public.orders",
        "SELECT * FROM orders;",
        "SELECT * FROM orders; -- trailing comment",
        "SELECT o.order_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id",
        "WITH t AS (SELECT * FROM orders) SELECT * FROM t",
        "SELECT * FROM orders UNION SELECT * FROM orders",
        "SELECT * FROM orders WHERE order_id IN (SELECT order_id FROM order_details)",
        "SELECT 1",
    ],
)
def test_accepts_safe_selects(sql):
    result = check(sql)
    assert result.ok, result.error
    assert result.error is None
    assert "LIMIT" in result.sql.upper()


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM orders", "Only SELECT"),
        ("delete from orders", "Only SELECT"),
        ("DeLeTe FROM orders", "Only SELECT"),
        ("UPDATE orders SET freight = 0", "Only SELECT"),
        ("INSERT INTO orders (order_id) VALUES (1)", "Only SELECT"),
        ("DROP TABLE orders", "Only SELECT"),
        ("TRUNCATE orders", "Only SELECT"),
        ("SELECT 1; DROP TABLE orders", "Exactly one statement"),
        ("SELECT * FROM orders; DELETE FROM orders", "Exactly one statement"),
        ("SELECT * FROM orders /* ; */; DELETE FROM orders", "Exactly one statement"),
        (
            "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
            "not allowed",
        ),
        ("SELECT * INTO newtbl FROM orders", "not allowed"),
        ("SELECT * FROM orders FOR UPDATE", "not allowed"),
        ("COPY orders TO '/tmp/x'", "Only SELECT"),
        ("SET default_transaction_read_only = off", "Only SELECT"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT PG_SLEEP(10)", "pg_sleep"),
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT current_setting('server_version')", "current_setting"),
        ("SELECT query_to_xml('delete from orders', true, false, '')", "query_to_xml"),
        ("SELECT lo_import('/etc/passwd')", "lo_import"),
        ("SELECT * FROM pg_catalog.pg_user", "Schema 'pg_catalog'"),
        ("SELECT * FROM information_schema.tables", "Schema 'information_schema'"),
        ("SELECT * FROM generate_series(1, 5)", "Table functions"),
        ("SELECT * FROM secrets", "Table 'secrets'"),
        ("SELECT * FROM orders JOIN secrets ON true", "Table 'secrets'"),
        ("SELECT * FROM orders WHERE x IN (SELECT y FROM secrets)", "Table 'secrets'"),
        ("", "Empty"),
        ("   ", "Empty"),
        (";", "Empty"),
        ("-- just a comment", "Empty"),
        ("SELECT * FROM (", "could not be parsed"),
        ("SELECT FROM WHERE", "could not be parsed"),
        ("SELEC * FORM orders", "Only SELECT"),
    ],
)
def test_rejects_unsafe_input(sql, reason):
    result = check(sql)
    assert not result.ok
    assert result.sql == ""
    assert reason.lower() in result.error.lower()


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM orders", "LIMIT 100"),
        ("SELECT * FROM orders LIMIT 100000", "LIMIT 100"),
        ("SELECT * FROM orders LIMIT ALL", "LIMIT 100"),
        ("SELECT * FROM orders FETCH FIRST 5000 ROWS ONLY", "LIMIT 100"),
        ("SELECT * FROM orders LIMIT 5", "LIMIT 5"),
        ("SELECT * FROM orders UNION SELECT * FROM orders", "LIMIT 100"),
    ],
)
def test_limit_is_injected_or_clamped(sql, expected):
    result = check(sql)
    assert result.ok, result.error
    assert result.sql.endswith(expected)


def test_limit_uses_the_given_max_rows():
    result = validate_select("SELECT * FROM orders", 101, ALLOWED)
    assert result.ok
    assert result.sql.endswith("LIMIT 101")


def test_cte_name_is_not_reported_as_a_table():
    result = check("WITH t AS (SELECT * FROM orders) SELECT * FROM t")
    assert result.ok
    assert result.tables == {"orders"}


def test_tables_are_lowercased_and_collected():
    result = check('SELECT * FROM "Orders" o JOIN public.customers c ON true')
    assert result.tables == {"orders", "customers"}


def test_allowlist_is_case_insensitive():
    result = validate_select("SELECT * FROM orders", MAX_ROWS, {"ORDERS"})
    assert result.ok


PII = {
    "customers": {"contact_name", "phone"},
    "employees": {"first_name", "last_name"},
}


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT order_id, freight FROM orders", []),
        ("SELECT company_name FROM customers", []),
        ("SELECT contact_name FROM customers", ["customers.contact_name"]),
        ("SELECT c.contact_name FROM customers c", ["customers.contact_name"]),
        ("SELECT CONTACT_NAME FROM Customers", ["customers.contact_name"]),
        ("SELECT * FROM customers", ["customers.contact_name", "customers.phone"]),
        ("SELECT c.* FROM customers c", ["customers.contact_name", "customers.phone"]),
        ("SELECT * FROM orders", []),
        ("SELECT COUNT(*) FROM customers", []),
        (
            "SELECT company_name FROM customers WHERE phone = '1'",
            ["customers.phone"],
        ),
        (
            "SELECT c.company_name FROM customers c JOIN orders o ON o.customer_id = c.customer_id",
            [],
        ),
        (
            "SELECT e.first_name FROM employees e JOIN orders o ON o.employee_id = e.employee_id",
            ["employees.first_name"],
        ),
        # o.* expands to orders columns only, so the PII table next to it is not touched
        (
            "SELECT o.* FROM orders o JOIN customers c ON c.customer_id = o.customer_id",
            [],
        ),
        # the same alias `c` means different tables in different scopes
        (
            (
                "SELECT c.phone FROM customers c WHERE EXISTS "
                "(SELECT 1 FROM categories c WHERE c.category_id = 1)"
            ),
            ["customers.phone"],
        ),
        # PII hidden behind a CTE or a subquery
        (
            "WITH t AS (SELECT * FROM customers) SELECT * FROM t",
            ["customers.contact_name", "customers.phone"],
        ),
        (
            "SELECT * FROM orders WHERE customer_id IN (SELECT contact_name FROM customers)",
            ["customers.contact_name"],
        ),
        # a table that is not in the PII map at all
        ("SELECT phone FROM shippers", []),
    ],
)
def test_find_pii_columns(sql, expected):
    assert find_pii_columns(sql, PII) == expected


def test_pii_detection_matches_the_few_shot_pii_flags():
    """The 20 seed examples were hand-tagged `pii: true`; the detector must agree."""
    pii = catalog.pii_columns()
    for example in catalog.few_shots():
        flagged = bool(find_pii_columns(example["sql"], pii))
        assert flagged == example.get("pii", False), example["question"]
