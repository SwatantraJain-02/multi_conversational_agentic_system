"""AST-based SQL guardrail: accepts exactly one read-only SELECT and enforces a row limit.

Pure functions, no database access. Used by the agent (layer 1) and again by the DB MCP
server (layer 2), so nothing the LLM writes is ever executed unchecked.
"""

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

DIALECT = "postgres"

# Anything below the root that is not a plain read. Walking the whole tree (not just the
# root) is what catches DML hidden inside a CTE, e.g. WITH d AS (DELETE ... RETURNING *) SELECT ...
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Set,
    exp.TruncateTable,
    exp.Into,  # SELECT ... INTO new_table
    exp.Lock,  # SELECT ... FOR UPDATE
)

# Functions sqlglot does not know are parsed as exp.Anonymous, so we can match on the name.
FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_")
FORBIDDEN_FUNCTIONS = {
    "dblink",
    "set_config",
    "current_setting",
    "query_to_xml",
    "query_to_xml_and_xmlschema",
    "query_to_xmlschema",
    "table_to_xml",
    "database_to_xml",
    "schema_to_xml",
    "cursor_to_xml",
    "txid_current",
}

ALLOWED_SCHEMAS = {"", "public"}


@dataclass
class ValidationResult:
    ok: bool
    sql: str = ""  # cleaned query with the limit applied; empty when rejected
    error: str | None = None
    tables: set[str] = field(default_factory=set)


def _reject(error: str) -> ValidationResult:
    return ValidationResult(ok=False, error=error)


def _check_forbidden_nodes(tree: exp.Expr) -> str | None:
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return f"{type(node).__name__} statements are not allowed; only SELECT is permitted"
    return None


def _check_functions(tree: exp.Expr) -> str | None:
    for func in tree.find_all(exp.Anonymous):
        name = str(func.name).lower()
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            return f"Function '{name}' is not allowed"
    return None


def _check_tables(
    tree: exp.Expr, allowed_tables: set[str]
) -> tuple[set[str], str | None]:
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    used: set[str] = set()

    for table in tree.find_all(exp.Table):
        # A non-identifier source is a table function, e.g. generate_series(1, 5).
        if not isinstance(table.this, exp.Identifier):
            return used, "Table functions are not allowed"

        name = table.name.lower()
        schema = (table.db or "").lower()
        if schema not in ALLOWED_SCHEMAS:
            return used, f"Schema '{schema}' is not allowed"
        if not schema and name in cte_names:
            continue  # a reference to a CTE defined in this query, not a real table
        if name not in allowed:
            return used, f"Table '{name}' is not allowed"
        used.add(name)

    return used, None


def _apply_limit(tree: exp.Query, max_rows: int) -> exp.Query:
    limit = tree.args.get("limit")
    if isinstance(limit, exp.Limit):
        value = limit.expression
        if (
            isinstance(value, exp.Literal)
            and value.is_int
            and int(value.name) <= max_rows
        ):
            return tree  # the query already asks for fewer rows than the cap
    # No limit, too large a limit, LIMIT ALL, or FETCH FIRST: replace with the cap.
    return tree.limit(max_rows)


def validate_select(
    sql: str, max_rows: int, allowed_tables: set[str]
) -> ValidationResult:
    """Return ok=True with a safe, limited query, or ok=False with a reason."""
    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except SqlglotError as exc:
        return _reject(f"SQL could not be parsed: {str(exc).splitlines()[0]}")

    # Empty input, comments and a trailing ';' produce None entries, or an exp.Semicolon
    # node when a comment follows the ';'. Neither is a real statement.
    statements = [
        s for s in statements if s is not None and not isinstance(s, exp.Semicolon)
    ]
    if len(statements) == 0:
        return _reject("Empty query")
    if len(statements) > 1:
        return _reject("Exactly one statement is allowed")

    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        return _reject(f"Only SELECT queries are allowed (got {type(tree).__name__})")

    if error := _check_forbidden_nodes(tree):
        return _reject(error)
    if error := _check_functions(tree):
        return _reject(error)

    tables, error = _check_tables(tree, allowed_tables)
    if error:
        return _reject(error)

    tree = _apply_limit(tree, max_rows)
    return ValidationResult(ok=True, sql=tree.sql(dialect=DIALECT), tables=tables)


def _real_tables_in_scope(select: exp.Select, cte_names: set[str]) -> list[exp.Table]:
    """Tables read directly by this SELECT (not by a nested subquery or CTE body)."""
    return [
        t
        for t in select.find_all(exp.Table)
        if t.find_ancestor(exp.Select) is select
        and (t.db or t.name.lower() not in cte_names)
    ]


def _enclosing_selects(node: exp.Expr):
    """The SELECT a node sits in, then the SELECTs around it (for correlated subqueries)."""
    select = node.find_ancestor(exp.Select)
    while select is not None:
        yield select
        select = select.find_ancestor(exp.Select)


def find_pii_columns(sql: str, pii_columns: dict[str, set[str]]) -> list[str]:
    """Return the PII columns a query touches, as sorted 'table.column' strings.

    `pii_columns` is {table: {column, ...}} in lower case (see catalog.pii_columns()).
    Call this on SQL that already passed validate_select(); unparseable SQL raises.
    The check errs on the side of flagging: an unqualified column is matched against every
    table in scope, and `SELECT *` / `t.*` count as touching every PII column of the tables.
    """
    tree = sqlglot.parse_one(sql, dialect=DIALECT)
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    found: set[str] = set()

    def flag(table: str, columns: set[str]) -> None:
        found.update(f"{table}.{c}" for c in columns)

    for node in tree.walk():
        if isinstance(node, exp.Star):
            # COUNT(*) is not a column read; only SELECT * and t.* expose columns.
            if isinstance(node.parent, exp.Select):
                qualifier = None
            elif isinstance(node.parent, exp.Column):
                qualifier = node.parent.table.lower()
            else:
                continue
            select = node.find_ancestor(exp.Select)
            if select is None:
                continue
            for table in _real_tables_in_scope(select, cte_names):
                name = table.name.lower()
                if qualifier is None or qualifier in (
                    name,
                    table.alias_or_name.lower(),
                ):
                    flag(name, pii_columns.get(name, set()))
        elif isinstance(node, exp.Column):
            column = node.name.lower()
            qualifier = node.table.lower()
            for select in _enclosing_selects(node):
                in_scope = _real_tables_in_scope(select, cte_names)
                if qualifier:
                    matches = [
                        t
                        for t in in_scope
                        if qualifier in (t.name.lower(), t.alias_or_name.lower())
                    ]
                else:
                    matches = in_scope
                for table in matches:
                    name = table.name.lower()
                    if column in pii_columns.get(name, set()):
                        found.add(f"{name}.{column}")
                if qualifier and matches:
                    break  # qualifier resolved in the innermost scope that defines it

    return sorted(found)
