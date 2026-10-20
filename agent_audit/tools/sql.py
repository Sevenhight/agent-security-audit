"""Read-only SQLite tool bound to a sandbox database."""

from __future__ import annotations

import os
import sqlite3

MAX_ROWS = 20

# Statements that would let a model escalate beyond a read. Checked before the
# connection is opened so a denial never touches the file.
FORBIDDEN_PREFIXES = ("insert", "update", "delete", "drop", "alter", "attach", "pragma")


def sql_query(workspace: str, sql: str) -> str:
    statement = sql.strip()
    lowered = statement.lower()

    if not lowered.startswith("select"):
        return "[sql_query] rejected: only SELECT statements are allowed"
    if any(keyword in lowered for keyword in FORBIDDEN_PREFIXES):
        # Chained statements such as "select 1; drop table x" land here.
        return "[sql_query] rejected: statement contains a forbidden keyword"

    db_path = os.path.join(workspace, "demo.db")
    if not os.path.isfile(db_path):
        return f"[sql_query] no such database: demo.db"

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as exc:
        return f"[sql_query] connection error: {exc}"

    try:
        cursor = connection.execute(statement)
        rows = cursor.fetchmany(MAX_ROWS)
        columns = [description[0] for description in cursor.description or []]
    except sqlite3.Error as exc:
        return f"[sql_query] error: {exc}"
    finally:
        connection.close()

    if not rows:
        return f"[sql_query] 0 rows; columns={columns}"
    rendered = "\n".join(str(dict(zip(columns, row))) for row in rows)
    return f"[sql_query] {len(rows)} rows; columns={columns}\n{rendered}"
