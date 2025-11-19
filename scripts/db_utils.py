"""
Reusable Postgres utility with connection pooling, safe parameterized queries,
and convenience helpers.

Env vars used (from .env):
  - DB_NAME
  - DB_USER
  - DB_PASSWORD
  - DB_HOST
  - DB_PORT

Example:
    db = PostgresDB(minconn=1, maxconn=5)
    rows = db.execute_query("SELECT 1 AS ok")
    print(rows)  # [{'ok': 1}]

    # Non-query (INSERT/UPDATE/DELETE)
    affected = db.execute_non_query("UPDATE my_table SET updated_at = NOW() WHERE id = %s", (123,))
    print(affected)

    # Transaction usage
    with db.transaction() as cur:
        cur.execute("INSERT INTO items(name) VALUES (%s) RETURNING id", ("item-1",))
        new_id = cur.fetchone()["id"]
        cur.execute("UPDATE items SET name = %s WHERE id = %s", ("item-1b", new_id))

    db.close()
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor


class PostgresDB:
    """
    Simple Postgres utility with connection pooling, safe parameterized queries,
    and convenience helpers.

    Env vars used (from .env):
      - DB_NAME
      - DB_USER
      - DB_PASSWORD
      - DB_HOST
      - DB_PORT

    Example:
        db = PostgresDB(minconn=1, maxconn=5)
        rows = db.execute_query("SELECT 1 AS ok")
        print(rows)  # [{'ok': 1}]

        # Non-query (INSERT/UPDATE/DELETE)
        affected = db.execute_non_query("UPDATE my_table SET updated_at = NOW() WHERE id = %s", (123,))
        print(affected)

        # Transaction usage
        with db.transaction() as cur:
            cur.execute("INSERT INTO items(name) VALUES (%s) RETURNING id", ("item-1",))
            new_id = cur.fetchone()["id"]
            cur.execute("UPDATE items SET name = %s WHERE id = %s", ("item-1b", new_id))

        db.close()
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        minconn: int = 1,
        maxconn: int = 5,
        cursor_factory=RealDictCursor,
        autocommit: bool = False,
    ) -> None:
        load_dotenv()
        self.cursor_factory = cursor_factory
        self.autocommit = autocommit

        if dsn is None:
            dbname = os.getenv("DB_NAME")
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            dsn = (
                f"dbname={dbname} user={user} password={password} "
                f"host={host} port={port}"
            )

        if not dsn:
            raise ValueError(
                "Postgres DSN is not configured. Check your .env variables."
            )

        self.pool = SimpleConnectionPool(minconn, maxconn, dsn=dsn)

    def close(self) -> None:
        if hasattr(self, "pool") and self.pool:
            self.pool.closeall()

    def _get_conn(self):
        conn = self.pool.getconn()
        conn.autocommit = self.autocommit
        return conn

    def _put_conn(self, conn) -> None:
        self.pool.putconn(conn)

    @contextmanager
    def connection(self):
        conn = self._get_conn()
        try:
            yield conn
        finally:
            self._put_conn(conn)

    @contextmanager
    def cursor(self):
        with self.connection() as conn:
            with conn.cursor(cursor_factory=self.cursor_factory) as cur:
                yield cur
                if not self.autocommit:
                    conn.commit()

    @contextmanager
    def transaction(self):
        """
        Transaction context manager yielding a cursor. Commits on success,
        rolls back on exception.
        """
        with self.connection() as conn:
            try:
                cur = conn.cursor(cursor_factory=self.cursor_factory)
                try:
                    yield cur
                    conn.commit()
                finally:
                    cur.close()
            except Exception:
                conn.rollback()
                raise

    def execute_query(
        self,
        sql: str,
        params: Optional[Union[Sequence[Any], Dict[str, Any]]] = None,
        *,
        fetch: str = "all",  # "all" | "one" | "val" | "none"
    ) -> Union[List[dict[str, Any]], dict[str, Any], Any, None]:
        """
        Execute a SELECT and fetch results.

        - fetch="all": returns List[Dict]
        - fetch="one": returns Dict or None
        - fetch="val": returns first column of first row or None
        - fetch="none": executes without fetching (returns None)
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            if fetch == "none":
                return None
            if fetch == "one":
                row = cur.fetchone()
                return row
            if fetch == "val":
                row = cur.fetchone()
                if row is None:
                    return None
                # RealDictCursor returns dict; get first column value
                return next(iter(row.values())) if isinstance(row, dict) else row[0]
            # default: all
            rows = cur.fetchall()
            return rows

    def execute_non_query(
        self,
        sql: str,
        params: Optional[Union[Sequence[Any], Dict[str, Any]]] = None,
    ) -> int:
        """Execute INSERT/UPDATE/DELETE. Returns affected row count."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def execute_many(
        self,
        sql: str,
        seq_of_params: Iterable[Union[Sequence[Any], Dict[str, Any]]],
    ) -> int:
        """Execute the same statement for many parameter sets. Returns total affected rows."""
        total = 0
        with self.cursor() as cur:
            for params in seq_of_params:
                cur.execute(sql, params)
                total += cur.rowcount
        return total

    def ping(self) -> bool:
        try:
            self.execute_query("SELECT 1", fetch="val")
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Minimal self-test. Ensure your Postgres is running and .env is configured.
    db = PostgresDB(minconn=1, maxconn=2)
    ok = db.ping()
    print({"postgres_ping": ok})
    db.close()
