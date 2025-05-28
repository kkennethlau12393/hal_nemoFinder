"""Initial schema baseline.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-13

This migration brings up the entire hal-nemoFinder schema as it stands
at the 0.1.0 release: analysis/claim/verification tables, the report
aggregate, batch jobs, the multi-tenant auth tables (tenants, users,
api_keys), and the HMAC-chained audit log.  The pgvector extension is
installed first because downstream embedding columns depend on it.

Because SQLAlchemy already knows how to emit dialect-appropriate DDL
for every one of these tables, the migration delegates to
``Base.metadata.create_all`` for the bulk of the work instead of
re-declaring every column in Alembic's DSL.  This keeps the baseline
aligned with the ORM (so a stray ``nullable=`` edit in a model can't
silently drift from the migration) and preserves SQLite compatibility
for the test suite — SQLAlchemy automatically skips JSONB→TEXT,
pgvector, and ``CREATE EXTENSION`` statements when running against
SQLite.

Subsequent migrations should use the normal ``op.create_table`` /
``op.add_column`` surface; this file is deliberately the only
autogenerate-style bulk-create baseline.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


#: Tables this baseline is responsible for, in creation order.  Listed
#: explicitly so ``downgrade()`` can drop them in reverse without
#: accidentally touching future tables from later migrations.
_TABLES_IN_CREATION_ORDER: tuple[str, ...] = (
    "tenants",
    "users",
    "api_keys",
    "batch_jobs",
    "analysis_jobs",
    "claims",
    "verification_results",
    "evidence_records",
    "reports",
    "audit_log",
)


def upgrade() -> None:
    """Create the baseline schema on an empty database."""
    bind = op.get_bind()
    dialect = bind.dialect.name

    # ---- pgvector extension (Postgres only) -----------------------------
    if dialect == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ---- Create every table declared on the ORM metadata ----------------
    # Importing :mod:`src.db.base` registers every model class on
    # ``Base.metadata`` as a side effect of the imports, so
    # ``create_all`` picks them up.
    from src.db.base import Base  # noqa: WPS433 - deliberate deferred import

    tables = [
        Base.metadata.tables[name]
        for name in _TABLES_IN_CREATION_ORDER
        if name in Base.metadata.tables
    ]
    Base.metadata.create_all(bind=bind, tables=tables, checkfirst=True)

    # ---- Append-only trigger on audit_log (Postgres only) ---------------
    # Defence in depth against a privileged operator mutating history
    # directly via SQL.  SQLite has no trigger-level RAISE so we rely on
    # application-level enforcement there (see ``AuditLogEntry``).
    if dialect == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE FUNCTION audit_log_reject_mutation()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION
                    'audit_log is append-only — % is not permitted',
                    TG_OP;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute(
            """
            CREATE TRIGGER audit_log_no_update
            BEFORE UPDATE OR DELETE ON audit_log
            FOR EACH ROW
            EXECUTE FUNCTION audit_log_reject_mutation();
            """
        )


def downgrade() -> None:
    """Drop every baseline table in reverse order."""
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS audit_log_no_update ON audit_log")
        op.execute("DROP FUNCTION IF EXISTS audit_log_reject_mutation()")

    from src.db.base import Base  # noqa: WPS433

    # Drop in reverse creation order so dependents disappear before
    # their parents.
    tables_reversed = [
        Base.metadata.tables[name]
        for name in reversed(_TABLES_IN_CREATION_ORDER)
        if name in Base.metadata.tables
    ]
    for table in tables_reversed:
        table.drop(bind=bind, checkfirst=True)

    if dialect == "postgresql":
        # The pgvector extension is left installed — other databases in
        # the same cluster may still need it.  Uncomment the line below
        # if you want a fully clean downgrade.
        # op.execute("DROP EXTENSION IF EXISTS vector")
        pass


# ---------------------------------------------------------------------------
# TODO (follow-up migration)
# ---------------------------------------------------------------------------
# * Add per-tenant row-level security policies on analysis_jobs,
#   reports, claims, and verification_results so the auth agent's
#   multi-tenant isolation story is enforced at the database layer.
# * Introduce a materialized view summarising audit events per tenant
#   for compliance dashboards.
