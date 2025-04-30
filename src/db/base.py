"""Re-export the canonical Base so that ``init_db`` and Alembic always see
the same metadata registry that the ORM models are attached to.

All model files import ``Base`` from :mod:`src.models.base`.  This module
simply re-exports it (and the models themselves) so that
:func:`Base.metadata.create_all` picks up every table.
"""

from src.models.base import Base  # noqa: F401
from src.models.job import AnalysisJob  # noqa: F401
from src.models.claim import Claim  # noqa: F401
from src.models.verification import VerificationResult, EvidenceRecord  # noqa: F401
from src.models.report import Report  # noqa: F401
from src.models.audit import AuditLogEntry  # noqa: F401
