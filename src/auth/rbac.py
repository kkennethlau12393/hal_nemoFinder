"""Role-based access control primitives.

Roles (see :class:`src.models.tenant.Role`) are coarse-grained labels
customers understand.  Internally, each role expands to a set of
:class:`Permission` values which individual endpoints guard with
:func:`require_permission`.  Adding a new permission therefore does
not require a schema migration — only an update to
:data:`ROLE_PERMISSIONS`.
"""

from __future__ import annotations

import enum
from typing import Callable

from src.models.tenant import Role


class Permission(str, enum.Enum):
    """Fine-grained action guards used by route handlers."""

    READ_JOBS = "read_jobs"
    WRITE_JOBS = "write_jobs"
    READ_REPORTS = "read_reports"
    MANAGE_USERS = "manage_users"
    MANAGE_API_KEYS = "manage_api_keys"
    MANAGE_TENANT = "manage_tenant"
    VIEW_METRICS = "view_metrics"
    RUN_CALIBRATION = "run_calibration"


#: Canonical role → permission mapping.  Admins have everything; analysts
#: can submit and inspect work; viewers are read-only.
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.admin: {
        Permission.READ_JOBS,
        Permission.WRITE_JOBS,
        Permission.READ_REPORTS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_API_KEYS,
        Permission.MANAGE_TENANT,
        Permission.VIEW_METRICS,
        Permission.RUN_CALIBRATION,
    },
    Role.analyst: {
        Permission.READ_JOBS,
        Permission.WRITE_JOBS,
        Permission.READ_REPORTS,
        Permission.VIEW_METRICS,
        Permission.RUN_CALIBRATION,
    },
    Role.viewer: {
        Permission.READ_JOBS,
        Permission.READ_REPORTS,
        Permission.VIEW_METRICS,
    },
}


def has_permission(role: Role, permission: Permission) -> bool:
    """Return True if *role* implies *permission*."""
    return permission in ROLE_PERMISSIONS.get(role, set())


def permissions_for_role(role: Role) -> set[Permission]:
    """Return a shallow copy of the permissions granted to *role*."""
    return set(ROLE_PERMISSIONS.get(role, set()))


def require_permission(permission: Permission) -> Callable:
    """Return a FastAPI dependency that enforces *permission*.

    This is re-exported from :mod:`src.auth.dependencies` where the
    actual dependency lives; placing a thin wrapper here lets callers
    import from either module.  The real implementation is looked up
    lazily to avoid a circular import with
    :mod:`src.auth.dependencies`.
    """
    from src.auth.dependencies import require_permission as _impl

    return _impl(permission)
