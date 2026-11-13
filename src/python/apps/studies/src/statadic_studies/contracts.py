"""Compatibility imports for frozen designs and execution failures."""

from statadic_studies.definitions import STUDIES as STUDIES
from statadic_studies.study_types import Study as Study


class ProcedureFailure(Exception):
    """An attempted procedure failed; infrastructure interruptions are separate."""
