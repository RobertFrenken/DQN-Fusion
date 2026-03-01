"""Custom exception hierarchy for pipeline errors.

Enables callers to catch pipeline-specific errors without catching
unrelated ValueError/FileNotFoundError from library code.
"""


class PipelineError(Exception):
    """Base for all pipeline errors."""


class ConfigValidationError(PipelineError):
    """Invalid or inconsistent pipeline configuration."""


class CheckpointNotFoundError(PipelineError, FileNotFoundError):
    """Required model checkpoint does not exist."""


class StageFailedError(PipelineError):
    """A pipeline stage failed during execution."""


class MemoryBudgetExceeded(PipelineError):
    """GPU memory budget exceeded during training."""
