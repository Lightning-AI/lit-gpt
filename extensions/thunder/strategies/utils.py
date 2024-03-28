from typing import TYPE_CHECKING, Optional, Tuple, Union, Sequence

if TYPE_CHECKING:
    from thunder import Executor


def _validate_executors(executors: Optional[Sequence[Union["Executor", str]]]) -> Optional[Tuple["Executor", ...]]:
    """Converts string executors into it's respective ``Executor`` object."""
    if executors is None:
        return None
    from thunder import get_all_executors

    final = []
    issues = []
    all = get_all_executors()
    for executor in executors:
        if isinstance(executor, str):
            for existing in all:
                if executor == existing.name:
                    final.append(existing)
                    break
            else:
                issues.append(executor)
        else:
            final.append(executor)
    if issues:
        raise ValueError(f"Did not find the executors {issues} in {all}")
    return tuple(final)
