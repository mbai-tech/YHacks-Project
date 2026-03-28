"""MPI utility wrappers built on top of mpi4py."""

from __future__ import annotations

from typing import Any


def get_comm():
    """Return the MPI world communicator (or a ``None``-safe stub if MPI is absent).

    Returns
    -------
    mpi4py.MPI.Comm | _SerialComm
        MPI communicator or a single-rank stub for non-MPI environments.
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return _SerialComm()


class _SerialComm:
    """Minimal MPI communicator stub for serial (non-MPI) execution."""

    rank: int = 0
    size: int = 1

    def Get_rank(self) -> int:  # noqa: N802
        return 0

    def Get_size(self) -> int:  # noqa: N802
        return 1

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def scatter(self, sendobj: list, root: int = 0) -> Any:
        return sendobj[0] if sendobj else None

    def gather(self, sendobj: Any, root: int = 0) -> list:
        return [sendobj]

    def Barrier(self) -> None:  # noqa: N802
        pass


def scatter_ensemble(params: list, comm=None) -> list:
    """Distribute a parameter list across MPI ranks.

    Parameters
    ----------
    params:
        Full list of parameter sets (only used on rank 0).
    comm:
        MPI communicator (defaults to ``get_comm()``).

    Returns
    -------
    list
        Sub-list assigned to the calling rank.
    """
    if comm is None:
        comm = get_comm()

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        chunks = [params[i::size] for i in range(size)]
    else:
        chunks = None

    return comm.scatter(chunks, root=0)


def gather_results(local_results: list, comm=None) -> list:
    """Collect per-rank results onto rank 0.

    Parameters
    ----------
    local_results:
        Results computed on the calling rank.
    comm:
        MPI communicator.

    Returns
    -------
    list
        Flattened list of all results on rank 0; ``None`` on other ranks.
    """
    if comm is None:
        comm = get_comm()

    all_results = comm.gather(local_results, root=0)
    if comm.Get_rank() == 0 and all_results is not None:
        return [item for sublist in all_results for item in sublist]
    return []
