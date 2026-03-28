"""Dask / Ray distributed task scheduling helpers."""

from __future__ import annotations

from typing import Callable, Any


class DaskScheduler:
    """Thin wrapper for submitting embarrassingly parallel tasks to Dask.

    Parameters
    ----------
    scheduler_address:
        Address of a running Dask scheduler (e.g. ``"tcp://localhost:8786"``).
        Pass ``None`` to use a local ``LocalCluster``.
    n_workers:
        Number of workers to spawn when creating a ``LocalCluster``.
    """

    def __init__(
        self,
        scheduler_address: str | None = None,
        n_workers: int = 4,
    ) -> None:
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers
        self._client = None

    def start(self) -> "DaskScheduler":
        """Connect to or create a Dask cluster.  Returns self."""
        from dask.distributed import Client, LocalCluster

        if self.scheduler_address:
            self._client = Client(self.scheduler_address)
        else:
            cluster = LocalCluster(n_workers=self.n_workers)
            self._client = Client(cluster)
        return self

    def map(self, fn: Callable, items: list) -> list:
        """Submit ``fn`` for each item and gather results.

        Parameters
        ----------
        fn:
            Function to apply to each item.
        items:
            List of inputs.

        Returns
        -------
        list
            Results in the same order as *items*.
        """
        if self._client is None:
            raise RuntimeError("Call start() before map().")
        futures = self._client.map(fn, items)
        return self._client.gather(futures)

    def close(self) -> None:
        """Shut down the Dask client and cluster."""
        if self._client:
            self._client.close()
            self._client = None


class RayScheduler:
    """Thin wrapper for submitting tasks via Ray.

    Parameters
    ----------
    address:
        Ray cluster address.  Pass ``None`` to start a local cluster.
    num_cpus:
        Number of CPUs for the local cluster (ignored if *address* is given).
    """

    def __init__(self, address: str | None = None, num_cpus: int | None = None) -> None:
        self.address = address
        self.num_cpus = num_cpus

    def start(self) -> "RayScheduler":
        """Initialise the Ray runtime.  Returns self."""
        import ray

        kwargs: dict[str, Any] = {}
        if self.address:
            kwargs["address"] = self.address
        if self.num_cpus:
            kwargs["num_cpus"] = self.num_cpus
        ray.init(**kwargs)
        return self

    def map(self, fn: Callable, items: list) -> list:
        """Apply *fn* to each item in parallel using Ray remote tasks.

        Parameters
        ----------
        fn:
            Plain Python callable.
        items:
            List of inputs.

        Returns
        -------
        list
            Results in input order.
        """
        import ray

        remote_fn = ray.remote(fn)
        futures = [remote_fn.remote(item) for item in items]
        return ray.get(futures)

    def close(self) -> None:
        """Shut down the Ray runtime."""
        import ray
        ray.shutdown()
