# first line: 90
def _fq_compute(
    instance_digest: str,
    generator: str,
    replications: int,
    package: str,
    numpy_version: str,
    scipy_version: str,
    python_version: str,
    *,
    inst: Instance,
    params: GeneratorParams,
    n_jobs: int,
) -> dict[str, dict]:
    """``naive_fq.generator_quantiles_and_load`` as plain dicts: the function the disk cache memoises.

    The key is the first seven arguments (``fq_cache_key``: instance content digest, ``generator_id``, replications,
    ``package_sha256``, the NumPy, SciPy and Python versions); ``inst``, ``params`` and ``n_jobs`` are ignored by it
    (``FQ_CACHE_IGNORE``), and ``fq_quantiles`` builds the first two from them.

    Returns:
        ``{"quantiles": (c, pool, j, k) -> F_Q quantile, "load": (c, pool) -> the load (69)}``, from one pass.

    """
    quantiles, load = generator_quantiles_and_load(inst, params, replications, n_jobs)
    return {"quantiles": dict(quantiles), "load": dict(load)}
