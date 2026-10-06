# first line: 176
def _cuts_compute(
    instance_digest: str,
    generator: str,
    draws: int,
    entropy: int,
    package: str,
    numpy_version: str,
    scipy_version: str,
    python_version: str,
    *,
    inst: Instance,
    params: GeneratorParams,
    n_jobs: int,
) -> dict:
    """``strata.cut_points`` as a plain dict: the function the cut-point disk cache memoises (``cut_points_cached``)."""
    c = cut_points(inst, params, draws, entropy, n_jobs)
    return {"quantiles": list(c.quantiles), "values": list(c.values), "draws": c.draws, "generator_id": c.generator_id}
