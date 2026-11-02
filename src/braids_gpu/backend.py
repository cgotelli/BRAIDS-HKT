"""Backend selection is a user decision, never a mesh-size heuristic."""


def require_backend(value):
    if value not in ('cpu','gpu'):
        raise ValueError('Choose backend explicitly: cpu or gpu; automatic selection is not supported')
    return value


def dispatch(backend, *, cpu_runner, gpu_runner, **kwargs):
    """A failed GPU run propagates its error; there is no hidden CPU fallback."""
    backend=require_backend(backend)
    return (gpu_runner if backend=='gpu' else cpu_runner)(**kwargs)

