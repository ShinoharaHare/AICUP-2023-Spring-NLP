from contextlib import contextmanager

__all__ = ['get_args', 'disable_datasets_cache']

def get_args():
    import inspect

    current_frame = inspect.currentframe()
    if current_frame:
        frame = current_frame.f_back
    args, _, _, f_locals = inspect.getargvalues(frame)
    args = {a: f_locals[a] for a in args}
    return args

@contextmanager
def disable_datasets_cache():
    from datasets import disable_caching, enable_caching, is_caching_enabled

    enabled = is_caching_enabled()
    disable_caching()
    
    yield

    if enabled:
        enable_caching()
