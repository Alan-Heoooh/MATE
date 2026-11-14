"""MATE component policies; load network dependencies only when requested."""

__all__ = ["RISE", "FT_Proprio", "RISE_FT_Proprio"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from policy import policy as components
    return getattr(components, name)
