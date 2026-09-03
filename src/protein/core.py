"""The one function this placeholder carries, so the gates have something to act on."""


def greet(name: str = "world") -> str:
    """Return a greeting addressed to `name`.

    Parameters
    ----------
    name
        Who to greet.

    Returns
    -------
    str
        The greeting.

    Examples
    --------
    >>> greet("lab")
    'Hello, lab!'
    """
    return f"Hello, {name}!"
