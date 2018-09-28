def fibonacci(n):
    """
    Returns the n-th number in the Fibonacci sequence.

    Parameters
    ----------
    n: int
       The n-th number in the Fibonacci sequence.
    """
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
