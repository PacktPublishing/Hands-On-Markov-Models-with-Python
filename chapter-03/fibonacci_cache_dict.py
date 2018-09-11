cache = {0: 0, 1: 1} # Initialize the first two values.
def fibonacci(n):
    """
    Returns the n-th number in the Fibonacci sequence.

    Parameters
    ----------
    n: int
       The n-th number in the Fibonacci sequence.
    """
     try:
         return cache[n]
     except KeyError:
         fib = fibonacci(n-1) + fibonacci(n-2)
         cache[n] = fib 
         return fib
