import numpy as np
import numpy.testing as npt
import time


def gen_random_samples():
    """
    Generate 5 million random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size 5 million
        An array of 5 million random samples
    """
    ## TODO FILL IN

    ## Generates a 1-D array of random numbers with inputed length
    sample = np.random.randn(5000000)
 
    return sample


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    timeElapse: float
        The time it took to calculate the sum of squares (in seconds)
    """
    timeElapse = 0
    ss = 0
    ## TODO FILL IN


    avg = 0
    timeElapse = time.time()

    ## Finds the sum of all the values in array the divide by len of array
    for i in samples:
        avg += i

    avg /= len(samples)
    
    temp = 0

    ## Finds the sum of square  by subtracte mean by each value, then squaring those values, finally summing all the square values
    for i in samples:
        temp = i - avg
        ss += (temp * temp)

    ## Calculates the Time Elaose 
    timeElapse = time.time() - timeElapse


    return ss, timeElapse


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    timeElapse: float
        The time it took to calculate the sum of squares (in seconds)
    """
    timeElapse = 0
    ss = 0
    ## TODO FILL IN
    timeElapse = time.time()

    ## Calculates the avg by using numpy functions
    ## sum functions adds all the values in array then divide by number of arr
    avg = np.sum(samples) / len(samples)

    ## Created an arr of the avg and then subtract that arr by the samples arr
    avgArr = np.full((len(samples)),avg)

    temp = np.subtract(samples,avgArr)

    ## Dot product of vals to find the sum of squares 
    ss = np.dot(temp,temp)

    timeElapse = time.time() - timeElapse
    
    return ss, timeElapse


def main():
    # generate the random samples
    samples = gen_random_samples()
    # call the sum of squares
    ssFor, timeFor = sum_squares_for(samples)
    # call the numpy version
    ssNp, timeNp = sum_squares_np(samples)
    # make sure they're the same value
    npt.assert_almost_equal(ssFor, ssNp, decimal=5)
    # print out the values
    print("Time [sec] (for loop):", timeFor)
    print("Time [sec] (np loop):", timeNp)


if __name__ == "__main__":
    main()
