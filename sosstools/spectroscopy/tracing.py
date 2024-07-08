# A place to store spectral tracing algorithms

from abc import ABC, abstractmethod
import numpy as np

from astropy.convolution import convolve_fft
from astropy.modeling.models import Gaussian1D

from scipy.interpolate import splev, splrep, sproot, interp1d
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import fminbound


import numpy as np
from astropy.modeling.models import Gaussian1D
from astropy.convolution import convolve_fft
from typing import List, Union


class TraceAlgorithm(ABC):

    @abstractmethod
    def trace(self):
        pass


class CCFTracer(TraceAlgorithm):
    # code below can be filled here
    pass


class EdgeTriggerTracers(TraceAlgorithm):
    # code below can be filled here
    pass


def create_kernel(
    mu1: float, sig1: float, mu2: float, sig2: float, npix: int
) -> np.ndarray:
    """Create a double Gaussian kernel."""
    x = np.linspace(0, npix, npix + 1)
    g1 = Gaussian1D(amplitude=1, mean=mu1 + npix / 2, stddev=sig1)
    g2 = Gaussian1D(amplitude=1, mean=mu2 + npix / 2, stddev=sig2)
    kernel = g1(x) + g2(x)
    return kernel


def get_ccf_convolve(
    signal: np.ndarray, ccf_parameters: List[float] = [-7.5, 3.0, 7.5, 3.0]
) -> np.ndarray:
    """
    Obtain the CCF by convolution using the signal and a double Gaussian
    kernel.

    Parameters
    ----------
    signal : np.ndarray
        Array containing the column signal of the detector.

    ccf_parameters : List[float]
        List of floats for the double Gaussian kernel parameters:
        [mu1, sig1, mu2, sig2].

    Returns
    -------
    ccf : np.ndarray
        Array containing the cross-correlation function between the signal and
        the kernel.
    """
    mu1, sig1, mu2, sig2 = ccf_parameters
    npix = len(signal)
    kernel = create_kernel(mu1, sig1, mu2, sig2, npix)
    ccf = convolve_fft(signal, kernel)
    return ccf


def trace_spectrum(
    image,
    dqflags,
    xstart,
    ystart,
    profile_radius=20,
    correct_outliers=False,
    nsigma=100,
    median_filter_radius=5,
    method="ccf",
    ccf_function="gaussian",
    ccf_parameters=None,
    ccf_step=0.001,
    gaussian_filter=False,
    gauss_filter_width=10,
    xend=None,
    y_tolerance=2,
    verbose=False,
):
    """
    Function that non-parametrically traces spectra. There are various methods
    to trace the spectra. The default method is `ccf`, which performs
    cross-correlation to find the trace positions given a user-specified
    function (default is 'gaussian'; can also be 'double gaussian' or a
    user-specified function). Tracing happens from columns `xstart` until
    `xend` --- default for `xend` is `0`.

    Parameters
    ----------

    image: numpy.array
        The image that wants to be traced.
    dqflags: ndarray
        The data quality flags for each pixel in the image. Only pixels with DQ
        flags of zero will be used in the tracing.
    xstart: float
        The x-position (column) on which the tracing algorithm will be started
    ystart: float
        The estimated y-position (row) of the center of the trace. An estimate
        within a few pixels is enough (defined by y_tolerance).
    profile_radius: float
        Expected radius of the profile measured from its center. Only this
        region will be used to estimate
        the trace position of the spectrum.
    correct_outliers : bool
        Decide if to correct outliers or not on each column. If True, outliers
        are detected via a median filter.
    nsigma : float
        Median filters are applied to each column in search of outliers if
        `correct_outliers` is `True`. `nsigma` defines how many n-sigma above
        the noise level the residuals of the median filter and the image should
        be considered outliers.
    median_filter_radius : int
        Radius of the median filter in case `correct_outliers` is `True`. Needs
        to be an odd number. Default is `5`.
    method : string
        Method by which the tracing is expected to happen. Default is `ccf`;
        can also be `centroid`, which will use the centroid of each column
        to estimate the center of the trace.
    ccf_function : string or function
        Function to cross-correlate cross-dispersion profiles against. Default
        is `gaussian` (useful for most instruments) --- can also be
        `double gaussian` (useful for e.g., NIRISS/SOSS --- double gaussian
        separation tailored to that instrument). Alternatively, a function can
        be passed directly --- this function needs to be evaluated at a set of
        arrays `x`, and be centered at `x=0`.
    ccf_parameters : list
        Parameters of the function against which data will be CCF'ed. For details, see the get_ccf function; default is None, which defaults to the get_ccf defaults.
    ccf_step : double
        Step at which the CCF will run. The smallest, the most accurate, but also the slower the CCF method is. Default is `0.001`.
    gaussian_filter : bool
        Flag that defines if each column will be convolved with a gaussian filter (good to smooth profile to match a gaussian better). Default is `False`.
    gauss_filter_width : float
        Width of the gaussian filter used to perform the centroiding of the first column, if `gaussian_filter` is `True`.
    xend: int
        x-position at which tracing ends. If none, trace all the columns left to xstart.
    y_tolerance: float
        When tracing, if the difference between the two difference traces at two contiguous columns is larger than this,
        then assume tracing failed (e.g., cosmic ray).
    verbose: boolean
        If True, print error messages.

    Returns
    -------

    x : numpy.array
        Columns at which the trace position is being calculated.
    y : numpy.array
        Estimated trace position.
    """

    # Define x-axis:
    if xend is not None:

        if xend < xstart:

            x = np.arange(xend, xstart + 1)
            indexes = range(len(x))[::-1]
            direction = "left"

        else:

            x = np.arange(xstart, xend + 1)
            indexes = range(len(x))
            direction = "right"

    else:

        x = np.arange(0, xstart + 1)

    # Define y-axis:
    y = np.arange(image.shape[0])

    # Define status of good/bad for each trace position:
    status = np.full(len(x), True, dtype=bool)

    # Define array that will save trace at each x:
    ytraces = np.zeros(len(x))

    first_time = True
    for i in indexes:

        xcurrent = x[i]

        # Perform median filter to identify nasty (i.e., cosmic rays) outliers in the column:
        mf = median_filter(image[:, xcurrent], size=median_filter_radius)

        if correct_outliers:

            residuals = mf - image[:, xcurrent]
            mad_sigma = get_mad_sigma(residuals)
            column_nsigma = np.abs(residuals) / mad_sigma

        else:

            column_nsigma = np.zeros(image.shape[0]) * nsigma

        # Extract data-quality flags for current column; index good pixels --- mask nans as well:
        idx_good = np.where(
            (dqflags[:, xcurrent] == 0)
            & (~np.isnan(image[:, xcurrent]) & (column_nsigma < nsigma))
        )[0]
        idx_bad = np.where(
            ~(
                (dqflags[:, xcurrent] == 0)
                & (~np.isnan(image[:, xcurrent]) & (column_nsigma < nsigma))
            )
        )[0]

        if len(idx_good) > 0:

            # Replace bad values with the ones in the median filter:
            column_data = np.copy(image[:, xcurrent])
            column_data[idx_bad] = mf[idx_bad]

            if gaussian_filter:

                # Convolve column with a gaussian filter; remove median before convolving:
                filtered_column = gaussian_filter1d(
                    column_data - np.median(column_data), gauss_filter_width
                )

            else:

                filtered_column = column_data - np.median(column_data)

            # Find trace depending on the method, only within pixels close to profile_radius:
            idx = np.where(np.abs(y - ystart) < profile_radius)[0]
            if method == "ccf":

                # Run CCF search using only the pixels within profile_radius:
                lags, ccf = get_ccf(
                    y[idx],
                    filtered_column[idx],
                    function=ccf_function,
                    parameters=ccf_parameters,
                    lag_step=ccf_step,
                )
                idx_max = np.where(ccf == np.max(ccf))[0]

                ytraces[i] = lags[idx_max]

            elif method == "convolve":
                # Run CCF using using convolution method applied to entire column
                ccf = get_ccf_convolve(filtered_column, ccf_parameters=ccf_parameters)

                # Find where the minimum occurs within the profile_radius
                # by finding the peak. A cubic spline interpolations is
                # used to smooth the ccf. The interpolated ccf has been
                # compared to numerically computed ccf with fine sampling
                # and yield similar results. By applying a miminization
                # scheme we can idenitfy the optimal position where the peak
                # occurs in the region of interest
                #
                y1, y2 = y[idx][[0, -1]]
                ytraces[i] = fminbound(interp1d(y, -1 * ccf, kind="cubic"), y1, y2)

            elif method == "centroid":

                # Find pixel centroid within profile_radius pixels of the initial guess:
                ytraces[i] = np.sum(y[idx] * filtered_column[idx]) / np.sum(
                    filtered_column[idx]
                )

            else:

                raise Exception(
                    'Cannot trace spectra with method "'
                    + method
                    + '": method not recognized. Available methods are "ccf" and "centroid"'
                )

            # Get the difference of the current trace position with the previous one (if any):
            if not first_time:

                if direction == "left":

                    previous_trace = ytraces[i + 1]

                else:

                    previous_trace = ytraces[i - 1]

            else:

                previous_trace = ystart
                first_time = False

            difference = np.abs(previous_trace - ytraces[i])

            if difference > y_tolerance:

                if verbose:
                    print(
                        "Tracing failed at column",
                        xcurrent,
                        "; estimated trace position:",
                        ytraces[i],
                        ", previous one:",
                        previous_trace,
                        "> than tolerance: ",
                        y_tolerance,
                        ". Replacing with closest good trace position.",
                    )

                ytraces[i] = previous_trace

            ystart = ytraces[i]

        else:

            print(xcurrent, "is a bad column. Setting to previous trace position:")
            ytraces[i] = previous_trace
            status[i] = True

    # Return all trace positions:
    return x, ytraces


def get_fwhm(x, y, k=3):
    """
    Given a profile (x,y), this function calculates the FWHM of the profile via spline
    interpolation. The idea is to interpolate the profile, and find where the profile
    minus half the maximum crosses zero via root-finding.

    Parameters
    ----------

    x : numpy.array
        Array containing the x-axis of the profile (e.g., pixels).
    y : numpy.array
        Array containing the y-axis of the profile (e.g., counts).
    k : int
        Interpolation degree

    Returns
    -------

    The FWHM (a double) if there is a single root; 0 if it doesn't exist/there are more than one root.
    """

    half_max = np.max(y) / 2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        return 0.0

    elif len(roots) < 2:
        return 0.0
    else:
        return abs(roots[1] - roots[0])


def trace_fwhm(tso, x, y, distance_from_trace=15):
    """
    This function calculates the FWHM time-series accross a set of frames (the `tso` array), as a function of position/wavelength (the trace position
    `x` and `y`).

    Parameters
    ----------

    tso : numpy.array
          Array of shape `(time, rows, columns)` of a set of frames on which one wants to calculate the FWHM as a function of `time` and `columns`.
    x : numpy.array
          Trace column position.
    y : int
          Trace row position.
    distance_from_trace : int
          Distance from the trace at which the FWHM will be calculated.

    Returns
    -------

    fwhms : numpy.array
          Array of shape `(time, columns)` having the FWHM at each of the `x` columns and `times`.

    superfwhm : numpy.array
          The median FWHM accross all `x` columns
    """

    fwhms = np.zeros([tso.shape[0], tso.shape[2]])

    row_index = np.arange(tso.shape[1])
    min_row = 0
    max_row = tso.shape[1] - 1

    for i in range(tso.shape[0]):

        for j in range(tso.shape[2]):

            if j in x:

                idx = np.where(j == x)[0]
                integer_row = int(y[idx])

                lower_row = np.max([min_row, integer_row - distance_from_trace])
                upper_row = np.min([max_row, integer_row + distance_from_trace])

                fwhms[i, j] = get_fwhm(
                    row_index[lower_row:upper_row], tso[i, lower_row:upper_row, j]
                )

    normalized_fwhms = np.zeros(fwhms.shape)

    # Nan the zeroes in fwhms:
    idx = np.where(fwhms == 0.0)
    fwhms[idx] = np.nan

    for i in range(tso.shape[2]):

        # Normalize:
        normalized_fwhms[:, i] = fwhms[:, i] - np.nanmedian(fwhms[:, i])

    super_fwhm = np.nanmedian(normalized_fwhms, axis=1)

    return fwhms, super_fwhm


def get_mad_sigma(x):

    x_median = np.nanmedian(x)

    return 1.4826 * np.nanmedian(np.abs(x - x_median))


def fit_spline(x, y, nknots=None, x_knots=None):
    """
    This function fits a spline to data `x` and `y`. The code can be use in three ways:

    1.  Passing a value to `nknots`; in that case, `nknots` equally spaced knots will be placed along `x`
        to fit the data.

    2.  Passing an array to `x_knots`. In this case, knots will be placed at `x_knots`.

    3.  Passing a list of `nknots` and `x_knots`. In this case, each element of `x_knots` is assumed to be the lower and
        upper limits of a region; the corresponding element of `nknots` will be used to put equally spaced knots in
        that range.

    Parameters
    ----------

    x : numpy.array
        x-values for input data.

    y : numpy.array
        y-values for input data.

    nknots : int or list
        Number of knots to be used.

    x_knots : numpy.array or list
        Position of the knots or regions of knots (see description)

    Returns
    -------

    function : spline object
        Function over which the spline can be evaluated at.
    prediction : numpy.array
        Array of same dimensions of `x` and `y` with the spline evaluated at the input `x`.

    """

    xmin, xmax = np.min(x), np.max(x)

    if (nknots is not None) and (x_knots is not None):

        knots = np.array([])
        for i in range(len(x_knots)):

            knots = np.append(
                knots, np.linspace(x_knots[i][0], x_knots[i][1], nknots[i])
            )

    elif x_knots is not None:

        knots = x_knots

    elif nknots is not None:

        idx = np.argsort(x)

        knots = np.linspace(x[idx][1], x[idx][-2], nknots)

    # Check knots are well-positioned:
    if np.min(knots) <= xmin:

        raise Exception(
            "Lower knot cannot be equal or smaller than the smallest x input value."
        )

    if np.max(knots) >= xmax:

        raise Exception(
            "Higher knot cannot be equal or larger than the largest x input value."
        )

    # Obtain spline representation:
    tck = splrep(x, y, t=knots)
    function = lambda x: splev(x, tck)

    # Return it:
    return function, function(x)


def trace_spectrum_test():
    print("Tracing Spectrum")
