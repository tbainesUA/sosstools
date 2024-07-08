import numpy as np


def getSimpleSpectrum(
    data,
    x,
    y,
    aperture_radius,
    background_radius=50,
    error_data=None,
    correct_bkg=False,
    method="sum",
    bkg_method="all",
):
    """
    This function takes as inputs two arrays (x,y) that follow the trace,
    and returns the added flux over the defined aperture radius (and its error, if an error matrix
    is given as well), substracting in the way any background between the aperture radius and the
    background radius. The background is calculated by taking the median of the points between the
    aperture_radius and the background_radius.

    Parameters
    ----------
    data: ndarray
        Image from which the spectrum wants to be extracted
    x: ndarray
        Array with the x-axis of the trace (i.e., the columns, wavelength direction)
    y: ndarray
        Array with the y-axis of the trace (i.e., rows, spatial direction)
    aperture_radius: float
        Distance from the center of the trace at which you want to add fluxes.
    background_radius: float
        Distance from the center of the trace from which you want to calculate the background. The
        background region will be between this radius and the aperture_radius.
    error_data: ndarray
        Image with the errors of each pixel value on the data ndarray above
    correct_bkg: boolean
        If True, apply background correction. If false, ommit this.
    method : string
        Method used to perform the extraction. Default is `sum`; `average` takes the average of the non-fractional pixels
        used to extract the spectrum. This latter one is useful if the input is a wavelength map.
    bkg_method : string
        Method for the background substraction. Currently accepts 'all' to use pixels at both sides, 'up' to use pixels "above" the spectrum and
        'down' to use pixels "below" the spectrum.
    """

    method = method.lower()

    # If average method being used, remove background correction:
    if method == "average":
        correct_bkg = False

    # Create array that will save our fluxes:
    flux = np.zeros(len(x))
    if error_data is not None:
        flux_error = np.zeros(len(x))
    max_column = data.shape[0] - 1

    for i in range(len(x)):

        # Cut the column with which we'll be working with:
        column = data[:, int(x[i])]
        if error_data is not None:
            variance_column = error_data[:, int(x[i])] ** 2

        # Define limits given by the aperture_radius and background_radius variables:
        if correct_bkg:
            left_side_bkg = np.max([y[i] - background_radius, 0])
            right_side_bkg = np.min([max_column, y[i] + background_radius])
        left_side_ap = np.max([y[i] - aperture_radius, 0])
        right_side_ap = np.min([max_column, y[i] + aperture_radius])

        # Extract background, being careful with edges:
        if correct_bkg:

            bkg_left = column[
                np.max([0, int(left_side_bkg)]) : np.max([0, int(left_side_ap)])
            ]
            bkg_right = column[
                np.min([int(right_side_ap), max_column]) : np.max(
                    [int(right_side_bkg), max_column]
                )
            ]

            if bkg_method == "all":

                bkg = np.median(np.append(bkg_left, bkg_right))

            elif bkg_method == "up":

                bkg = np.median(bkg_right)

            elif bkg_method == "down":

                bkg = np.median(bkg_left)

        else:

            bkg = 0.0

        # Substract it from the column:
        column -= bkg

        # Perform aperture extraction of the background-substracted column, being careful with pixelization
        # at the edges. First, deal with left (up) side:
        l_decimal, l_integer = np.modf(left_side_ap)
        l_integer = int(l_integer)
        if l_decimal < 0.5:
            l_fraction = (0.5 - l_decimal) * column[np.min([l_integer, max_column])]
            l_limit = l_integer + 1
            if error_data is not None:
                l_fraction_variance = ((0.5 - l_decimal) ** 2) * variance_column[
                    np.min([l_integer, max_column])
                ]
        else:
            l_fraction = (1.0 - (l_decimal - 0.5)) * column[
                np.min([l_integer + 1, max_column])
            ]
            l_limit = l_integer + 2
            if error_data is not None:
                l_fraction_variance = (
                    (1.0 - (l_decimal - 0.5)) ** 2
                ) * variance_column[np.min([l_integer + 1, max_column])]

        # Now right (down) side:
        r_decimal, r_integer = np.modf(right_side_ap)
        r_integer = int(r_integer)
        if r_decimal < 0.5:
            r_fraction = (1.0 - (0.5 - r_decimal)) * column[
                np.min([max_column, r_integer])
            ]
            r_limit = r_integer
            if error_data is not None:
                r_fraction_variance = (
                    (1.0 - (0.5 - r_decimal)) ** 2
                ) * variance_column[np.min([max_column, r_integer])]
        else:
            r_fraction = (r_decimal - 0.5) * column[np.min([max_column, r_integer + 1])]
            r_limit = r_integer + 1
            if error_data is not None:
                r_fraction_variance = ((r_decimal - 0.5) ** 2) * variance_column[
                    np.min([max_column, r_integer + 1])
                ]

        # Save total flux in current column:
        if method == "sum":
            flux[i] = l_fraction + r_fraction + np.sum(column[l_limit:r_limit])

        elif method == "average":
            flux[i] = np.mean(column[l_limit:r_limit])

        else:
            raise Exception(
                'Method "'
                + method
                + '" currently not supported for aperture extraction. Select either "sum" or "average".'
            )

        if error_data is not None:
            # Total error is the sum of the variances:
            flux_error[i] = np.sqrt(
                np.sum(variance_column[l_limit:r_limit])
                + l_fraction_variance
                + r_fraction_variance
            )
    if error_data is not None:
        return flux, flux_error
    else:
        return flux


def extract_1d_test():
    print("extracting spectrum")
