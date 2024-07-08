import os
import numpy as np

import matplotlib.pyplot as plt


BACKGROUND_TEMPLATE_FILE = (
    "/Users/tbaines/stsci/jwst/niriss/soss/models/models/model_background256.npy"
)


def load_background_template():
    return np.load(BACKGROUND_TEMPLATE_FILE)


def get_background_scaler(
    image,
    bkg_model,
    box_bounds=[210, 250, 600, 800],
    percentile_range=[0.25, 0.75],
    plot=False,
):

    # extract subregion of background
    iy1, iy2, ix1, ix2 = box_bounds
    bkg_postage = image[iy1:iy2, ix1:ix2]
    model_bkg_postage = bkg_model[iy1:iy2, ix1:ix2]

    # mask the bad pixel and compute ratios
    bad_pixels = np.isnan(bkg_postage)
    ratio = bkg_postage[~bad_pixels] / model_bkg_postage[~bad_pixels]
    ratio = ratio.flatten()

    # estiamte the scale factor i.e. the median ratio
    idx_sorted = np.argsort(ratio)
    npixels = len(ratio)

    lower_percentile, upper_percentile = percentile_range

    pixel_index = np.arange(len(ratio))

    idx_lower = int(npixels * lower_percentile)
    idx_upper = int(npixels * upper_percentile)

    median_ratio = np.median(ratio[idx_sorted][idx_lower:idx_upper])

    if plot:
        print("plotting scaling results")
        original_pixels = bkg_postage.flatten()
        bkg_substracted_pixels = (
            bkg_postage.flatten() - model_bkg_postage.flatten() * median_ratio
        )
        non_outliers_original = np.where(
            (bkg_postage.flatten() < 10.5) & (bkg_postage.flatten() > 1.0)
        )[0]
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fontsize = 16

        # Plot 1: Background Distrubition ratio
        # axes[0].set_title('Background Distrubition')
        axes[0].plot(pixel_index, ratio[idx_sorted], label="Background Distribution")
        axes[0].plot(
            pixel_index[idx_lower:idx_upper],
            ratio[idx_sorted][idx_lower:idx_upper],
            label=f"{lower_percentile*100} to {upper_percentile*100} Percentile Region",
        )
        axes[0].set_xlabel("Pixel index", fontsize=fontsize)
        axes[0].set_ylabel("Data / Model", fontsize=fontsize)
        axes[0].tick_params(axis="both", labelsize=fontsize)
        axes[0].legend()

        # Plot 2: Ratio distribution
        axes[1].plot(ratio, ".", label="Ratio")
        axes[1].plot(
            [0, len(ratio)], [median_ratio, median_ratio], label="Median Ratio"
        )
        axes[1].set_xlabel("Pixel index", fontsize=fontsize)
        axes[1].set_ylabel("Data / Model", fontsize=fontsize)
        axes[1].tick_params(axis="both", labelsize=fontsize)
        axes[1].legend()

        # Plot 3: Original vs. Corrected Pixels
        axes[2].plot(original_pixels[non_outliers_original], label="Original")
        axes[2].plot(bkg_substracted_pixels[non_outliers_original], label="Corrected")
        axes[2].plot([0, 8000], [0, 0], c="k")
        axes[2].set_xlabel("Pixel Index", fontsize=fontsize)
        axes[2].set_ylabel("Pixel Value", fontsize=fontsize)
        axes[2].tick_params(axis="both", labelsize=fontsize)
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    return median_ratio
