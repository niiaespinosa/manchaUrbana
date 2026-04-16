import os
import glob
import rasterio
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def create_true_color_png(folder_path):
    print(f"\nProcessing {folder_path} for True Color PNG...")
    
    # Define paths for the target bands
    red_path = os.path.join(folder_path, "red.tif")
    green_path = os.path.join(folder_path, "green.tif")
    blue_path = os.path.join(folder_path, "blue.tif")
    
    if not (os.path.exists(red_path) and os.path.exists(green_path) and os.path.exists(blue_path)):
        print(f"Skipping. Missing RGB bands in {folder_path}")
        return

    # 1. Read the bands
    with rasterio.open(red_path) as src:
        red = src.read(1)
        nodata = src.nodata
    with rasterio.open(green_path) as src:
        green = src.read(1)
    with rasterio.open(blue_path) as src:
        blue = src.read(1)

    # 2. Stack into a 3D array (Height, Width, Channels)
    # The arrays are currently float32 (e.g. 0.0 to 1.0)
    rgb = np.dstack((red, green, blue))

    # 3. Handle NaNs (missing edges/clouds)
    # Convert all np.nan to 0 so the image viewer doesn't freak out
    rgb = np.nan_to_num(rgb, nan=0.0)

    # 4. Contrast Stretching
    # true reflectance values for urban/land often peak around 0.3. 
    # If we map 0-1 to 0-255, the image looks extremely dark. 
    # We apply robust contrast stretching (ignoring the 0 values used for borders)
    valid_pixels = rgb[rgb > 0]
    if valid_pixels.size == 0:
        print("Image is entirely empty!")
        return
        
    p2, p98 = np.percentile(valid_pixels, (2, 98))
    
    # Rescale intensity dynamically maps the 2nd-98th percentile to 0-1 floats
    rgb_stretched = exposure.rescale_intensity(rgb, in_range=(p2, p98), out_range=(0, 1))

    # 5. Export as a standard visualization image (.png)
    folder_name = os.path.basename(folder_path.rstrip('/'))
    output_png = f"{folder_name}_true_color.png"
    
    plt.imsave(output_png, rgb_stretched)
    print(f"Saved visualization successfully to: {output_png}")


if __name__ == "__main__":
    print("WARNING: This visualization script requires 'scikit-image' and 'matplotlib'.")
    print("Install via: pip install scikit-image matplotlib\n")
    
    # Search for all subdirectories in the images/ folder
    folders = glob.glob("images/*/")
    
    if not folders:
        print("No folders found inside 'images/'. Did the download script finish?")
    else:
        for f in folders:
            create_true_color_png(f)
