import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import box
from skimage import exposure
import matplotlib

def read_band(file_path):
    """Read a single band from a GeoTIFF file."""
    with rasterio.open(file_path) as src:
        band = src.read(1)
        no_data_value = src.nodata
        band = np.where(band == no_data_value, np.nan, band)
        return band, src.meta

def extract_metadata(metadata_path, satellite):
    """Extract scale factor and offset for each band from the metadata file."""
    metadata = {}
    with open(metadata_path, 'r') as file:
        for line in file:
            for band_num in range(1, 8 if satellite == 'L7' else 12):  # L7: 1-7, L8: 1-11
                if satellite == 'L7' and band_num == 6:  # Skip band 6 for Landsat 7
                    continue
                mult_key = f'REFLECTANCE_MULT_BAND_{band_num}' if satellite == 'L8' else f'RADIANCE_MULT_BAND_{band_num}'
                add_key = f'REFLECTANCE_ADD_BAND_{band_num}' if satellite == 'L8' else f'RADIANCE_ADD_BAND_{band_num}'
                if mult_key in line:
                    metadata[mult_key] = float(line.split('=')[1].strip())
                elif add_key in line:
                    metadata[add_key] = float(line.split('=')[1].strip())
    return metadata

def scale_image(image, scale_factor, offset):
    """Scale the image using the provided scale factor and offset."""
    # Apply the scale factor and offset
    scaled_image = image * scale_factor + offset
    # Normalize the image to the range 0 to 1
    scaled_image = (scaled_image - np.nanmin(scaled_image)) / (np.nanmax(scaled_image) - np.nanmin(scaled_image))
    return scaled_image

def calculate_index(bandA, bandB):
    """Calculate Normalized Difference Vegetation Index."""
    return (bandA - bandB) / (bandA + bandB + 1e-8)  # Add small number to avoid division by zero

def display_image(image, title, cmap=None):
    """Display the image with a given title."""
    plt.figure(figsize=(12, 12))
    if cmap:
        plt.imshow(image, cmap=cmap)
        plt.colorbar(label=title)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_raster(array, reference_path, output_path):
    """Save a numpy array as a GeoTIFF using the reference file's metadata."""
    with rasterio.open(reference_path) as src:
        kwargs = src.meta.copy()
    kwargs.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write(array.astype(rasterio.float32), 1)

def cut_image(image_path, bbox):
    """Cut the image using the provided bounding box coordinates."""
    with rasterio.open(image_path) as src:
        out_image, out_transform = mask(src, [bbox], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        return out_image[0], out_meta

def contrast_stretch(image):
    """Apply contrast stretching to the image."""
    p2, p98 = np.percentile(image, (2, 98))
    stretched_image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return stretched_image

def save_png(image, output_path, cmap='RdYlGn'):
    """Save the image as a PNG file."""
    matplotlib.image.imsave(output_path, image, cmap=cmap)

def main(file_paths, metadata_path, bbox, satellite, patron):
    # Read and cut bands
    bands = {}
    for band_name, file_path in file_paths.items():
        bands[band_name], _ = cut_image(file_path, bbox)
        print(f"Band {band_name} min: {np.nanmin(bands[band_name])}, max: {np.nanmax(bands[band_name])}")

    # Extract metadata
    metadata = extract_metadata(metadata_path, satellite)
    print("Metadata:", metadata)

    # Scale bands
    scaled_bands = {}
    if satellite == 'L7':
        band_numbers = [1, 2, 3, 4, 5, 7]  # Exclude band 6 for Landsat 7
        band_names = ['blue', 'green', 'red', 'nir', 'swir', 'swir2']
    else:  # Landsat 8
        band_numbers = [2, 3, 4, 5, 6, 7]
        band_names = ['blue', 'green', 'red', 'nir', 'swir', 'swir2']

    for band_num, band_name in zip(band_numbers, band_names):
        scale_factor = metadata[f'REFLECTANCE_MULT_BAND_{band_num}'] if satellite == 'L8' else metadata[f'RADIANCE_MULT_BAND_{band_num}']
        offset = metadata[f'REFLECTANCE_ADD_BAND_{band_num}'] if satellite == 'L8' else metadata[f'RADIANCE_ADD_BAND_{band_num}']
        scaled_bands[band_name] = scale_image(bands[band_name], scale_factor, offset)
        print(f"Scaled Band {band_name} min: {np.nanmin(scaled_bands[band_name])}, max: {np.nanmax(scaled_bands[band_name])}")

    # Calculate and display index
    index = calculate_index(scaled_bands['swir2'], scaled_bands['nir'])
    display_image(index, 'Calculated Index', cmap='RdYlGn')

    # Save the index image as a PNG
    index_output_path = f"images/{patron}_index.png"
    save_png(index, index_output_path, cmap='RdYlGn')
    print(f"Index image saved to {index_output_path}")

if __name__ == "__main__":
    # Define the bounding box coordinates (minx, miny, maxx, maxy)
    bbox = box(582115, 2110016, 574659, 2118130)
    
#######################################################################################################################################################   
#CAMBIAR AQUI
    # Define the initial pattern of the files
    patron = 'LC08_L2SP_025047_20210127_20210305_02_T1'
    satellite = 'L8'  # Change to 'L7' for Landsat 7
#######################################################################################################################################################
    # File paths
    if satellite == 'L7':
        file_paths = {
            'red': f'{patron}/{patron}_SR_B3.TIF',
            'green': f'{patron}/{patron}_SR_B2.TIF',
            'blue': f'{patron}/{patron}_SR_B1.TIF',
            'nir': f'{patron}/{patron}_SR_B4.TIF',
            'swir': f'{patron}/{patron}_SR_B5.TIF',
            'swir2': f'{patron}/{patron}_SR_B7.TIF',
            'thermal': f'{patron}/{patron}_ST_B6.TIF'  # Thermal band for Landsat 7
        }
    else:  # Landsat 8
        file_paths = {
            'red': f'{patron}/{patron}_SR_B4.TIF',
            'green': f'{patron}/{patron}_SR_B3.TIF',
            'blue': f'{patron}/{patron}_SR_B2.TIF',
            'nir': f'{patron}/{patron}_SR_B5.TIF',
            'swir': f'{patron}/{patron}_SR_B6.TIF',
            'swir2': f'{patron}/{patron}_SR_B7.TIF',
        }

    # Metadata path
    metadata_path = f'{patron}/{patron}_MTL.txt'

    # Run the main function
    main(file_paths, metadata_path, bbox, satellite, patron)