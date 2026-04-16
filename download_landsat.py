import os
import time
import json
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.windows import from_bounds
import pystac_client
import planetary_computer
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# THE ORIGINAL BOX FROM YOUR main.py
# bbox = box(minx, miny, maxx, maxy) -> box(574659, 2110016, 582115, 2118130)
UTM_MIN_X = 574659
UTM_MIN_Y = 2110016
UTM_MAX_X = 582115
UTM_MAX_Y = 2118130

# The EPSG Code of your target area (Likely UTM Zone 14N for Central Mexico "mancha_urbana")
TARGET_EPSG = "EPSG:32614"

# Let's dynamically project your UTM coordinates into WGS84 Lat/Lon (EPSG:4326) which STAC requires!
from pyproj import Transformer
transformer = Transformer.from_crs(TARGET_EPSG, "EPSG:4326", always_xy=True)
min_lon, min_lat = transformer.transform(UTM_MIN_X, UTM_MIN_Y)
max_lon, max_lat = transformer.transform(UTM_MAX_X, UTM_MAX_Y)

BBOX_LATLON = [min_lon, min_lat, max_lon, max_lat]

START_YEAR = 2000
END_YEAR = datetime.today().year

OUTPUT_DIR = Path("images")
HISTORY_FILE = OUTPUT_DIR / "download_history.json"

MAX_CLOUD_COVER = 20  # Only consider images with less than 20% clouds
IMAGES_PER_MONTH = 2  # Keep the top N clearest images per month

# Bands we care about mapping to STAC assets
BANDS_MAPPING = {
    "landsat-5": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir08",
        "swir1": "swir16",
        "swir2": "swir22"
    },
    "landsat-8": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir08",
        "swir1": "swir16",
        "swir2": "swir22"
    },
    "landsat-9": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir08",
        "swir1": "swir16",
        "swir2": "swir22"
    }
}


# ==============================================================================
# HELPERS
# ==============================================================================

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60), 
    stop=stop_after_attempt(5), 
    retry=retry_if_exception_type(Exception)
)
def fetch_stac_items(catalog, start_date, end_date):
    """Fetch STAC items with retries for a specific date range."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Querying STAC API for {start_date} to {end_date}...")
    
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX_LATLON,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {"lt": MAX_CLOUD_COVER},
            "platform": {"in": ["landsat-5", "landsat-8", "landsat-9"]}
        }
    )
    items = list(search.items())
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(items)} scenes in this range.")
    return items

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60), 
    stop=stop_after_attempt(5)
)
def crop_and_download_band(url, output_path):
    """Reads a remote COG URL window and saves it to output_path."""
    # We must transform the BBOX_LATLON (EPSG:4326) into the image's native CRS to crop it correctly
    try:
        from rasterio.warp import transform_bounds
        
        with rasterio.Env(GDAL_HTTP_RETRY_DELAY=5, GDAL_HTTP_MAX_RETRIES=5):
            with rasterio.open(url) as src:
                # Transform standard Lat/Lon bounding box to the TIF's native CRS
                native_bbox = transform_bounds('EPSG:4326', src.crs, *BBOX_LATLON)
                
                # Get the window corresponding to the bounding box
                window = from_bounds(*native_bbox, transform=src.transform)
                
                # Read only the data within the window
                meta = src.meta.copy()
                data = src.read(1, window=window)
                
                # Update metadata shape and transform for the new cropped image
                meta.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

        # Make sure we don't save empty/all-nan arrays if the bounding box somehow misses the valid data area
        import numpy as np
        if np.all(data == src.nodata):
            print(f"Skipping {output_path.name} (only nodata in bounding box)")
            return False

        # --- SURFACE REFLECTANCE TRANSFORMATION ---
        # USGS Landsat Collection 2 Level-2 scaling: SR = (DN * 0.0000275) - 0.2
        # We process it as float32 to keep precision and handle missing pixels as NaN
        data_float = np.where(
            data == src.nodata,
            np.nan,  # Ensure empty edges/clouds masks are registered strictly as NaN
            (data.astype(np.float32) * 0.0000275) - 0.2
        )

        # Skip images that are on the edge of the satellite path (e.g., more than 5% missing data)
        nan_ratio = np.isnan(data_float).sum() / data_float.size
        if nan_ratio > 0.05:
            print(f"       Skipping {output_path.name}: {nan_ratio:.1%} missing data (satellite edge).")
            return False

        meta.update({
            "dtype": "float32",
            "nodata": np.nan
        })

        # Save to disk as the mathematically ready float array
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data_float, 1)
        
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise e


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    
    # 1. Open the Microsoft Planetary Computer Catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 2. Iterate year by year to avoid exhausting STAC limits in one query
    for year in range(START_YEAR, END_YEAR + 1):
        year_str = str(year)
        
        # Skip if year is fully completed in history
        if history.get(year_str) == "COMPLETED":
            print(f"Year {year} is already completed. Skipping.")
            continue
            
        print(f"\n======================================")
        print(f"Processing Year: {year}")
        print(f"======================================")
        
        # Initialize tracking for the year if not present
        if year_str not in history:
            history[year_str] = {}
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        try:
            # Query the scenes
            scenes = fetch_stac_items(catalog, start_date, end_date)
            
            # Organize scenes by YYYY-MM
            scenes_by_month = defaultdict(list)
            for item in scenes:
                date_obj = item.datetime
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                scenes_by_month[month_key].append(item)
                
            # Process each month
            for month_key in sorted(scenes_by_month.keys()):
                
                # Skip if month is already processed
                if history[year_str].get(month_key) == "COMPLETED":
                    continue
                    
                # Sort scenes by cloud cover (lowest first)
                monthly_scenes = scenes_by_month[month_key]
                monthly_scenes.sort(key=lambda x: x.properties["eo:cloud_cover"])
                
                # Take top N clearest
                target_scenes = monthly_scenes[:IMAGES_PER_MONTH]
                
                for scene_idx, scene in enumerate(target_scenes):
                    platform = scene.properties["platform"]
                    acq_date = scene.datetime.strftime("%Y%m%d")
                    cloud_cover = scene.properties["eo:cloud_cover"]
                    
                    # Create directory for the scene
                    scene_dir = OUTPUT_DIR / f"{acq_date}_{platform}_p{scene_idx}"
                    scene_dir.mkdir(exist_ok=True)
                    
                    print(f"  -> Downloading {platform} for {acq_date} ({cloud_cover:.1f}% clouds)...")
                    
                    # Save the metadata cleanly
                    metadata_path = scene_dir / "metadata_stac.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(scene.to_dict(), f, indent=4)
                        
                    # Find and map the required bands
                    band_map = BANDS_MAPPING.get(platform)
                    if not band_map:
                        continue # Should not happen based on STAC query, but safety first
                    
                    success_all_bands = True
                    for friendly_name, stac_asset_key in band_map.items():
                        out_path = scene_dir / f"{friendly_name}.tif"
                        
                        # Only download if we don't have it
                        if not out_path.exists():
                            if stac_asset_key in scene.assets:
                                asset_url = scene.assets[stac_asset_key].href
                                try:
                                    crop_success = crop_and_download_band(asset_url, out_path)
                                    if not crop_success:
                                        print(f"       Warning: No overlapping data in {friendly_name}.")
                                        success_all_bands = False
                                    
                                    # Very polite rate limit to Microsoft API / AWS S3
                                    time.sleep(1.5)
                                    
                                except Exception as e:
                                    print(f"       Failed to download {friendly_name}: {e}")
                                    success_all_bands = False
                                    break
                            else:
                                print(f"       Asset {stac_asset_key} missing in STAC item!")
                                success_all_bands = False
                                break
                    
                    if success_all_bands:
                        print(f"       Successfully completed {acq_date}")
                    else:
                        print(f"       Failed/Incomplete for {acq_date}")
                
                # Mark month as done, save intermediate state
                history[year_str][month_key] = "COMPLETED"
                save_history(history)
                
                # Polite pause between chunks
                time.sleep(5)
                
            # If all months went through without major crashes, mark year done
            history[year_str] = "COMPLETED"
            save_history(history)
            time.sleep(15)  # Heavy pause between years to reset any API throttle states

        except Exception as e:
            print(f"CRITICAL ERROR on year {year}: {e}")
            print(f"Saving history and exiting. Rerun script to resume from this point.")
            save_history(history)
            break
            
if __name__ == "__main__":
    print("WARNING: This script requires 'pystac-client', 'planetary-computer', 'rasterio', and 'tenacity'.")
    print("Install via: pip install pystac-client planetary-computer rasterio tenacity")
    main()
