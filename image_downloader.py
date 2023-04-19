from bing_image_downloader import downloader
import os

folder = 'data'
sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
for subfolder in sub_folders:
    downloader.download(subfolder, limit=30, output_dir='data', adult_filter_off=False, force_replace=False, timeout=60, verbose=True)
