import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk

from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def quantize_maps(source_dir, target_dir, quantization_levels=5):
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file_name in tqdm(sorted(os.listdir(source_dir))):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            # Load the NIfTI file
            file_path = os.path.join(source_dir, file_name)
            nii = nib.load(file_path)
            image_data = nii.get_fdata()

            # Identify true background pixels
            background_mask = image_data == 0

            # Normalize intensities to 1-255 for non-background pixels
            non_background = image_data > 0
            max_val = image_data[non_background].max()
            min_val = image_data[non_background].min()
            normalized_data = np.zeros_like(image_data)
            normalized_data[non_background] = 1 + (image_data[non_background] - min_val) / (max_val - min_val) * 254

            # Quantize intensities into specified levels above 0
            quantization_step = 255 / quantization_levels
            quantized_data = np.ceil(normalized_data / quantization_step)

            # Apply Gaussian smoothing
            smoothed_data = gaussian_filter(quantized_data, sigma=1.5)

            # Re-quantize after smoothing to ensure specified levels above 0
            re_quantized_data = np.round(smoothed_data)
            final_data = np.clip(re_quantized_data, 0, quantization_levels).astype(np.int16)  # Ensure values are within [0, quantization_levels]

            # Ensure true background remains 0
            final_data[background_mask] = 0

            # Create a new NIfTI image, ensuring to preserve the original header
            new_nii = nib.Nifti1Image(final_data, affine=nii.affine, header=nii.header)

            new_file_path = os.path.join(target_dir, file_name)
            nib.save(new_nii, new_file_path)
            print(f"Processed and saved: {new_file_path}") 
            
def convert_series_to_nifti(input_directory, output_file):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_directory)
    reader.SetFileNames(dicom_names)
    image_series = reader.Execute()
    
    # Convert to numpy array to manipulate the pixel data directly
    img_array = sitk.GetArrayFromImage(image_series)
        
    # Convert the numpy array back to a SimpleITK Image
    processed_image = sitk.GetImageFromArray(img_array)
    processed_image.SetSpacing(image_series.GetSpacing())
    processed_image.SetOrigin(image_series.GetOrigin())
    processed_image.SetDirection(image_series.GetDirection())

    # Write the processed image as a NIfTI file
    sitk.WriteImage(processed_image, output_file)