import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, IntSlider, interactive_output, VBox, Checkbox
from IPython.display import display
import re
import glob

class ImageVisualizer:
    def __init__(self, prediction_dir, ground_truth_dir, ct_images_dir):
        self.prediction_dir = prediction_dir
        self.ground_truth_dir = ground_truth_dir
        self.ct_images_dir = ct_images_dir
        
        # Extract base names for the dropdown options
        file_names = os.listdir(prediction_dir)
        base_names = set(f.split('.')[0].rsplit('_', 1)[0] for f in file_names if f.endswith('.nii') or f.endswith('.nii.gz'))
        self.common_files = sorted(list(base_names))
        
        self.file_name_widget = Dropdown(options=self.common_files)
        self.slice_idx_widget = IntSlider(min=0, max=1, step=1, value=0, description='Slice Index')
        self.window_level_widget = IntSlider(min=-1000, max=1000, step=1, value=40, description='Window Level')
        self.window_width_widget = IntSlider(min=-1000, max=2000, step=1, value=80, description='Window Width')
        self.overlay_toggle = Checkbox(value=True, description='Show Overlay')
        self.file_name_widget.observe(self.update_slice_idx_range, 'value')

        self.update_slice_idx_range()  # Initial call to set slice index range

    def apply_window(self, image, level, width):
        lower = level - (width / 2)
        upper = level + (width / 2)
        return np.clip((image - lower) / (upper - lower), 0, 1)

    def find_matching_file(self, dir, base_name):
        """Find a file in `dir` that matches `base_name`."""
        for f in sorted(os.listdir(dir)):
            if f.startswith(base_name) and (f.endswith('.nii') or f.endswith('.nii.gz')):
                return os.path.join(dir, f)
        return None

    def plot_images(self, base_name, slice_idx, window_level, window_width, show_overlay):
        # Find corresponding files
        prediction_file_path = self.find_matching_file(self.prediction_dir, base_name)
        ground_truth_file_path = self.find_matching_file(self.ground_truth_dir, base_name)
        ct_image_file_path = self.find_matching_file(self.ct_images_dir, base_name)

        if not all([prediction_file_path, ground_truth_file_path, ct_image_file_path]):
            print("One or more files could not be found for the patient:", base_name)
            return

        # Load and convert the data to numpy arrays
        prediction_data = nib.load(prediction_file_path).get_fdata()
        ground_truth_data = nib.load(ground_truth_file_path).get_fdata()
        ct_data = nib.load(ct_image_file_path).get_fdata()

        # Adjust slice_idx if it's out of bounds for any of the images
        max_slices = min(prediction_data.shape[2], ground_truth_data.shape[2], ct_data.shape[2])
        slice_idx = min(slice_idx, max_slices - 1)

        # Apply windowing to the CT head image
        ct_data_windowed = self.apply_window(ct_data[:, :, slice_idx], window_level, window_width)

        # Plot the images
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(ground_truth_data[:, :, slice_idx], cmap='gray')
        axes[0].set_title('Ground Truth Image')
        axes[1].imshow(prediction_data[:, :, slice_idx], cmap='gray')
        axes[1].set_title('Prediction')
        axes[2].imshow(ct_data_windowed, cmap='gray')
        if show_overlay:
            axes[2].imshow(ground_truth_data[:, :, slice_idx], cmap='hot', alpha=0.5)
        axes[2].set_title('CT with Ground Truth Overlay')
        plt.show()

    def update_slice_idx_range(self, *args):
        base_name = self.file_name_widget.value
        ct_image_file_path = self.find_matching_file(self.ct_images_dir, base_name)

        if ct_image_file_path:
            ct_img = nib.load(ct_image_file_path)
            ct_data = ct_img.get_fdata()
            self.slice_idx_widget.max = ct_data.shape[2] - 1
            self.slice_idx_widget.value = min(self.slice_idx_widget.value, self.slice_idx_widget.max)
        else:
            print("CT image file could not be found for the patient:", base_name)

    def display(self):
        # Link widgets to plot function
        out = interactive_output(self.plot_images, {
            'base_name': self.file_name_widget, 
            'slice_idx': self.slice_idx_widget, 
            'window_level': self.window_level_widget, 
            'window_width': self.window_width_widget, 
            'show_overlay': self.overlay_toggle
        })
        
        # Display the widgets and the output together
        display(VBox([self.file_name_widget, self.slice_idx_widget, self.window_level_widget, self.window_width_widget, self.overlay_toggle, out]))


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


def convert_nii_to_niigz(input_dir, output_dir=None):
    """
    Converts all .nii files in the input directory to .nii.gz format.

    Parameters:
    - input_dir: The directory containing .nii files.
    - output_dir: Optional. The directory where .nii.gz files will be saved. If None, saves in the input directory.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith('.nii'):
            # Construct the full file path for the .nii file
            file_path = os.path.join(input_dir, file_name)
            # Load the .nii file
            nii_image = nib.load(file_path)
            # Construct the output file path with .nii.gz extension
            output_file_path = os.path.join(output_dir, file_name + '.gz')
            # Save the image in .nii.gz format
            nib.save(nii_image, output_file_path)
            print(f'Converted and saved: {output_file_path}')

def delete_niigz_files(directory):
    """
    Deletes all files with the .nii.gz extension in the specified directory.

    Args:
    directory (str): The path to the directory where the files are located.
    """
    # Create the full path pattern to find .nii.gz files
    pattern = os.path.join(directory, '**', '*.nii.gz')
    
    # Use glob to find all files matching the pattern
    files = glob.glob(pattern, recursive=True)
    
    # Iterate over the list of file paths
    for file_path in files:
        try:
            os.remove(file_path)  # Attempt to remove the file
            print(f'Deleted: {file_path}')  # Print confirmation of deletion
        except Exception as e:
            print(f'Failed to delete {file_path}: {e}')  # Print any error messages

def delete_nii_files(directory):
    """
    Deletes all files with the .nii extension in the specified directory.

    Args:
    directory (str): The path to the directory where the files are located.
    """
    # Create the full path pattern to find .nii files
    pattern = os.path.join(directory, '**', '*.nii')
    
    # Use glob to find all files matching the pattern
    files = glob.glob(pattern, recursive=True)
    
    # Iterate over the list of file paths
    for file_path in files:
        try:
            os.remove(file_path)  # Attempt to remove the file
            print(f'Deleted: {file_path}')  # Print confirmation of deletion
        except Exception as e:
            print(f'Failed to delete {file_path}: {e}') 

def convert_niigz_to_nii(input_dir, output_dir=None):
    """
    Converts all .nii.gz files in the input directory to .nii format.

    Parameters:
    - input_dir: The directory containing .nii.gz files.
    - output_dir: Optional. The directory where .nii files will be saved. If None, saves in the input directory.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith('.nii.gz'):
            # Construct the full file path for the .nii.gz file
            file_path = os.path.join(input_dir, file_name)
            # Load the .nii.gz file
            niigz_image = nib.load(file_path)
            # Construct the output file path with .nii extension
            # Removing the '.gz' extension from the file name
            output_file_path = os.path.join(output_dir, file_name[:-3])
            # Save the image in .nii format
            nib.save(niigz_image, output_file_path)
            print(f'Converted and saved: {output_file_path}')