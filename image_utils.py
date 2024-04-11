import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, IntSlider, interactive_output, VBox
from IPython.display import display


class ImageVisualizer:
    def __init__(self, prediction_dir, ground_truth_dir, ct_images_dir):
        self.prediction_dir = prediction_dir
        self.ground_truth_dir = ground_truth_dir
        self.ct_images_dir = ct_images_dir
        self.common_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        self.file_name_widget = Dropdown(options=self.common_files)
        self.slice_idx_widget = IntSlider(min=0, max=1, step=1, value=0)
        self.file_name_widget.observe(self.update_slice_idx_range, 'value')
        
        self.update_slice_idx_range()  # Initial call to set slice index range

    def apply_window(self, image, level=40, width=80):
        lower = level - (width / 2)
        upper = level + (width / 2)
        return np.clip((image - lower) / (upper - lower), 0, 1)

    def plot_images(self, file_name, slice_idx):
        # Construct file paths
        prediction_file_path = os.path.join(self.prediction_dir, file_name)
        ground_truth_file_path = os.path.join(self.ground_truth_dir, file_name)
        ct_image_file_path = os.path.join(self.ct_images_dir, file_name)

        # Load the files
        prediction_img = nib.load(prediction_file_path)
        ground_truth_img = nib.load(ground_truth_file_path)
        ct_img = nib.load(ct_image_file_path)

        # Convert the data to numpy arrays
        prediction_data = prediction_img.get_fdata()
        ground_truth_data = ground_truth_img.get_fdata()
        ct_data = ct_img.get_fdata()

        # Adjust slice_idx if it's out of bounds for any of the images
        max_slices = min(prediction_data.shape[2], ground_truth_data.shape[2], ct_data.shape[2])
        slice_idx = min(slice_idx, max_slices - 1)

        # Apply custom windowing to the CT head image
        ct_data_windowed = self.apply_window(ct_data)

        # Plot the images
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(ground_truth_data[:, :, slice_idx], cmap='gray')
        axes[0].set_title('Ground Truth Image')
        axes[1].imshow(prediction_data[:, :, slice_idx], cmap='gray')
        axes[1].set_title('Prediction')
        axes[2].imshow(ct_data_windowed[:, :, slice_idx], cmap='gray')
        axes[2].imshow(ground_truth_data[:, :, slice_idx], cmap='hot', alpha=0.5)
        axes[2].set_title('CT with Ground Truth Overlay')
        plt.show()

    def update_slice_idx_range(self, *args):
        ct_img = nib.load(os.path.join(self.ct_images_dir, self.file_name_widget.value))
        ct_data = ct_img.get_fdata()
        self.slice_idx_widget.max = ct_data.shape[2] - 1
        self.slice_idx_widget.value = min(self.slice_idx_widget.value, self.slice_idx_widget.max)

    def display(self):
        # Link widgets to plot function without auto-creating them
        out = interactive_output(self.plot_images, {'file_name': self.file_name_widget, 'slice_idx': self.slice_idx_widget})
        
        # Display the manual widgets and the output together
        display(VBox([self.file_name_widget, self.slice_idx_widget, out]))


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