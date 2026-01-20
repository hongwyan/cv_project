import nibabel as nib

img = nib.load(r"C:\BraTS_Project\data\BraTS2021_Training_Data\BraTS2021_00000\BraTS2021_00000_flair.nii.gz")
print(img.shape)
