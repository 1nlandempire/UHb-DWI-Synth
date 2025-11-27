from dipy.io.image import load_nifti, save_nifti

dwi, _ = load_nifti('/data/wtl/Caffine/sub-015/dwi.nii.gz', return_img=False)
print(dwi.shape)