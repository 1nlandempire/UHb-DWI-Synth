from os.path import join
import numpy as np
from dipy.io.image import load_nifti, save_nifti

sub_path = '/data/wtl/HCP_MGH/mgh_1001/'

dwi_b1k, affine = load_nifti(join(sub_path, 'dwi_b1k.nii.gz'), return_img=False)
dwi_b3k, affine = load_nifti(join(sub_path, 'dwi_b3k.nii.gz'), return_img=False)
dwi_b5k, affine = load_nifti(join(sub_path, 'dwi_b5k.nii.gz'), return_img=False)
dwi_b10k, affine = load_nifti(join(sub_path, 'dwi_b10k.nii.gz'), return_img=False)

dwi_sh_b1k, affine = load_nifti(join(sub_path, 'DWI_SH_b1k.nii.gz'), return_img=False)
dwi_sh_b3k, affine = load_nifti(join(sub_path, 'DWI_SH_b3k.nii.gz'), return_img=False)
dwi_sh_b5k, affine = load_nifti(join(sub_path, 'DWI_SH_b5k.nii.gz'), return_img=False)
dwi_sh_b10k, affine = load_nifti(join(sub_path, 'DWI_SH_b10k.nii.gz'), return_img=False)

mask, _ = load_nifti(join(sub_path, 'dwi_mask.nii.gz'), return_img=False)
mask = np.expand_dims(mask, axis=-1)

dwi_b1k *= mask
dwi_b3k *= mask
dwi_b5k *= mask
dwi_b10k *= mask

dwi_sh_b1k *= mask
dwi_sh_b3k *= mask
dwi_sh_b5k *= mask
dwi_sh_b10k *= mask

save_nifti(join(sub_path, 'dwi_b1k_masked.nii.gz'), dwi_b1k, affine)
save_nifti(join(sub_path, 'dwi_b3k_masked.nii.gz'), dwi_b3k, affine)
save_nifti(join(sub_path, 'dwi_b5k_masked.nii.gz'), dwi_b5k, affine)
save_nifti(join(sub_path, 'dwi_b10k_masked.nii.gz'), dwi_b10k, affine)

save_nifti(join(sub_path, 'DWI_SH_b1k_masked.nii.gz'), dwi_sh_b1k, affine)
save_nifti(join(sub_path, 'DWI_SH_b3k_masked.nii.gz'), dwi_sh_b3k, affine)
save_nifti(join(sub_path, 'DWI_SH_b5k_masked.nii.gz'), dwi_sh_b5k, affine)
save_nifti(join(sub_path, 'DWI_SH_b10k_masked.nii.gz'), dwi_sh_b10k, affine)