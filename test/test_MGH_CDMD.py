from dipy.io import read_bvals_bvecs
from os.path import join


sub_path = '/data/wtl/MGH-USC-CDMD/sub_001/dwi/'
bval, bvec = read_bvals_bvecs(join(sub_path, 'sub_001_dwi.bval'), join(sub_path, 'sub_001_dwi.bvec'))


count_b = {}
for b in bval:
    if b in count_b:
        count_b[b] = count_b[b] + 1
    else:
        count_b[b] = 1

for b in sorted(count_b):
    print(f'{b}:{count_b[b]}')