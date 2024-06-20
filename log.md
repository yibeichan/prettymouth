
## Group ICA

code
```
feat group_ica[#].fsf
```

### 20 ICs
Eyeball output

- IC1 (7.74%): veins
- IC2 (6.65%): PCC?
- IC3 (5.95%): DMN?
- IC4 (5.82%): noise?
- IC5 (5.75%): visual cortex?
- IC6 (5.73%): noise
- IC7 (5.51%): noise
- IC8 (5.17%): noise
- IC9 (5.01%): noise; edge
- IC10 (4.84%): PFC/DMN?
- IC11 (4.77%): noise; edge
- IC12 (4.70%): STG/DMN?
- IC13 (4.50%): lateral prefrontal cortex?
- IC14 (4.40%): PCC?
- IC15 (4.26%): noise
- IC16 (4.06%): noise
- IC17 (3.93%): ?
- IC18 (3.89%): noise
- IC19 (3.76%): DMN?
- IC20 (3.59%): noise


## Dual Regression

code
```
dual_regression /home/yc/GitHub/prettymouth/output/group_ica20.gica/groupmelodic.ica/melodic_IC.nii.gz 1 /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.mat /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.con 5000 /home/yc/GitHub/prettymouth/output/dual_reg20 `cat /home/yc/GitHub/prettymouth/output/group_ica20.gica/.filelist`

dual_regression /home/yc/GitHub/prettymouth/output/group_ica30.gica/groupmelodic.ica/melodic_IC.nii.gz 1 /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.mat /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.con 5000 /home/yc/GitHub/prettymouth/output/dual_reg30 `cat /home/yc/GitHub/prettymouth/output/group_ica30.gica/.filelist`

dual_regression /home/yc/GitHub/prettymouth/output/group_ica40.gica/groupmelodic.ica/melodic_IC.nii.gz 1 /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.mat /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.con 5000 /home/yc/GitHub/prettymouth/output/dual_reg40 `cat /home/yc/GitHub/prettymouth/output/group_ica40.gica/.filelist`

dual_regression /home/yc/GitHub/prettymouth/output/group_ica50.gica/groupmelodic.ica/melodic_IC.nii.gz 1 /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.mat /home/yc/GitHub/prettymouth/code/fsl_designmatrix/dual_reg_dm/dual_reg_dm.con 5000 /home/yc/GitHub/prettymouth/output/dual_reg50 `cat /home/yc/GitHub/prettymouth/output/group_ica50.gica/.filelist`
```


## Identify networks-of-interest

In the Melodic guide, we used `fslcc` to do the reference network correlations. We can pipe that command into a series of other commands and end up with a list of networks to use later as networks of interest:

```
fslcc --noabs -p 3 -t .204 output/yeo_MNI152_2mm_4d.nii.gz output/group_ica20.gica/groupmelodic.ica/melodic_IC.nii.gz | tr -s ' ' | cut -d ' ' -f 3 | sort -u | awk '{ printf "%02d\n", $1 - 1 }' >> output/gica20_nets_of_interest.txt

fslcc --noabs -p 3 -t .204 output/yeo_MNI152_2mm_4d.nii.gz output/group_ica30.gica/groupmelodic.ica/melodic_IC.nii.gz | tr -s ' ' >> output/gica30_nets_of_interest.txt

fslcc --noabs -p 3 -t .204 output/yeo_MNI152_2mm_4d.nii.gz output/group_ica40.gica/groupmelodic.ica/melodic_IC.nii.gz | tr -s ' ' >> output/gica40_nets_of_interest.txt
```


| No | Netwotk Name  | Physiological Interpretation |
|----|:-------------:|-----------------------------:|
| 1  | VIS-1         | Visual                       |
| 2  | VIS-2         |                              |
| 3  | MOT-1         | Motor                        |
| 4  | MOT-2         |                              |
| 5  | DAN-2         | Dorsal Attention             |
| 6  | DAN-1         |                              |
| 7  | VAN-1         | Ventral Attention            |
| 8  | FP-1          | Frontoparietal               |
| 9  | LIM-1         | Limbic                       |
| 10 | LIM-2         |                              |
| 11 | FP-2          | Frontoparietal               |
| 12 | FP-3          |                              |
| 13 | FP-4          |                              |
| 14 | MOT-3         | Motor                        |
| 15 | DMN-3         | Default Mode                 |
| 16 | DMN-1         |                              |
| 17 | DMN-2         |                              |



