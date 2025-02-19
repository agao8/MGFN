# CMPE 258 Project: Video Level Anomaly Detection using Contrastive Learning
This project was done independently by Andrew Gao

This project was built off the work done by Chen et al. Their work can be found at https://github.com/carolchenyx/MGFN.


### Pretrained models available in the saved_models folder

**Extracted I3d features for UCF-Crime dataset**

[**UCF-Crime train I3d features on Google drive**](https://drive.google.com/file/d/16LumirTnWOOu8_Uh7fcC7RWpSBFobDUA/view?usp=sharing)  

[**UCF-Crime test I3d features on Google drive**](https://drive.google.com/drive/folders/1QCBTDUMBXYU9PonPh1TWnRtpTKOX-fxr?usp=sharing)  

Dataset obtained from https://github.com/tianyu0207/RTFM/tree/main

#### Prepare the environment: 
        pip install -r requirement.txt
#### Test: Run 
        python test.py
#### Dataset Prepare: [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE). Rename the data path in ucf-i3d.list and ucf-i3d-test.list based on your data path.
#### Train: Modify the option.py and run 
        python main.py
