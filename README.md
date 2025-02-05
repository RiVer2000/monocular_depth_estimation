# Monocular Depth Estimation

Download the NYU-DepthV2 dataset and put it in the folder as shown below

monocular_depth_estimation
├── classes.py
├── environment.yml
├── main.py
├── model1.pth
├── nyu_data
│   ├── data
│   │   ├── nyu2_test
│   │   ├── nyu2_train
│   │   └── nyu2_train.csv
├── README.md
├── requirements.txt
├── train_kitti.ipynb
└── train_nyu.py

To Run the code add create a conda environment using the environment.yml file as
```
conda env create --file environment.yml -n depth
conda install --yes --file requirements.txt
```
- To run the training on NYU-Depthv2 dataset activate the conda environment and run the file train_nyu.py
- To run the training on KITTI dataset run the train_kitti.ipynb notebook
- To do a real time depth estimation run the file main.py in the conda environment with the desired model file.(NYU Depth model is preloaded)
## Results
![Screenshot from 2024-12-18 21-42-18](https://github.com/user-attachments/assets/2ea6b37d-a159-4ee7-bc57-9192a46062c0)
