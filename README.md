
![Logo](https://raw.githubusercontent.com/yokahealthcare/C-Eyes-App-New/master/thumbnail.png)


# C-Eyes (Camera Eyes) - Precision in Vision


Welcome! We are a dedicated team at Popuri, committed to addressing the challenges faced by the visually impaired in understanding and navigating their surroundings.

Introducing our groundbreaking solution, the "C-Eyes" app, where 'C' stands for 'Camera-Eyes.' By harnessing the power of advanced Depth Mapping and immersive 8D Sound technologies, we've revolutionized the way blind individuals perceive their environment.

With the C-Eyes app, we can precisely estimate distances in real-time and provide users with vital information through immersive 8D Audio feedback. This feedback is seamlessly delivered via a headset or earbuds, enhancing the user's spatial awareness and overall experience.

Our mission is to empower the visually impaired, giving them the ability to navigate their world with newfound confidence and independence. Join us in making a difference and opening up new possibilities for those who need it most.




## Authors

Popuri Teams


## IMPORTANT

- Internet access is mandatory
- The program display is slightly different (updated & optimized) from screenshot below
- **Run in GPU is priotized**, if you have GPU attached to your computer, the program will use that
- **The files attached for setup anaconda environment are for Windows only**
## Installation
We are using anaconda environment for this project, you can download in [here](https://www.anaconda.com/download)

### 1. Creating Environment

- Open Anaconda Prompt 

- Locate to the "C-Eyes" Project Directory
```bash
  cd <path to C-Eyes directory>
```
    
- Create an environment

```bash
  conda env create -f environment.yml
```

Wait until the process is completed

**If there is an error like this,**
```
ERROR conda.core.link:_execute(945): An error occurred while installing package 'defaults::vs2015_runtime-14.27.29016-h5e58377_2'.
Rolling back transaction: done

[Errno 13] Permission denied
```

**[Solution 1] You can try re-run the 'Create an environment' command again**

Make sure to **delete the created environment first** (although error occurs, ananconda still managed to create the environment)

using this command,

```
  conda env remove -n ceyes
```

**[Solution 2] Install the package manually**

Create an empty environment (without any python package)

```
  conda create -n ceyes
  conda activate ceyes
  conda install pip
  pip install opencv-python mtcnn-opencv timm numpy pygame pyfiglet yaml
```


### 2. Running Environment

Activate newly created environment

```bash
  conda activate ceyes
```

**Note**: If you run "solution 2" you don't need to run this again

### 3. Installing PyTorch 

**IMPORTANT** : Make sure there is no installed pytorch on the environment 
```bash
  pip3 uninstall torch torchvision torchaudio
```

Download & Install the latest pytorch version with GPU support
```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Wait until the process is completed
## Download Assets

### YOLOv5 (automatic)

In this project we use the YOLOv5 pre-trained model

- Nano (poor): [yolov5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)
- Small (ok): [yolov5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
- Medium (medium): [yolov5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)
- Large (good): [yolov5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)
- Xtra Large (excellent, GPU recommended): [yolov5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)

***automatic :** it will be downloaded in the program 


### MiDAS (mandatory manually)

In this project we use the MiDAS pre-trained model

MiDaS 3.1

- For highest quality (recommended): [dpt_large_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt) [~1.28 GB]
- For moderately less quality: [dpt_hybrid_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt) [~470 MB]
- For poor quality: [midas_v21_small_256](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt) [~80 MB]


**After downloading move the file to the "/weights" folder**







## Deployment

To run this project run

```bash
  python main.py
```


## Screenshots

Focal Length Distance Estimation

![Focal Length Distance Estimation](https://raw.githubusercontent.com/yokahealthcare/C-Eyes-App-New/master/production/result/focal%20length.PNG)

Depth Map Distance Estimation

![Depth Map Distance Estimation](https://raw.githubusercontent.com/yokahealthcare/C-Eyes-App-New/master/production/result/depth%20map.PNG)

