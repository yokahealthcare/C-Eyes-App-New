
![Logo](https://raw.githubusercontent.com/yokahealthcare/C-Eyes-App-New/master/thumbnail.png)


# C-Eyes (Camera Eyes) - Precision in Vision


Welcome! We are a dedicated team at Popuri, committed to addressing the challenges faced by the visually impaired in understanding and navigating their surroundings.

Introducing our groundbreaking solution, the "C-Eyes" app, where 'C' stands for 'Camera-Eyes.' By harnessing the power of advanced Depth Mapping and immersive 8D Sound technologies, we've revolutionized the way blind individuals perceive their environment.

With the C-Eyes app, we can precisely estimate distances in real-time and provide users with vital information through immersive 8D Audio feedback. This feedback is seamlessly delivered via a headset or earbuds, enhancing the user's spatial awareness and overall experience.

Our mission is to empower the visually impaired, giving them the ability to navigate their world with newfound confidence and independence. Join us in making a difference and opening up new possibilities for those who need it most.




## Authors

- [@yokahealthcare](https://github.com/yokahealthcare)


## IMPORTANT

Internet access is mandatory


## Installation

- Open Anaconda Prompt

- Locate to the "C-Eyes" Project Directory
```bash
  cd <path to C-Eyes directory>
```
    
- Create an environment

```bash
  conda create -f environment.yml
```

Wait until the process is completed

- Activate new environment

```bash
  conda activate ceyes
```
## Download Assets

In this project we use the MiDAS pre-trained model

MiDaS 3.1

- For highest quality: [dpt_beit_large_512](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt)
- For moderately less quality, but better speed-performance trade-off: [dpt_swin2_large_384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt)
- For embedded devices: [dpt_swin2_tiny_256](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt), [dpt_levit_224](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt)

MiDaS 3.0: Legacy transformer models [dpt_large_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt) and [dpt_hybrid_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt)

MiDaS 2.1: Legacy convolutional models [midas_v21_384](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt) and [midas_v21_small_256](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt)


**Default Included :** *midas_v21_small_256*

**After downloading move the file to the "/weights" folder**







## Deployment

To run this project run

```bash
  python main.py
```


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

