# LPM_Python

A Python implementation of the Locality Preserving Matching (LPM) method for pruning outliers in image matching.

The code is established according to the MATLAB version https://github.com/jiayi-ma/LPM and supposed to have the same output and similar time cost. The parameters are tunable inside the function LPM_filter in LPM.py.

If you find this code useful for your research, plese cite the paper:

```
@article{ma2019locality,
  title={Locality preserving matching},
  author={Ma, Jiayi and Zhao, Ji and Jiang, Junjun and Zhou, Huabing and Guo, Xiaojie},
  journal={International Journal of Computer Vision},
  volume={127},
  number={5},
  pages={512--531},
  year={2019},
  publisher={Springer}
}
```

# USAGE

Dependencies: numpy and sklearn packages are required for the core function LPM_filter,

opencv-python and scipy are additionally required to run the demo.

After installing dependencies, just run 
```
python demo.py 
``` 
for a simple example.
