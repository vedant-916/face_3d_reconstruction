# face_3d_reconstruction


## Getting Started

### Prerequisite

* Python 2.7 (numpy, skimage, scipy)

* TensorFlow >= 1.4

  Optional:

* dlib (for detecting face.  You do not have to install if you can provide bounding box information. )

* opencv2 (for showing results)

GPU is highly recommended. The run time is ~0.01s with GPU(GeForce GTX 1080) and ~0.2s with CPU(Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz).

### Usage

1. Clone the repository

```bash
git clone https://github.com/vedant-916/face_3d_reconstruction.git
cd face_3d_reconstruction
```

2. Run with your own images

   `python demo_OWN.py -i <inputDir> -o <outputDir> --isDlib True  `



