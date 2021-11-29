# STIN

This repository contains PyTorch implementation for STIN.


**Abstract:**

In single-photon LiDAR, photon-efficient imaging captures the 3D structure of a scene by only several detected signal photons per pixel. The existing deep learning models for this task are trained on simulated datasets, which poses the domain shift challenge when applied to realistic scenarios. In this paper, we propose a spatiotemporal inception network (STIN) for photon-efficient imaging, which is able to precisely predict the depth from a sparse and high-noise photon counting histogram by fully exploiting spatial and temporal information. Then the domain adversarial adaptation frameworks, including domain-adversarial neural network and adversarial discriminative domain adaptation, are effectively applied to STIN to alleviate the domain shift problem for realistic applications. Comprehensive experiments on the simulated data generated from the NYU~v2 and the Middlebury datasets demonstrate that STIN outperforms the state-of-the-art models at low signal-to-background ratios from 2:10 to 2:100. Moreover, experimental results on the real-world dataset captured by the single-photon imaging prototype show that the STIN with domain adversarial training achieves better generalization performance compared with the state-of-the-arts as well as the baseline STIN trained by simulated data.

## Usage

### Requirements

- torch>=1.0.0
- torchvision>=0.2.0
- opencv-python==4.5.3

Or just use the following code:

`pip install -r requirements.txt`


### Training

To train STIN, run:
```
python main.py
```

## License
MIT License

## Citation
If you find our work useful in your research, please consider citing:
```
