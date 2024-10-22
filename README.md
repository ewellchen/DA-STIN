# Deep Domain Adversarial Adaptation for Photon-efficient Imaging

This repository is the offcial code of the paper published on Physical Review Applied: http://dx.doi.org/10.1103/PhysRevApplied.18.054048


**Abstract:**

Photon-efficient imaging with the single-photon light detection and ranging captures the three-dimensional structure of a scene by only a few detected signal photons per pixel. However, the existing computational methods for photon-efficient imaging are pretuned on a restricted scenario or trained on simulated datasets. When applied to realistic scenarios whose signal-to-background ratios and other hardware-specific properties differ from those of the original task, the model performance often significantly deteriorates. In this paper, we present a domain adversarial adaptation design to alleviate this domain shift problem by exploiting unlabeled real-world data, with significant resource savings. This method demonstrates superior performance on simulated and real-world experiments using our home-built up-conversion single-photon imaging system, which provides an efficient approach to bypass the lack of ground-truth depth information in implementing computational imaging algorithms for realistic applications.

## Usage

### Requirements

- torch>=1.0.0
- torchvision>=0.2.0
- opencv-python==4.5.3

### Data simulation
Code is aviliable at https://www.computationalimaging.org/publications/single-photon-3d-imaging-with-deep-sensor-fusion

### Training

To train STIN on simulated dataset, generate training data by simulate.m and then run:
```
python train_sim.py
```
To train adversarial STIN by DANN on simulated dataset, generate source and target training data by simulate.m and then run:
```
python train_sim_adver.py
```

### Evaluating

To test STIN on simulated dataset, run:
```
python test_sim.py
```
To test adversarial STIN by DANN on simulated dataset, run:
```
python test_sim_adver.py
```

## License
MIT License

## Citation
If you find our work useful in your research, please consider citing:
```
@article{chen2022deep,
  title={Deep Domain Adversarial Adaptation for Photon-efficient Imaging},
  author={Chen, Yiwei and Yao, Gongxin and Liu, Yong and Pan, Yu},
  journal={arXiv preprint arXiv:2201.02475},
  year={2022}
}

@article{chen2022deep,
  title = {Deep Domain Adversarial Adaptation for Photon-Efficient Imaging},
  author = {Chen, Yiwei and Yao, Gongxin and Liu, Yong and Su, Hongye and Hu, Xiaomin and Pan, Yu},
  journal = {Phys. Rev. Applied},
  volume = {18},
  issue = {5},
  pages = {054048},
  numpages = {8},
  year = {2022},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.18.054048},
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.18.054048}
}
