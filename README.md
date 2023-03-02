# [IDPCNN](https://doi.org/10.1016/j.cam.2021.113973): Iterative denoising and projecting CNN for MRI reconstruction



To run the code for MRI reconstruction, you could input the following code:
```bash
python main_test_pnpadmm.py --config_file ./options/SIAT_pnp_admm.yaml
```
in which the config file contains all of the parameters setting.

To test the denoiser, you could input the following code:
```bash
python main_test.py --config_file ./options/SIAT_dncnn_denoising.yaml
```
in which the config file contains all of the parameters setting.

The [train data](https://github.com/yqx7150/SIAT_MRIdata200) and [test data](https://github.com/yqx7150/EDAEPRec/tree/master/test_data_31) is from Shenzhen Institutes of Advanced Technology, the Chinese Academy of Science.

A improved version is [TRPA](https://github.com/Houruizhi/TRPA), in which the detailed visual results are shown. Please cite the following papers if you use our codes.

```latex
@ARTICLE{TRPA,
  author={Hou, Ruizhi and Li, Fang and Zhang, Guixu},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Truncated Residual Based Plug-and-Play ADMM Algorithm for MRI Reconstruction}, 
  year={2022},
  volume={8},
  number={},
  pages={96-108},
  doi={10.1109/TCI.2022.3145187}}

@article{IDPCNN,
title = {IDPCNN: Iterative denoising and projecting CNN for MRI reconstruction},
journal = {Journal of Computational and Applied Mathematics},
volume = {406},
pages = {113973},
year = {2022},
issn = {0377-0427},
doi = {https://doi.org/10.1016/j.cam.2021.113973},
url = {https://www.sciencedirect.com/science/article/pii/S0377042721005719},
author = {Ruizhi Hou and Fang Li},
keywords = {Magnetic resonance imaging, MRI reconstruction, Image denoising, CNN}
}
```
