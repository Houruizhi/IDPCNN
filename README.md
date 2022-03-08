# [IDPCNN](10.1016/j.cam.2021.113973): Iterative denoising and projecting CNN for MRI reconstruction



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
