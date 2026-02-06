# Learning Turbulent Flows with Generative Models

This code is part of the article: "Learning Turbulent Flows with Generative Models for Super Resolution and Sparse Flow Reconstruction", accepted in Nature Communications.

Neural operators are promising surrogates for dynamical systems but when trained with standard L² losses they tend to oversmooth fine-scale turbulent structures. Here, we show that combining operator learning with generative modeling overcomes this limitation. We consider three practical turbulent-flow challenges where conventional neural operators fail: 1) spatio-temporal super-resolution, 2) forecasting, and 3) sparse flow reconstruction. Check out our [Project page](https://vivekoommen.github.io/Gen4Turb) for more details. 

## Requirements

- Python 3.9.16
- Install dependencies: `pip install -r requirements.txt`

## Repository Structure

The three main tasks are organized as three main directories in the root. Each directory has a data folder. Before training, please copy the correct data files for each task into the appropriate data folder from [Zenodo](https://zenodo.org/records/17088765).

### Task 1: Super-resolution

The different models compared in this task are located in:
- **NO**: `1_superresolution/no`
- **adv-NO**: `1_superresolution/no/adv_no`
- **NO+VAE**: `1_superresolution/no/vae`
- **NO+GAN**: `1_superresolution/no/gan`
- **NO+DM**: `1_superresolution/no/dm`

### Task 2: Forecasting

The different models compared in this task are located in:
- **NO**: `2_forecasting/no`
- **adv-NO**: `2_forecasting/no/adv_no`
- **NO+div**: `2_forecasting/no_div`
- **GenCFD**: `2_forecasting/GenCFD` (follow the README provided in GenCFD for installation)

### Task 3: Flow Reconstruction

The different models compared in this task are located in:
- **GAN**: `3_flow_reconstruction/no/adv_training`
- **Diffusion Model**: `3_flow_reconstruction/dm`

## Training

To train a model, navigate to the corresponding model directory and run:
```bash
python -u train_*.py
```

**Special case for GenCFD**: Run the following script:
```bash
2_forecasting/GenCFD/scripts/train_hit3d.sh
```

## Post-processing

The `postprocess.ipynb` notebook in each model's directory loads the trained model (also provided) and saves the prediction.

**Special case for GenCFD**: Run `2_forecasting/GenCFD/colab/postprocess.ipynb`

## Model Comparison

To compare the different models for each task, run the `comparison.ipynb` notebook in each of the three task directories.


## Citing This Work

If you find this work useful, please cite:

```bibtex
@article{oommen2025learning,
      author={Oommen, Vivek and Khodakarami, Siavash and Bora, Aniruddha and Wang, Zhicheng and Karniadakis, George Em},
      title={Learning Turbulent Flows with Generative Models: Super-resolution, Forecasting, and Sparse Flow Reconstruction},
      journal={arXiv preprint arXiv:2509.08752},
      year={2025}
      }
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Acknowledgments

We acknowledge the following repositories:
1. **Real-Esrgan** - [https://github.com/lizhuoq/Real-Esrgan](https://github.com/lizhuoq/Real-Esrgan)
2. **MedicalNet** - [https://github.com/Tencent/MedicalNet](https://github.com/Tencent/MedicalNet)
3. **denoising-diffusion-pytorch** - [https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main)
