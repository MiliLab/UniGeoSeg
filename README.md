
<div align="center">

<h1>UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes</h1>

Shuo Ni<sup>1,3</sup>, 
Di Wang<sup>2,3</sup>, 
He Chen<sup>1</sup>, 
Haonan Guo<sup>2,3 †</sup>, 
Ning Zhang<sup>1.4 †</sup>, 
Jing Zhang<sup>2 †</sup>.

<sup>1</sup> Beijing Institute of Technology,  <sup>2</sup> Wuhan University,  <sup>3</sup> Zhongguancun Academy,  <sup>4</sup> Hong Kong Polytechnic University.

<sup>†</sup> Corresponding author

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-models">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >


## 🔥 Update

- **[2026.03]** 📦 **GeoSeg-1M** is now publicly available on **Baidu Netdisk**. The **Hugging Face** version is still being uploaded.
- **[2026.03]** 🎉 Our paper has been **accepted to CVPR 2026**!
- **[2025.11]** 📄 Paper available on arXiv: **[arXiv:2511.23332](https://arxiv.org/abs/2511.23332)**


## 🌞 Abstract

Instruction-driven segmentation in remote sensing generates masks from guidance, offering great potential for accessible and generalizable applications. However, existing methods suffer from fragmented task formulations and limited instruction data, hindering effective understanding and generalization. To address these issues, we introduce GeoSeg-1M, the first million-scale dataset for remote sensing instruction-driven segmentation, constructed via an automatic mask filtering and instruction generation pipeline that synthesizes referring, interactive, and reasoning segmentation instructions from multiple public datasets. GeoSeg-1M contains 590K images, 117 categories, and 1.1M image–mask–instruction triplets. Building upon this foundation, we further curate GeoSeg-Bench, a challenging benchmark designed to evaluate contextual understanding and reasoning capabilities across diverse instruction-driven tasks and complex geospatial scenes. Furthermore, we present UniGeoSeg, a unified framework that serves as a strong baseline, incorporating task-aware text enhancement, latent knowledge memory, and a progressive training strategy to facilitate multi-task learning.  Extensive experiments demonstrate the state-of-the-art performance of UniGeoSeg across GeoSeg-Bench and diverse public benchmarks, while exhibiting strong zero-shot generalization. 

<figure>
<div align="center">
<img src=Figs/intro.png width="50%">
</div>

<div align='center'>
 
**Figure 1. Examples from GeoSeg-1M.**

</div>
<br>

<div align="center">
<img src=Figs/framework.png width="100%">
</div>

<div align='center'>

**Figure 2. The diagram of UniGeoSeg.**

</div>

## 📖 Datasets

The GeoSeg-Bench can be downloaded at **[Hugging Face](https://huggingface.co/datasets/nishuo1999/GeoSeg-Bench)**.

The **GeoSeg-1M** dataset is now available via **Baidu Pan**:

- Dataset: `GeoSeg-1M`
- Link: https://pan.baidu.com/s/1GldxhdvJqlsI-qyq3Uvqkw
- Extraction code: `thxx`

The **Hugging Face** version of **GeoSeg-1M** is still being uploaded and will be released soon.

## 🚀 Models

The checkpoint can be downloaded at **[Hugging Face](https://huggingface.co/nishuo1999/UniGeoSeg)**.

## 🔨 Usage

### Training

Wait for update.

### Inference

We provide an inference script:

```
python scripts/eval.sh
```

## 🍭 Results


<div align="center">
<img src=Figs/result.png width="100%">
</div>


## ⭐ Citation

If you find UniGeoSeg helpful, please give a ⭐ and cite it as follows:

```
@misc{ni2025unigeosegunifiedopenworldsegmentation,
      title={UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes}, 
      author={Shuo Ni and Di Wang and He Chen and Haonan Guo and Ning Zhang and Jing Zhang},
      year={2025},
      eprint={2511.23332},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.23332}, 
}
```

## 🎺 Statement

For any other questions please contact Shuo Ni at [bit.edu.cn](mailto:3120245503@bit.edu.cn) or [126.com](nishuo1999@126.com).


## 💖 Thanks
This project is based on [PSALM](https://github.com/zamling/PSALM), [SegEarth-R1](https://github.com/earth-insights/SegEarth-R1),  Thanks for their wonderful work!<br>

