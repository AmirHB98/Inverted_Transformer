# iTransformer


🌟 Considering the characteristics of multivariate time series, iTransformer breaks the conventional structure without modifying any Transformer modules. **Inverted Transformer is all you need in MTSF**.



## Overall Architecture

iTransformer regards **independent time series as variate tokens** to **capture multivariate correlations by attention** and **utilize layernorm and feed-forward networks to learn series representations**.

<p align="center">
<img src="./figures/architecture.png" alt="" align=center />
</p>

The pseudo-code of iTransformer is as simple as the following:

<p align="center">
<img src="./figures/algorithm.png" alt="" align=center />
</p>

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Compare the performance of Transformer and iTransformer
bash ./scripts/boost_performance/Weather/iTransformer.sh

# Train the model with partial variates, and generalize on the unseen variates
bash ./scripts/variate_generalization/ECL/iTransformer.sh

# Test the performance on the enlarged lookback window
bash ./scripts/increasing_lookback/Traffic/iTransformer.sh

# Utilize FlashAttention for acceleration
bash ./scripts/efficient_attentions/iFlashTransformer.sh
```

## Main Result of Multivariate Forecasting

We evaluate the iTransformer on challenging multivariate forecasting benchmarks (**generally hundreds of variates**). **Comprehensive good performance** (MSE/MAE $\downarrow$) is achieved.



### Online Transaction Load Prediction of Alipay Trading Platform (Avg Results) 

<p align="center">
<img src="./figures/main_results_alipay.png" alt="" align=center />
</p>

## General Performance Boosting on Transformers

By introducing the proposed framework, Transformer and its variants achieve **significant performance improvement**, demonstrating the **generality of the iTransformer approach** and **benefiting from efficient attention mechanisms**.

<p align="center">
<img src="./figures/boosting.png" alt="" align=center />
</p>

## Zero-Shot Generalization on Variates

**Technically, iTransformer is able to forecast with arbitrary numbers of variables**. We train iTransformers on partial variates and forecast unseen variates with good generalizability.

<p align="center">
<img src="./figures/generability.png" alt="" align=center />
</p>

## Model Analysis

Benefiting from inverted Transformer modules: 

- (Left) Inverted Transformers learn **better time series representations** (more similar [CKA](https://github.com/jayroxis/CKA-similarity)) favored by forecasting.
- (Right) The inverted self-attention module learns **interpretable multivariate correlations**.

<p align="center">
<img src="./figures/analysis.png" alt="" align=center />
</p>

## Citation

If you find this repo helpful, please cite our paper. 

```
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FlashAttention (https://github.com/shreyansh26/FlashAttention-PyTorch)
- Autoformer (https://github.com/thuml/Autoformer)
- Stationary (https://github.com/thuml/Nonstationary_Transformers)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- lucidrains (https://github.com/lucidrains/iTransformer)

# Reviewing iTransformer

1. Embedding layer receives raw TS data at input and transforms it to tokens, Why MLP is used for this layer and how is it trained? What are the series mapped to?
