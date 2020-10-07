# HAES: A New Hybrid Approach for Movie Recommendation with Elastic Serendipity

This is the implementation of our paper:

Xueqi Li, Wenjun Jiang, Weiguang Chen, Jie Wu and Guojun Wang. HAES: A New Hybrid Approach for Movie Recommendation with Elastic Serendipity. In CIKM 2019. <a href="https://dl.acm.org/doi/10.1145/3357384.3357868">paper</a>

## Environment Settings

- pytorch == 1.3.1
- numpy == 1.18.1
- pandas == 1.1.0

## Simple example to run the code

```python
python main --dataset_name ml-ls --epoch 50 --batch_size 128
```

## Datasets
We conduct experiments on two datasets, MovieLens-latest-small (ml-ls) and MovieLens-1m (ml-1m).

## Some Supplements
In the implementation, for keeping the statistical characteristics of datasets, we only filter out users with less than 20 ratings instead of iteratively filtering out users and items with less than 20 ratings. Therefore, the quantitative results may differ from that in paper, but the comparison between our approch and baselines is consistent with that in paper. We briefly show the comparison results on two datasets, MovieLens-latest-small (ml-ls) and MovieLens-1m (ml-1m), as follows.

<caption>Results on genre accuracy. (ml-ls)</caption>

|method|F1@5|F1@10|F1@15|h@5|h@10|h@15|
|----|----|----|----|----|----|----|
ACC|0.5555|0.5846|0.5859|0.3696|0.4224|0.4583
RAND|0.5474|0.5888|0.5858|0.3573|0.3984|0.4434
HAES|0.5889|0.6164|0.6251|0.3543|0.3798|0.3929

<caption>Results on content difference. (ml-ls)</caption>

|method|dif@5|dif@10|dif@15|div@5|div@10|div@15|
|----|----|----|----|----|----|----|
ACC|0.3569|0.3571|0.3572|0.0374|0.0372|0.0359
RAND|0.4467|0.4403|0.4402|0.1762|0.1699|0.17
HAES|0.4649|0.4578|0.4532|0.2203|0.1984|0.1864

<caption>Results on genre accuracy. (ml-1m)</caption>

|method|F1@5|F1@10|F1@15|h@5|h@10|h@15|
|----|----|----|----|----|----|----|
ACC|0.4834|0.5097|0.5098|0.334|0.3919|0.4353
RAND|0.4755|0.4993|0.4981|0.3224|0.3866|0.4367
HAES|0.5413|0.5523|0.5483|0.2775|0.3195|0.354

<caption>Results on content difference. (ml-1m)</caption>

|method|dif@5|dif@10|dif@15|div@5|div@10|div@15|
|----|----|----|----|----|----|----|
ACC|0.5615|0.568|0.5731|0.302|0.3076|0.3086
RAND|0.706|0.706|0.7055|0.3436|0.3441|0.3437
HAES|0.7185|0.718|0.7177|0.3405|0.3403|0.3396
