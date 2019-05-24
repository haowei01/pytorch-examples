# examples of training models in pytorch

Some implementations of Deep Learning algorithms in PyTorch.

## Ranking - Learn to Rank

### RankNet
Feed forward NN, minimize document pairwise cross entropy loss function

to train the model
```
python ranking/RankNet.py --lr 0.001 --debug --standardize
```
`--debug` print the parameter norm and parameter grad norm. This enable to evaluate whether there is gradient vanishing and gradient exploding problem
`--standardize` makes sure input are scaled to have 0 as mean and 1.0 as standard deviation

NN structure: 136 -> 64 -> 16 -> 1, ReLU6 as activation function

| optimizer| lr | epoch |loss (train)|loss (eval)| ndcg@10 | ndcg@30 | sec/epoch | Factorization | pairs/sec |
| :----:| ------ |:-----:|------------|-----------|---------| -----| ----------| --------------- | ----------- |
| adam  | 0.001  |  25 | 0.63002 | 0.635508 | 0.41785 | 0.49337 | 312 | loss func | 203739 |
| adam  | 0.001  |  50 | 0.62595 | 0.633082 | 0.42392 | 0.49771 | 312 | loss func | 203739 |
| adam  | 0.001  | 100 | 0.62282 | 0.632495 | 0.42438 | 0.49817 | 312 | loss func | 203739 |
| adam  | 0.01   | 25  | 0.62668 | 0.631554 | 0.42658 | 0.50032 | 312 | loss func | 203739 |
| adam  | 0.01   | 50  | 0.62118 | 0.629217 | 0.43317 | 0.50533 | 312 | loss func | 203739 |
| adam  | 0.01 | 25 | 0.62349 | 0.633035 | 0.42979 | 0.50108 | 202 | gradient | 314687 |
| adam  | 0.01 | 50 | 0.61781 | 0.630417 | 0.43397 | 0.50540 | 202 | gradient | 314687 |

### LambdaRank
Feed forward NN. Gradient is proportional to NDCG change of swapping two pairs of document

to choose the optimal learning rate, use smaller dataset:
```
python ranking/LambdaRank.py --lr 0.01 --ndcg_gain_in_train exp2 --small_dataset --debug --standardize
```
otherwise, use normal dataset:
```
python ranking/LambdaRank.py --lr 0.01 --ndcg_gain_in_train exp2 --standardize
```
to switch identity gain in NDCG in training, use `--ndcg_gain_in_train identity`

Total pairs per epoch are 63566774 currently each pairs are calculated twice.
The following ndcg number are at eval phase and are using exp2 gain

| optimizer| lr | epoch |loss (eval)| ndcg@10 | ndcg@30 | sec/epoch | Gain func | pairs/sec |
| :----:| ------ |:-----:|-----------|---------| -----| ----------| --------------- | ----------- |
| adam  | 0.001  |  25 | 0.638664 | 0.42470 | 0.49858 | 204 | identity | 311602 |
| adam  | 0.001  |  50 | 0.637417 | 0.42910 | 0.50267 | 204 | identity | 311602 |
| adam  | 0.01   | 25  | 0.635290 | 0.43667 | 0.50687 | 204 | identity | 311602 |
| adam  | 0.01   | 50  | 0.639860 | 0.43874 | 0.50896 | 204 | identity | 311602 |
| adam  | 0.01   | 5   | 0.645545 | 0.43627 | 0.50459 | 208 | exp2 | 304876 |
| adam  | 0.01   | 25  | 0.646903 | 0.44155 | 0.51165 | 208 | exp2 | 304876 |
| adam  | 0.01   | 35  | 0.644680 | 0.44454 | 0.51364 | 208 | exp2 | 304876 |

As the result compared with RankNet, LambdaRank's NDCG is generally better than RankNet, but cross entropy loss is higher
This is mainly due to LambdaRank maximizing the NDCG, while RankNet minimizing the pairwise cross entropy loss.


## Dependencies:
* pytorch-1.0
* pandas
* numpy
* sklearn
