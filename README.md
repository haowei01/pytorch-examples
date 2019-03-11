# examples of training models in pytorch

Some implementations of Deep Learning algorithms in PyTorch.

## Ranking

### RankNet
Feed forward NN, minimize document pairwise cross entropy loss function

to train the model
```
python ranking/RankNet.py --lr 0.001 --debug --standardize
```
`--debug` print the parameter norm and parameter grad norm. This enable to evaluate whether there is gradient vanishing and gradient exploding problem
`--standardize` makes sure input are scaled to have 0 as mean and 1.0 as standard deviation

NN structure: 136 -> 64 -> 16 -> 1, ReLU6 as activation function

| optimizer | lr | epoch |loss (train)|loss (eval)| ndcg@10 | ndcg@30 | sec/epoch |
| :----:| ------ |:-----:|------------|-----------|---------| -----| ----------|
| adam  | 0.001  |  25 | 0.63002 | 0.635508 | 0.41785 | 0.49337 | 312 |
| adam  | 0.001  |  50 | 0.62595 | 0.633082 | 0.42392 | 0.49771 | 312 |
| adam  | 0.001  | 100 | 0.62282 | 0.632495 | 0.42438 | 0.49817 | 312 |
| adam  | 0.01   | 25  | 0.62668 | 0.631554 | 0.42658 | 0.50032 | 312 |
| adam  | 0.01   | 50  | 0.62118 | 0.629217 | 0.43317 | 0.50533 | 312 |
### LambdaRank
Feed forward NN. Gradient is proportional to NDCG change of swapping two pairs of document

to choose the optimal learning rate, use smaller dataset:
```
python ranking/LambdaRank.py --lr 0.0001 --ndcg_gain_in_train exp2 --small_dataset --debug --standardize
```
otherwise, use normal dataset:
```
python ranking/LambdaRank.py --lr 0.0001 --ndcg_gain_in_train exp2 --standardize
```
to switch identity gain in NDCG, use `--ndcg_gain_in_train identity`

## Dependencies:
pytorch-1.0
