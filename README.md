# examples of training models in pytorch

Some implementations of Deep Learning algorithms in PyTorch.

## Ranking

* RankNet
Feed forward NN, minimize document pairwise cross entropy loss function

* LambdaRank
Feed forward NN. Gradient is proportional to NDCG change of swapping two pairs of document

to choose the optimal learning rate, use smaller dataset:
```
python ranking/LambdaRank.py --lr 0.001 --ndcg_gain_in_train exp2 --small_dataset
```
otherwise, use normal dataset:
```
python ranking/LambdaRank.py --lr 0.001 --ndcg_gain_in_train exp2
```
to switch identity gain in NDCG, use `--ndcg_gain_in_train identity`

## Dependencies:
pytorch-1.0
