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
