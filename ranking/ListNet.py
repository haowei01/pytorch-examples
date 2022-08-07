"""
ListNet:
Learning to Rank: From Pairwise Approach to Listwise Approach
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf

ListNet Rank
1. For each query, qi, generate the ListWise Loss function L(yi, zi), yi is the list of score of
    labels, zi is the list of score generated from the ranking function.
2. Permutation Probability, e.g.
    P(<1, 2, 3>) = phi(s1) / (phi(s1) + phi(s2) + phi(s3)) * phi(s2) / (phi(s2) + phi(s3)) * phi(s3)/phi(s3)
3. Document dj's probability on becoming the top 1 position:
    P(sj) = phi(sj) / Sum(phi(sk))
4. Loss function given the probability of labeled score yi vs ranking score si, e.g. using Cross Entropy:
    L(yi, zi) = - Sum [P(yj) * log (P(zj))]
"""