"""
Uplift Ranker:
Improve User Retention with Causal Learning
http://proceedings.mlr.press/v104/du19a/du19a.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpliftRanker(nn.Module):
    def __init__(self, net_structures):
        """
        :param list net_structures: width of each FC layer
        """
        super(UpliftRanker, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            layer = nn.Linear(net_structures[i], net_structures[i+1])
            setattr(self, 'fc' + str(i + 1), layer)

        last_layer = nn.Linear(net_structures[-1], 1)
        setattr(self, 'fc' + str(len(net_structures)), last_layer)
        self.activation = torch.tanh

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1))

    def dump_param(self):
        for i in range(1, self.fc_layers + 1):
            print("fc{} layers".format(i))
            fc = getattr(self, 'fc' + str(i))

            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(fc.bias).item(), torch.norm(fc.bias.grad).item()
            try:
                weight_ratio = weight_grad_norm / weight_norm if weight_norm else float('inf') if weight_grad_norm else 0.0
                bias_ratio = bias_grad_norm / bias_norm if bias_norm else float('inf') if bias_grad_norm else 0.0
            except Exception:
                import ipdb; ipdb.set_trace()

            print(
                '\tweight norm {:.4e}'.format(weight_norm), ', grad norm {:.4e}'.format(weight_grad_norm),
                ', ratio {:.4e}'.format(weight_ratio),
                # 'weight type {}, weight grad type {}'.format(fc.weight.type(), fc.weight.grad.type())
            )
            print(
                '\tbias norm {:.4e}'.format(bias_norm), ', grad norm {:.4e}'.format(bias_grad_norm),
                ', ratio {:.4e}'.format(bias_ratio),
                # 'bias type {}, bias grad type {}'.format(fc.bias.type(), fc.bias.grad.type())
            )


class UpliftRankerTrainer:

    def __init__(
            self, ranker, df_train, df_valid, features,
            treatment='is_treatment', effect='effect', cost='cost'
    ):
        """Initialize Uplift Ranker Trainer with ranker structure, and training, validation data.

        :param torch.nn.Module ranker:
        :param pandas.DataFrame df_train:
        :param pandas.DataFrame df_valid:
        :param list features: feature name list.
        :param str treatment: column indicate if user receives treatment.
        :param str effect: column contains the effect of user.
        :param str cost: column contains the cost of user.
        """
        self.ranker = ranker
        self.df_train = df_train
        self.df_valid = df_valid
        self.features = features
        self.data_fields = {
            'treatment': treatment,
            'effect': effect,
            'cost': cost,
        }
        self.train_feature = torch.tensor(df_train[features].astype(np.float32).values)
        self.valid_feature = torch.tensor(df_valid[features].astype(np.float32).values)

        self.is_treatment = torch.tensor(df_train[treatment].astype(np.uint8).values).unsqueeze(1)
        self.treatment_cnt = df_train[treatment].sum()
        self.control_cnt = df_train.shape[0] - df_train[treatment].sum()

        self.is_treatment_valid = torch.tensor(df_valid[treatment].astype(np.uint8).values).unsqueeze(1)
        self.treatment_cnt_valid = df_valid[treatment].sum()
        self.control_cnt_valid = df_valid.shape[0] - df_valid[treatment].sum()

        self.effect = torch.tensor(df_train[effect].astype(np.float32).values)
        self.effect_valid = torch.tensor(df_valid[effect].astype(np.float32).values)
        self.cost = torch.tensor(df_train[cost].astype(np.float32).values)
        self.cost_valid = torch.tensor(df_valid[cost].astype(np.float32).values)

    def calculate_loss(self, validate=False):
        """Calculate the loss.

        :param bool validate: if to use validation data.
        :return:
        """
        if not validate:
            feature = self.train_feature
            is_treatment = self.is_treatment
            effect = self.effect
            cost = self.cost
        else:
            feature = self.valid_feature
            is_treatment = self.is_treatment_valid
            effect = self.effect_valid
            cost = self.cost_valid

        score = self.ranker(feature)
        exp_score = torch.exp(score)
        exp_score_mask_t = torch.where(is_treatment, exp_score, torch.zeros_like(exp_score))
        exp_score_mask_c = torch.where(is_treatment, torch.zeros_like(exp_score), exp_score)
        prob_select_t = torch.div(exp_score_mask_t, torch.sum(exp_score_mask_t))
        prob_select_c = torch.div(exp_score_mask_c, torch.sum(exp_score_mask_c))

        treatment_effect_weight = torch.sub(prob_select_t, prob_select_c).squeeze(1)
        weighted_effect = torch.dot(effect, treatment_effect_weight)
        weighted_cost = torch.dot(cost, treatment_effect_weight)
        loss = torch.div(weighted_cost, weighted_effect)
        return loss
