"""
Uplift Ranker:
Improve User Retention with Causal Learning
http://proceedings.mlr.press/v104/du19a/du19a.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt


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
            treatment='is_treatment', effect='effect', cost='cost',
            learning_rate=0.001, weight_decay=1e-5
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

        self.init_step(learning_rate=learning_rate, weight_decay=weight_decay)

    def init_step(self, save_model_dst='model.pt', learning_rate=0.001, weight_decay=0.0001):
        """
        :param str save_model_dst: save model to dst.
        :param float learning_rate:
        :param float weight_decay:
        """
        self.uniform_init_weight()
        self.losses = []
        self.eval_losses = []
        self.best_eval_loss = None
        self.save_model_dst = save_model_dst
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.ranker.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)

    def uniform_init_weight(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.ranker.apply(init_weights)

    def init_and_train(
            self, train_rounds=1, save_model_dst='model.pt', learning_rate=0.001, weight_decay=0.0001,
            step_size=10, gamma=0.75, steps=1000, validate_steps=1, debug=False,
    ):
        """initialize the training with multiple rounds, and find the best model.

        :param int train_rounds: initialize the train the model multiple rounds.
        :param str save_model_dst: saved model destination.
        :param float learning_rate:
        :param float weight_decay: used for L2 regularization.
        :param int step_size: learning rate scheduler, adjust the learning rate per step_size.
        :param float gamma: learning rate weight decay.
        :param int steps: max step to train one model.
        :param int validate_steps: run validation step every N steps.
        """
        self.best_eval_loss = None
        self.save_model_dst = save_model_dst
        for train_round in range(train_rounds):
            self.uniform_init_weight()
            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.ranker.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma)
            self.losses = []
            self.eval_losses = []
            self.train(steps, validate_steps, debug)
            plt.scatter(np.arange(0, len(self.eval_losses)), self.eval_losses)
            plt.scatter(np.arange(0, len(self.losses)), self.losses)
            plt.show()

    def train(self, steps, validate_steps=1, debug=False):
        """
        :param int steps:
        :param int validate_steps: validate every N steps
        """
        for step in range(steps):
            self.ranker.train()
            loss = self.train_step()
            if debug and (step == 0 or (step + 1) % min(5, validate_steps) == 0):
                print(step, "train loss: ", loss)
            self.losses.append(loss.item())

            if (step + 1) % validate_steps == 0:
                self.ranker.eval()
                with torch.no_grad():
                    eval_loss = self.calculate_loss(validate=True)

                if debug:
                    print("eval loss: ", eval_loss)
                self.eval_losses.append(eval_loss.item())
                if self.best_eval_loss is None or eval_loss.item() < self.best_eval_loss:
                    # save the best model based on the eval loss
                    torch.save({
                        'model_state_dict': self.ranker.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler': self.scheduler.state_dict(),
                    }, 'ckptdir/{}'.format(self.save_model_dst))
                    self.best_eval_loss = eval_loss.item()

    def predict(self, df_features):
        """
        :param pandas.DataFrame df_features:
        :rytpe: numpy.array
        """
        predict_features = torch.tensor(df_features[self.features].astype(np.float32).values)
        with torch.no_grad():
            score = self.ranker(predict_features)
        return score.squeeze(1).detach().numpy()

    def train_step(self):
        """
        :rtype: torch.Tensor
        """
        self.ranker.zero_grad()
        loss = self.calculate_loss()
        loss.backward()
        self.optimizer.step()
        return loss

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


class UpliftRankerEnsemble:

    def __init__(self, features, *model_struct_paths):
        self.features = features
        self.models = []
        for net_struct, model_path in model_struct_paths:
            ranker = UpliftRanker(net_struct)
            ranker.load_state_dict(torch.load(model_path)['model_state_dict'])
            ranker.eval()
            self.models.append(ranker)

    def predict(self, df_features):
        """
        :param pandas.DataFrame df_features:
        :rytpe: numpy.array
        """
        predict_features = torch.tensor(df_features[self.features].astype(np.float32).values)
        scores = np.zeros(df_features.shape[0])
        for model in self.models:
            scores += model(predict_features).squeeze(1).detach().numpy()
        scores /= len(self.models)
        return scores
