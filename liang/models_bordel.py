# This seems to be a purely overfitting setting
#
# Baseline : test 0.611
# Dropout on h : test 0.6
# Bayesian nets don't work (funcs can be arbitrarily different) : test 0.5
# Jacobian reg || (J_j(Loss)) ||_2 à la Ma: test 0.6
# Consistency training w. dropout (.5, .9) on h: test 0.6
# Noise added to hidden features e.g. IB : test 0.6
#
# Variance minimization on features : test 0.924
# Feature selection : test 1.
#

#
#   I(x_1, ..., x_N; y) = 3
#   spectrum ^ long tail
#
#     [\sum_j I(x_j; y)
#  -> [\sum_j I(x_j; y | x_i), \sum_ji I(x_i, x_j; y)
#  ..
#       I(x_1; ..., x_N; y) = 3
#       I(x_1, x_2; y) = 3
#
#       I(x_1; y) = 1
#       I(x_2; y) = 1
#       I(x_N; y) = 1
#
#
#   I(x_1, y | x_\not 1) = 
#


def safelog(x):
    return torch.log(x+1e-6)


class GradDiv(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
    self.fc2 = nn.Linear(50, 1)
  
  def predict(self, x):
    with torch.no_grad():
        y = self.forward(x)[1]
    return y

  def get_grad(self, h, py, mb_y):
    """Grad wrt last layer.
    """
    return mb_y[:, None] * h * (1 - py) + (1 - mb_y[:, None]) * h * py

  def get_all_grads(self, lossi):
    grads = []
    for i in range(lossi.size(0)):
        gradi = torch.autograd.grad(
            lossi[i], list(self.parameters()),
            create_graph=True)
        grad = []
        for g in gradi:
            grad.append(g.flatten())
        grads.append(torch.cat(grad, 0))
    return torch.stack(grads, 0)

  def forward(self, x, y, train=True):
    h = self.fc1(x)
    py = torch.sigmoid(self.fc2(h)).squeeze()
    outputs = {}
    lossi = F.binary_cross_entropy(py, y, reduction='none')
    loss = lossi.mean()
    acc = py.gt(0.5).float().eq(y).float().mean()
    if train:
        all_example_gradients = self.get_all_grads(lossi)
        all_example_gradients = all_example_gradients / (1e-5 + all_example_gradients.norm(2, 1)[:, None])
        all_gradient_cosine = torch.matmul(all_example_gradients, all_example_gradients.t())
        all_gradient_cosine = all_gradient_cosine * (1. - torch.eye(all_gradient_cosine.size(0)))
        diversity = torch.relu(all_gradient_cosine).mean()
    else:
        diversity = 0.
    outputs['acc'] = acc
    outputs['div'] = diversity
    outputs['loss'] = loss + 10 * diversity
    return outputs

# lossyi, _, entpyi, pyi, _ = learning_loss(mb_x * mask_x, mb_y)
# _, _, _, pyni, _ = learning_loss(mb_x * (1 - mask_x), mb_y)

# min MI(y, rest | i) = E_y, all [log p(y | all) / p(y | i)]
#
# max MI(y, i | rest) = E_y, all [log p(y | all) / p(y | all - i)]
#
# max MI(y, i) = E_p(y, i) [ log p(y | i) / p(y) ]

# mi_y_rest_g_i = py * (safelog(py) - safelog(pyi)) + (1-py) * (safelog(1-py) - safelog(1-pyi))
# mi_y_i_g_rest = py * (safelog(py) - safelog(pynx)) + (1-py) * (safelog(1-py) - safelog(1-pynx))

# pyi = pyi.reshape(n_batch, r)
# mi_y_i_weight = F.softmax((pyi * safelog(pyi) + (1-pyi) * safelog(1-pyi)), 1)
# mi_y_rest_g_i = (mi_y_rest_g_i.reshape(n_batch, r)).mean()
# mi_y_i_g_rest = (mi_y_i_g_rest.reshape(n_batch, r) * mi_y_i_weight.detach()).sum(-1).mean()

# cons_i = (py-pyi)**2.0
# cons_rest = (py-pyni)**2.0
# (lossi + cons_i.mean() - cons_rest.mean()).backward()
# mi = (mi_y_rest_g_i.mean() - mi_y_i_g_rest.mean())
# (lossi + mi).backward()

class JacobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def forward(self, x, y, train=True):
        h = self.fc1(x)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        jacob = torch.autograd.grad(loss, h, create_graph=True)[0]
        jacob_norm = jacob.norm(2, dim=1).mean()
        acc = py.gt(0.5).float().eq(y).float().mean()
        outputs['acc'] = acc
        outputs['jacob'] = jacob_norm
        outputs['loss'] = loss + jacob_norm
        return outputs


class MINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def forward(self, x, y, train=True):
        h = self.fc1(x)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        nce_b = ((h - h.mean(0)[None, :]) ** 2.0).sum(-1).mean(0)
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        outputs['acc'] = acc
        outputs['nce'] = nce_b
        outputs['loss'] = loss + nce_b
        return outputs


class NoiseNet(nn.Module):
    def __init__(self, noise_level=0., dropout=0.):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 200), nn.ReLU())
        self.fc2 = nn.Linear(200, 1)
        self.noise_level = noise_level
        self.dropout = dropout

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def feats(self, x):
        with torch.no_grad():
            return self.fc1(x)

    def forward(self, x, y, train=True):
        h = self.fc1(x)
        if train:
            if self.noise_level > 0.:
                h = h + self.noise_level * torch.randn_like(h) * h.norm(2, dim=1)[:, None]
        if self.dropout > 0.:
            h = F.dropout(h, self.dropout, training=train)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        outputs['acc'] = acc
        outputs['loss'] = loss
        return outputs


class SVDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.fc3 = nn.Linear(100, 1)

    def feats(self, x):
        with torch.no_grad():
            return self.fc2(self.fc1(x))

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc3(self.fc2(self.fc1(x))))
        return o

    def forward(self, x, y, train=True):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        py = torch.sigmoid(self.fc3(h2)).squeeze()
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        nce_t = 0
        for h in (h1, h2):
            u, s, v = torch.svd(h)
            h1 = torch.matmul(h, v[:, 1:])
            m = y[:, None].eq(y[None, :])
            v = torch.mm(h1, h1.T)
            pos_v = v * m.float()
            neg_v = v - 10 * m.float()
            neg_m = neg_v.max(dim=1)[0].unsqueeze(1).detach()
            neg_v = neg_v - neg_m
            pos_v = pos_v - neg_m
            neg_sum_exp = torch.sum(torch.exp(neg_v), 1)[:, None]
            nll = -pos_v + torch.log(neg_sum_exp + torch.exp(pos_v))
            nce_t = nce_t + (nll * m).sum() / m.sum()
            # nce_t = ((1 - v * m)).sum() / m.sum()
            # nce_t = nce_t + -(F.log_softmax(v, 1) * m).mean()
        outputs['acc'] = acc
        outputs['nce'] = nce_t
        outputs['loss'] = loss + nce_t
        return outputs


class ConsistentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def forward_(self, x, noise=False):
        if noise:
            x = x + torch.randn_like(x) * 2
        h = self.fc1(x)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        return py

    def forward(self, x, y, train=True):
        py = self.forward_(x)
        py_noise = self.forward_(x, noise=True)
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        cons = ((py - py_noise) ** 2.0).mean()
        outputs['acc'] = acc
        outputs['cons'] = cons
        outputs['loss'] = loss + cons
        return outputs


class FeatureSelectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)
        self.gs = nn.Parameter(torch.ones(50))

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def forward(self, x, y, train=True):
        g = F.softmax(self.gs)
        h = self.fc1(x)
        h = g[None, :] * h
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        ent = -(g * torch.log(g + 1e-6)).sum()
        outputs['acc'] = acc
        outputs['ent'] = ent
        outputs['loss'] = loss + ent
        return outputs


class Ma(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)
        self.step = 0

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        self.train()
        return o

    def forward(self, x, y, g=None, train=True):
        h = self.fc1(x)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        if g is not None:
            grad = torch.autograd.grad(loss, h, create_graph=True)[0]
            g_mask = (g.eq(1).float() + g.eq(2).float())
            grad_loss = (grad.norm(2, dim=1) * g_mask[:, None]).sum() / (g_mask.sum() + 1e-6)
            outputs['gloss'] = 100 * grad_loss
        else:
            grad_loss = 0.
        outputs['acc'] = acc
        outputs['loss'] = loss + grad_loss
        self.step += 1
        return outputs


class VarianceMin(nn.Module):
    def __init__(self, per_class=False, softmax=False):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)
        self.step = 0
        self.softmax = softmax
        self.per_class = per_class

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        self.train()
        return o

    def forward(self, x, y, train=True):
        h = self.fc1(x)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        h_clip = h
        # per class
        class0 = y.eq(0).float().unsqueeze(1)
        class1 = 1.-class0
        if self.softmax:
            hs = F.softmax(torch.abs(h), dim=0)
            var_x = (-(hs * torch.log(hs + 1e-6)).mean(0)).sum(0)
        if self.per_class:
            var0 = (((h * class0) - ((h * class0).sum(0) / (class0.sum(0) + 1e-6))[None, :]) ** 2.0).sum(-1).mean()
            var1 = (((h * class1) - ((h * class1).sum(0) / (class1.sum(0) + 1e-6))[None, :]) ** 2.0).sum(-1).mean()
            var_x = var0 + var1
        else:
            var_x = ((h - h.mean(0)[None, :]) ** 2.0).sum(-1).mean(0)
        outputs['acc'] = acc
        outputs['var_x'] = var_x
        outputs['loss'] = loss + var_x
        self.step += 1
        return outputs


class Confidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(102, 50), nn.ReLU())
        self.fc2 = nn.Linear(50, 1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        self.train()
        return o

    def forward(self, x, y, train=True):
        h = self.fc1(x)
        py = torch.sigmoid(self.fc2(h)).squeeze()
        outputs = {}
        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()
        ent = -(py * torch.log(py+1e-6) + (1-py) * torch.log(1-py+1e-6))
        var = ((ent[:, None] - ent[None, :]) ** 2.0).mean()
        outputs['acc'] = acc
        outputs['conf'] = 10 * var
        outputs['loss'] = loss + var
        return outputs

class Avgs:
    def __init__(self):
        self.avgs = {}
        self.n = 0

    def update(self, outputs):
        for key, val in outputs.items():
            val = self.avgs.get(key, 0) + val.item()
            self.avgs[key] = val
        self.n += 1

    def __str__(self):
        return ", ".join(
            ["{}: {:.3f}".format(k, (v / self.n)) for k, v in self.avgs.items()])

    def plot(self, name):
        plt.plot(len(self.avgs[name]), self.avgs[name])


def fit(X, y, x_test, y_test, nepochs=10, device='cpu', method='base', g=None,
        amount=0., prune_every=0):

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    if g is not None:
        g = torch.from_numpy(g).float()

    if method == 'feature_selection_x':
        f = FeatureSelectionNet().to(device)
    elif method == 'noise_net':
        f = NoiseNet(noise_level=0.2).to(device)
    elif method == 'dropout_net':
        f = NoiseNet(dropout=0.5).to(device)
    elif method == 'mi_net':
        f = MINet().to(device)
    elif method == 'jacob_net':
        f = JacobNet().to(device)
    elif method == 'consistent_net':
        f = ConsistentNet().to(device)
    elif method == 'svd_net':
        f = SVDNet().to(device)
    elif method in ['base', 'sam', 'sam_min', 'sam_maj', 'reweight']:
        f = NoiseNet(noise_level=0.).to(device)
    elif method == 'fvar_min':
        f = VarianceMin().to(device)
    elif method == 'fvar_min_sm':
        f = VarianceMin(softmax=True).to(device)
    elif method == 'fvar_min_per_class':
        f = VarianceMin(per_class=True).to(device)
    elif method == 'ma':
        f = Ma().to(device)
    elif method == 'confidence':
        f = Confidence().to(device)

    init_state_dict = dict()
    for child in f.modules():
        if isinstance(child, nn.Linear):
            init_state_dict[child] = (child.weight.data.cpu().numpy(),
                                      child.bias.data.cpu().numpy())

    if method.startswith("sam"):
        fopt = SAM(list(f.parameters()), torch.optim.Adam, rho=0.5, lr=0.0005, weight_decay=1e-5)
    else:
        fopt = torch.optim.Adam(list(f.parameters()), lr=0.0005, weight_decay=1e-5)

    nx = len(X)
    n_batch = nx
    global_step = 0
    globavgs = Avgs()
    # cross-entropy pass
    for n in range(nepochs):
        avgs = Avgs()
        step = 0

        for k in range(1000):
            # mb_x = (n_batch, )
            # mb_f = (n_batch, )
            mb_i = np.random.randint(nx, size=n_batch)
            mb_x = X[mb_i]
            mb_y = y[mb_i].flatten()
            mb_x = mb_x.to(device)
            mb_y = mb_y.to(device)

            fopt.zero_grad()

            if method.startswith("sam"):
                if method == "sam":
                    max_x, max_y = mb_x, mb_y
                elif method == "sam_min":
                    g_x = g_spu[mb_i]
                    max_i = np.where((g_x != 0) & (g_x != 3))[0]
                    max_x = mb_x[max_i]
                    max_y = mb_y[max_i]
                elif method == "sam_maj":
                    g_x = g_spu[mb_i]
                    max_i = np.where((g_x != 1) & (g_x != 2))[0]
                    max_x = mb_x[max_i]
                    max_y = mb_y[max_i]

                loss1 = f(max_x, max_y)['loss']
                loss1.backward()
                fopt.first_step(zero_grad=True)
                fopt.zero_grad()
                outputs = f(mb_x, mb_y)
                avgs.update(outputs)
                outputs['loss'].backward()
                fopt.second_step(zero_grad=True)
            elif method == "reweight":
                g_x = g_spu[mb_i]
                # minorities
                max_i = np.where((g_x != 0) & (g_x != 3))[0]
                max_x = mb_x[max_i]
                max_y = mb_y[max_i]
                n_min = len(max_i)
                min_outputs = f(max_x, max_y)
                max_i = np.where((g_x != 1) & (g_x != 2))[0]
                max_x = mb_x[max_i]
                max_y = mb_y[max_i]
                n_maj = len(max_i)
                maj_outputs = f(max_x, max_y)
                maj_l = maj_outputs['loss']
                min_l = min_outputs['loss']
                avgs.update(maj_outputs)
                (float(n_maj) / float(n_min) * min_l + maj_l).backward()
                fopt.step()
            else:
                outputs = f(mb_x, mb_y)
                avgs.update(outputs)
                outputs['loss'].backward()
                fopt.step()

            if prune_every and global_step % prune_every == 0:
                prune(f)
            global_step += 1

        print('iter: {}/{}, {}'.format(n, nepochs, str(avgs)))
        xt = torch.from_numpy(x_test).float()
        yt = torch.from_numpy(y_test).float()
        test_outputs = f(xt, yt, train=False)
        print('  test: {:.3f}'.format(test_outputs['acc']))
    return f.to("cpu")


def model_chooser(args):
    method = args.method

    if method == 'feature_selection_x':
        f = FeatureSelectionNet()
    elif method == 'noise_net':
        f = NoiseNet(noise_level=0.2)
    elif method == 'dropout_net':
        f = NoiseNet(dropout=0.5)
    elif method == 'mi_net':
        f = MINet()
    elif method == 'jacob_net':
        f = JacobNet()
    elif method == 'consistent_net':
        f = ConsistentNet()
    elif method == 'svd_net':
        f = SVDNet()
    elif method in ['base', 'sam', 'sam_min', 'sam_maj', 'reweight']:
        f = NoiseNet(noise_level=0.)
    elif method == 'fvar_min':
        f = VarianceMin()
    elif method == 'fvar_min_sm':
        f = VarianceMin(softmax=True)
    elif method == 'fvar_min_per_class':
        f = VarianceMin(per_class=True)
    elif method == 'ma':
        f = Ma()
    elif method == 'confidence':
        f = Confidence()
    return f
