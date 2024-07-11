import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def compute_density(features, uncertainties=None, max_increase=0.1):
    normalized_uncertainty = sigmoid(uncertainties - uncertainties.mean())
    densities = []
    for i in range(features.shape[0]):
        weight = 1 + normalized_uncertainty[i] * max_increase
        distance = torch.norm(features[i] - features, dim=1) * weight
        density = 1.0 / (torch.sum(distance, dim=0).item() + 1e-7)
        densities.append(density)

    return densities


def Detective_active(tgt_unlabeled_loader_full, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality, model, cfg, logger,
               t_step):
    lambda_1 = cfg.UNCERTAINTY.LAMBDA_1
    lambda_2 = cfg.UNCERTAINTY.LAMBDA_2

    sample_num = math.ceil(totality * active_ratio)
    remove_num = t_step * sample_num
    # remove_num = math.ceil(0.001 * totality * t_step ** 4)

    model.eval()
    first_stat = list()
    with torch.no_grad():
        for _, data in enumerate(tgt_unlabeled_loader_full):
            tgt_img, tgt_lbl = data['img'], data['label']
            tgt_path, tgt_index = data['path'], data['index']
            tgt_img, tgt_lbl = tgt_img.cuda(), tgt_lbl.cuda()
            tgt_out , tgt_features = model(tgt_img, return_feat=True)

            alpha = torch.exp(tgt_out)
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
            expected_p = alpha / total_alpha
            eps = 1e-7

            point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
            data_uncertainty = torch.sum(
                (alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            distributional_uncertainty = point_entropy - data_uncertainty

            final_uncertainty = lambda_1 * distributional_uncertainty + lambda_2 * data_uncertainty
            for i in range(len(distributional_uncertainty)):
                first_stat.append([tgt_path[i], tgt_lbl[i].item(), tgt_index[i].item(),
                                   final_uncertainty[i].item(),
                                   tgt_features[i].cpu().numpy()
                                   ])

    first_stat = sorted(first_stat, key=lambda x: x[3], reverse=True)  # reverse=True: descending order
    first_stat = first_stat[:sample_num + remove_num]

    features_array = np.stack([item[4] for item in first_stat])
    features = torch.tensor(features_array)
    uncertainties = [stat[3] for stat in first_stat]
    uncertainties = torch.tensor(uncertainties)

    densities = compute_density(features, uncertainties)

    first_stat = [item[:4] for item in first_stat]
    for i in range(len(first_stat)):
        first_stat[i].append(densities[i])

    first_stat = sorted(first_stat, key=lambda x: x[4], reverse=False)  # reverse=False: ascending order

    first_stat = first_stat[:sample_num]
    first_stat = np.array(first_stat)

    selected_active_samples = first_stat[:, 0:2, ...]
    selected_candidate_ds_index = first_stat[:, 2, ...]
    selected_candidate_ds_index = np.array(selected_candidate_ds_index, dtype=np.int32)

    tgt_selected_ds.add_item(selected_active_samples)
    tgt_unlabeled_ds.remove_item(selected_candidate_ds_index)

    return selected_active_samples
