import json
import numpy as np
from sklearn import metrics


def cal_cmc_aps(feats, pids, cams, query, gallery, log_file, cal_aps=True, batch_size=10):
    """
    calculate cmc and aps
    :param feats:  feats of tracklets
    :param pids:  person ids of tracklets
    :param cams:  camera ids of tracklets
    :param query:  query mask
    :param gallery:  gallery mask
    :param log_file:  log file to store results
    :param cal_aps:  whether or not calculate aps
    :param batch_size:  batch size to calculate distances
    :return:
    """

    query_num = len(query)
    gallery_num = len(gallery)

    query_feat = feats[query].contiguous().view(query_num, 1, -1)
    gallery_feat = feats[gallery].contiguous().view(1, gallery_num, -1)
    query_pid, gallery_pid = pids[query], pids[gallery]
    query_cam, gallery_cam = cams[query], cams[gallery]
    if batch_size is None or batch_size > query_num:
        dists = (query_feat - gallery_feat).pow(2).sum(dim=-1).cpu().numpy()
    else:
        dists = []
        for i in range(query_num // batch_size):
            diff = query_feat[i * batch_size: (i + 1) * batch_size] - gallery_feat
            dists.append(diff.pow(2).sum(dim=-1).cpu().numpy())
        if query_num > (i + 1) * batch_size:
            diff = query_feat[(i + 1) * batch_size: query_num] - gallery_feat
            dists.append(diff.pow(2).sum(dim=-1).cpu().numpy())
        dists = np.concatenate(dists)

    matches = query_pid[:, np.newaxis] == gallery_pid[np.newaxis, :]
    mask = np.logical_and(matches, query_cam[:, np.newaxis] == gallery_cam[np.newaxis, :])
    matches[mask] = False

    aps = []
    cmc = np.zeros(gallery_num)
    for i in range(query_num):
        dists[i, mask[i]] = np.inf
        if cal_aps:
            scores = 1 / (1 + dists[i])
            p, r, _ = metrics.precision_recall_curve(matches[i], scores)
            ap = metrics.auc(r, p)
            if np.isnan(ap):
                print('Encountered an AP of nan which usually means a person only appears once.')
                continue
            aps.append(ap * 100)
        k = np.where(matches[i, np.argsort(dists[i])])[0][0]
        cmc[k:] += 1
    cmc = cmc / query_num * 100
    if cal_aps:
        mean_ap = np.array(aps).mean()
        print('rank1|rank5|rank10 : {:.2f}|{:.2f}|{:.2f}, mAP : {:.2f}'.format(
            cmc[0], cmc[4], cmc[9], mean_ap))
        with open(log_file, 'w') as f:
            json.dump({'CMC': list(cmc), 'mAP': mean_ap, 'aps': aps}, f)
    else:
        print('rank1|rank5|rank10 : {:.2f}|{:.2f}|{:.2f}'.format(
            cmc[0], cmc[4], cmc[9]))
        with open(log_file, 'w') as f:
            json.dump({'CMC': list(cmc)}, f)
