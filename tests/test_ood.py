from tests.test_FRR_vs_TRR import get_FRR_TRR, METHOD_NAMES, PERTURBATION_TYPE, DATASET_TYPE, NORMALIZATION_KIND
from utils.create_in_out_distr_merged_dataset import create_dataset_mixed

import sklearn.metrics as skm
import numpy as np

def get_FRR_vs_TRR_ood(method_name, dataset_name, t, e, p, iterations, pt_type, in_local=False, dt_type=DATASET_TYPE.TEST, normalization_kind=NORMALIZATION_KIND.ALREADY, logits=False, GAMMAS=50000):
    dt_n = dataset_name + "_3_p_" + str(p) + "_it_"

    d_aurocs_frr = {'AUROC' : [], 'FRR': []}

    print('\nCombine the dataset', iterations, 'times:')
    for i in range(iterations):
        T_0, T_1 = get_FRR_TRR(method_name=method_name,
                    dataset_name=dt_n  + str(i),
                    t=t,
                    e=e,
                    a=2,
                    in_local=in_local,
                    dt_type=dt_type,
                    pt_type=pt_type,
                    normalization_kind=normalization_kind,
                    logits=logits,
                    GAMMAS=GAMMAS)

        auc = round(skm.auc(T_0, T_1) * 100, 1)
        frr = round(np.interp(0.95, np.sort(T_1), np.sort(T_0))*100, 1)
        d_aurocs_frr['AUROC'].append(auc)
        d_aurocs_frr['FRR'].append(frr)

    mean_auc, mean_frr = np.mean(d_aurocs_frr['AUROC']), np.mean(d_aurocs_frr['FRR'])
    st_d_auc, st_d_frr = np.std(d_aurocs_frr['AUROC']), np.std(d_aurocs_frr['FRR'])
    return mean_auc, st_d_auc, mean_frr,  st_d_frr

