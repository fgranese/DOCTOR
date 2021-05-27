from tests.test_FRR_vs_TRR import *
from tests.test_ood import get_FRR_vs_TRR_ood
from utils.create_in_out_distr_merged_dataset import create_dataset_mixed
import sys

def test_tbb_pbb(dataset_name, scenario, ood):
    dataset_name = dataset_name.lower()
    scenario = scenario.lower()

    if not ood:
        if scenario == 'tbb':

            ##################################### SETTING IN TBB ##########################################
            plot_type = 'test_tbb'
            methods = [METHOD_NAMES.ALPHA, METHOD_NAMES.BETA, METHOD_NAMES.SR, METHOD_NAMES.MAHALANOBIS]
            ts = [1] * 4
            es = [0] * 4
            pt_type = [PERTURBATION_TYPE.NONE] * 4
            logits = [False] * 4

        elif scenario == 'pbb':

            ##################################### SETTING IN PBB ##########################################
            plot_type = 'test_pbb'
            methods = [METHOD_NAMES.ALPHA, METHOD_NAMES.BETA, METHOD_NAMES.ODIN, METHOD_NAMES.MAHALANOBIS]
            ts = [1, 1.5, 1.3, 1]
            es = [0.00035, 0.00035, 0, 0.0002]
            pt_type = [PERTURBATION_TYPE.ALPHA, PERTURBATION_TYPE.BETA, PERTURBATION_TYPE.ODIN, PERTURBATION_TYPE.MAHALANOBIS]
            logits = [False, False, False, True]

        else:
            print('Available scenario TBB and PBB, please select one of the two')
            sys.exit()


        plot_FRR_vs_TRR(dataset_name=dataset_name,
                        methods=methods, Ts=ts, es=es,
                        plot_type=plot_type, pt_type=pt_type,
                        normalization_kind=NORMALIZATION_KIND.ALREADY,
                        in_local=False, logits=logits, GAMMAS=10000)

    else:
        if scenario == 'tbb':

            ##################################### SETTING IN TBB ##########################################
            methods = [METHOD_NAMES.ALPHA, METHOD_NAMES.BETA, METHOD_NAMES.SR]
            ts = [1] * 3
            es = [0] * 3
            pt_type = [PERTURBATION_TYPE.NONE] * 3

        elif scenario == 'pbb':

            ##################################### SETTING IN PBB ##########################################
            methods = [METHOD_NAMES.ALPHA, METHOD_NAMES.BETA, METHOD_NAMES.ODIN, METHOD_NAMES.ODIN]
            ts = [1, 1.5, 1.3, 1000]
            es = [0.00035, 0.00035, 0, 0.0014]
            pt_type = [PERTURBATION_TYPE.ALPHA, PERTURBATION_TYPE.BETA, PERTURBATION_TYPE.ODIN, PERTURBATION_TYPE.ODIN]

        else:
            print('Available scenario TBB and PBB, please select one of the two')
            sys.exit()

        s = '\n'
        for i in range(len(methods)):

            # If not available create the randomized dataset that consists
            # of a mix of in-distribution and out-distribution samples.
            # p denotes the percentage of ood samples between the wrongly classified samples,
            # e.g, p = 1 ==> 1 over two wrongly classified samples is ood.

            for j in range(5):
                create_dataset_mixed(in_dataset=dataset_name.split("_",1)[1],
                                     out_dataset=dataset_name,
                                     method_name=methods[i],
                                     t=ts[i],
                                     e=es[i],
                                     p=1,
                                     i=j,
                                     pt_type=pt_type[i])

            m_a, st_a, m_f, st_f = get_FRR_vs_TRR_ood(method_name=methods[i],
                                               dataset_name=dataset_name,
                                               t=ts[i],
                                               e=es[i],
                                               p=1,
                                               iterations=5,
                                               pt_type=pt_type[i],
                                               GAMMAS=50000)

            if ts[i] == 1000:
                methods[i] = methods[i] + ' (default setting of ODIN) '
            s += methods[i].upper() + ': AUROC ' + str(round(m_a, 1)) + ' % / ' +  str(round(st_a, 1)) +\
                 ' % --- FRR ' + str(round(m_f, 1)) + ' % / ' + str(round(st_f, 1)) + ' %\n'

        print(s)
    return