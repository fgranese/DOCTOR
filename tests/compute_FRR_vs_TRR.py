import pandas as pd
import math

import utils.dataset_utils as dut
from lib_discriminators.discriminators import *
from utils import GUI_tools
from utils.files_utils import *
from utils.var_utils import *

# GLOBAL VARIABLE
PROGRESS_BAR = 50

METHOD_NAMES = enum(ODIN='odin', SR='sr', BETA='beta', ALPHA='alpha', MAHALANOBIS='mahalanobis')

def compute_FRR_TRR(method_name, dataset_name, df, wrong, correct, T, eps, dt_type, pt_type, alpha=2, GAMMAS=10000, logits=False, normalization_kind=NORMALIZATION_KIND.ALREADY):
    gui_tool = GUI_tools.GUI_tools()

    if normalization_kind == NORMALIZATION_KIND.LOGITS:
        if method_name == METHOD_NAMES.MAHALANOBIS:
            data = dut.load_from_csv(get_df_path(method_name=method_name, dataset_name=dataset_name, T=1, eps=0,
                                                 logits=True, dt_type=DATASET_TYPE.TRAIN, pt_type=PERTURBATION_TYPE.NONE))
            F, min_ratio, max_ratio = mahalanobis(df_test=df, df_tr=data)
            #print('Train:' + '\n', data)

    else:
        df = normalize(df, normalization_kind)
        if method_name == METHOD_NAMES.BETA:
            F = doctor_ratio(pe_x(df))
        elif method_name == METHOD_NAMES.ALPHA:
            F = doctor_ratio(g_x(df, alpha))
        elif method_name == METHOD_NAMES.ODIN or method_name == METHOD_NAMES.SR:
            F = soft_odin(df)
        elif method_name == METHOD_NAMES.MAHALANOBIS:
            data = dut.load_from_csv(get_df_path(method_name=method_name, dataset_name=dataset_name, T=1, eps=0, logits=logits, dt_type=DATASET_TYPE.TRAIN, pt_type=PERTURBATION_TYPE.NONE))
            F, min_ratio, max_ratio = mahalanobis(df_test=df, df_tr=data)
            #print('Train:' + '\n', data)

    #print('Test:' + '\n', df)
    max_ratio = np.max(F)
    min_ratio = np.min(F)

    if dataset_name in ['cifar10', 'cifar100', 'svhn', 'tinyimagenrt', 'imdb', 'amazon_fashion', 'amazon_software']:
        gap = int((max_ratio - min_ratio) * GAMMAS)

        if method_name == METHOD_NAMES.MAHALANOBIS:
            gap = 10000
    else:
        gap = GAMMAS

    thresholds = np.linspace(min_ratio, max_ratio, gap)

    slop = '[>' + ('.' * PROGRESS_BAR) + ']'
    print("\nComputing plot FRR VS TRR with method {} in progress:".format(method_name.upper()))
    print(slop, end='')

    T_0 = []
    T_1 = []
    for i, thr in enumerate(thresholds, 1):

        if method_name in [METHOD_NAMES.ALPHA, METHOD_NAMES.BETA, METHOD_NAMES.MAHALANOBIS]:
            A, A_c = decision_region_doctor(F, thr)
        else:
            A, A_c = decision_region_odin(F, thr)
        type_0, type_1 = errors(A, A_c, wrong, correct)

        T_0.append(type_0)
        T_1.append(1 - type_1)

        if i % (math.ceil(gap / PROGRESS_BAR)) == 0 or i == gap - 1:
            slop = gui_tool.print_progress(i, slop, gap, progress_bar=PROGRESS_BAR)

    df_path = get_error_path(dataset_name=dataset_name, method_name=method_name, T=T, eps=eps, logits=logits, dt_type=dt_type, pt_type=pt_type, norm=normalization_kind)
    dut.save_to_csv(df_path, pd.DataFrame({'T_0': T_0, 'T_1': T_1}), False)
    return  T_0, T_1