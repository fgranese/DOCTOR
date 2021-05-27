from tests.compute_FRR_vs_TRR import *
from utils.plot_utils import plot_rocs
import time
import sklearn.metrics as skm

# FALSE IF YOU NEED TO CREATE THE DATAFRAME OF THE ERRORS,
# TRUE IF YOU WANT TO LOAD THE DATAFRAME ALREADY CREATED
LOAD_BETA = False
LOAD_ODIN = False
LOAD_ALPHA = False
LOAD_SR = False
LOAD_MAHALANOBIS = False

METHOD_NAMES = enum(ODIN='odin', SR='sr', ALPHA='alpha', BETA='beta', MAHALANOBIS='mahalanobis')
DATASET_TYPE = enum(TRAIN='train', TEST='test', VALIDATION='validation')
PERTURBATION_TYPE = enum(ODIN='odin', ALPHA='alpha', BETA='beta', ADV='adv', NONE='none', MAHALANOBIS='mahalanobis')
NORMALIZATION_KIND = enum(ALREADY='already', SOFTMAX='softmax', ZERONORM='zeronorm', MAXNORM='maxnorm', LOGITS='logits')

def get_df_wrong_correct(method_name, dataset_name, T, eps, in_local, dt_type, pt_type, logits=False):
    if not in_local:
        df_path = get_df_path(method_name=method_name,
                              dataset_name=dataset_name,
                              T=T, eps=eps, logits=logits,
                              dt_type=dt_type,
                              pt_type=pt_type)
    else:
        df_path = get_df_path_local(method_name=method_name,
                                    dataset_name=dataset_name,
                                    T=T, eps=eps, logits=logits,
                                    dt_type=dt_type,
                                    pt_type=pt_type)

    df = dut.load_from_csv(df_path)

    if dataset_name in ['cifar10', 'cifar100', 'tinyimagenet', 'svhn', 'amazon_fashion', 'amazon_software', 'imdb']:
        wrong = df.where(df['label'] != df['true_label']).dropna()
        wrong = wrong.index.values.tolist()
        correct = df.where(df['label'] == df['true_label']).dropna()
        correct = correct.index.values.tolist()
    else: # OOD case
        wrong = df.where(df['true_label'] == -1).dropna()
        wrong = wrong.index.values.tolist()
        correct = df.where(df['true_label'] == 1).dropna()
        correct = correct.index.values.tolist()

    return df, wrong, correct

def get_FRR_TRR(method_name, dataset_name, t, e, a, in_local, dt_type, pt_type, normalization_kind=NORMALIZATION_KIND.ALREADY, logits=False, GAMMAS=10000):
    if method_name == METHOD_NAMES.BETA and not LOAD_BETA:
        d, w, c = get_df_wrong_correct(method_name=METHOD_NAMES.BETA, dataset_name=dataset_name, T=t, eps=e, in_local=in_local, logits=logits, dt_type=dt_type, pt_type=pt_type)
        T_0, T_1 = compute_FRR_TRR(method_name=method_name, dataset_name=dataset_name, df=d, wrong=w, correct=c, T=t, eps=e, alpha=a, logits=logits, GAMMAS=GAMMAS,
                                   normalization_kind=normalization_kind, dt_type=dt_type, pt_type=pt_type)
    elif method_name == METHOD_NAMES.ALPHA and not LOAD_ALPHA:
        d, w, c = get_df_wrong_correct(method_name=METHOD_NAMES.ALPHA, dataset_name=dataset_name, T=t, eps=e, in_local=in_local, logits=logits, dt_type=dt_type, pt_type=pt_type)
        T_0, T_1 = compute_FRR_TRR(method_name=method_name, dataset_name=dataset_name, df=d, wrong=w,
                              correct=c, T=t, eps=e, alpha=a, logits=logits, GAMMAS=GAMMAS, normalization_kind=normalization_kind, dt_type=dt_type, pt_type=pt_type)
    elif method_name == METHOD_NAMES.ODIN and not LOAD_ODIN:
        d, w, c = get_df_wrong_correct(method_name=METHOD_NAMES.ODIN, dataset_name=dataset_name, T=t, eps=e, in_local=in_local, logits=logits, dt_type=dt_type, pt_type=pt_type)
        T_0, T_1 = compute_FRR_TRR(method_name=method_name, dataset_name=dataset_name, df=d, wrong=w, correct=c, T=t, eps=e, alpha=a, logits=logits, GAMMAS=GAMMAS,
                                   normalization_kind=normalization_kind, dt_type=dt_type, pt_type=pt_type)
    elif method_name == METHOD_NAMES.SR and not LOAD_SR:
        d, w, c = get_df_wrong_correct(method_name=METHOD_NAMES.SR, dataset_name=dataset_name, T=1, eps=0, in_local=in_local, logits=logits, dt_type=dt_type, pt_type=pt_type)
        T_0, T_1 = compute_FRR_TRR(method_name=method_name, dataset_name=dataset_name, df=d, wrong=w, correct=c, T=1, eps=0, alpha=a, logits=logits, GAMMAS=GAMMAS,
                                   normalization_kind=normalization_kind, dt_type=dt_type, pt_type=pt_type)
    elif method_name == METHOD_NAMES.MAHALANOBIS and not LOAD_MAHALANOBIS:
        d, w, c = get_df_wrong_correct(method_name=METHOD_NAMES.MAHALANOBIS, dataset_name=dataset_name, T=t, eps=e, in_local=in_local, logits=logits, dt_type=dt_type, pt_type=pt_type)
        T_0, T_1 = compute_FRR_TRR(method_name=method_name, dataset_name=dataset_name, df=d, wrong=w, correct=c, T=t, eps=e, alpha=a, logits=logits, GAMMAS=GAMMAS,
                                   normalization_kind=normalization_kind, dt_type=dt_type, pt_type=pt_type)
    else:
        if not in_local:
            error_path_ = get_error_path(dataset_name=dataset_name, method_name=method_name, T=t, eps=e, logits=logits, dt_type=dt_type, pt_type=pt_type, norm=normalization_kind)
        else:
            error_path_ = get_error_path_local(dataset_name=dataset_name, method_name=method_name, T=t, eps=e, logits=logits, dt_type=dt_type, pt_type=pt_type, norm=normalization_kind)
        df_ = dut.load_from_csv(error_path_)
        T_0, T_1 = df_['T_0'].to_list(), df_['T_1'].to_list()
    return T_0, T_1

def plot_FRR_vs_TRR(dataset_name, methods, Ts, es, plot_type, pt_type, logits, normalization_kind=NORMALIZATION_KIND.ALREADY,
                    in_local=False, GAMMAS=10000, dt_type=DATASET_TYPE.TEST):
    T_0 = []
    T_1 = []
    labels = []
    colors = []
    aurocs = []

    for i in range(len(methods)):
        start_time = time.time()

        #print(methods[i])

        t_0, t_1 = get_FRR_TRR(method_name=methods[i], dataset_name=dataset_name,
                               t=Ts[i], e=es[i], a=2, in_local=in_local,
                               logits=logits[i], GAMMAS=GAMMAS,
                               normalization_kind=normalization_kind,
                               dt_type=dt_type, pt_type=pt_type[i])
        T_0.append(t_0)
        T_1.append(t_1)
        aurocs.append(skm.auc(t_0, t_1))

        x = np.interp(0.95, np.sort(T_1[i]), np.sort(T_0[i]))
        print(methods[i].upper(), ': AUROC', round(skm.auc(t_0, t_1) * 100, 1), '% --- FRR (95% TRR)', round(x * 100, 1), '%')

        if methods[i] == METHOD_NAMES.ALPHA:
            labels.append(r'$D_\alpha$')
            colors.append('green')
        elif methods[i] == METHOD_NAMES.BETA:
            labels.append(r'$D_\beta$')
            colors.append('orange')
        if methods[i] == METHOD_NAMES.ODIN:
            labels.append(r'$ODIN$')
            colors.append('blue')
        elif methods[i] == METHOD_NAMES.SR:
            labels.append(r'$SR$')
            colors.append('red')
        elif methods[i] == METHOD_NAMES.MAHALANOBIS:
            labels.append(r'$MHLNB$')
            colors.append('violet')

        #print("--- %s seconds ---" % (time.time() - start_time))

    plot_rocs(dataset_name=dataset_name, T_0=T_0, T_1=T_1, labels=labels, colors=colors, plot_type=plot_type)

    return aurocs
