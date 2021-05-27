from utils.var_utils import enum

ROOT = './'
METHOD_NAMES = enum(ODIN='odin', SR='sr', ALPHA='alpha', BETA='beta', MAHALANOBIS='mahalanobis')
DATASET_TYPE = enum(TRAIN='train', TEST='test', VALIDATION='validation')
PERTURBATION_TYPE = enum(ODIN='odin', ALPHA='alpha', BETA='beta', ADV='adv', NONE='none', MAHALANOBIS='mahalanobis')
NORMALIZATION_KIND = enum(ALREADY='already', SOFTMAX='softmax', ZERONORM='zeronorm', MAXNORM='maxnorm', LOGITS='logits')

def get_error_path(method_name, dataset_name, T, eps, dt_type, pt_type, logits=False, norm=NORMALIZATION_KIND.ALREADY):
    error_path = ROOT + "datasets/"
    if dataset_name[:6] == 'amazon':
        error_path += "amazon/" + dataset_name[7:len(dataset_name)]
    elif dataset_name == 'toy_star':
        error_path += "toy/" + dataset_name
    elif dataset_name[:8] == 'toy_pred' or dataset_name[:8] == 'toy_true':
        error_path += "toy/" + dataset_name[:8]
    else:
        error_path += dataset_name
    if pt_type != PERTURBATION_TYPE.NONE:
        error_path += "/error_" + method_name + "/" + dataset_name + "_T_" + str(T) + "_eps_" + str(eps) + "_errors_" + method_name + "_pt_" + pt_type + "_" + dt_type + ".csv"
    else:
        error_path += "/error_" + method_name + "/" + dataset_name + "_T_" + str(T) + "_eps_" + str(eps) + "_errors_" + method_name + "_" + dt_type + ".csv"
    if logits:
        error_path = error_path.replace(".csv", "_logits.csv")
    if method_name == METHOD_NAMES.MAHALANOBIS and norm == NORMALIZATION_KIND.SOFTMAX:
        error_path = error_path.replace(".csv", "_soft.csv")
    return error_path

def get_df_path(method_name, dataset_name, T, eps, dt_type, pt_type, logits=False):
    df_path = ROOT + "datasets/"
    if dataset_name[:6] == 'amazon':
        df_path += "amazon/" + dataset_name[7:len(dataset_name)]
    elif dataset_name == 'toy_star':
        df_path += "toy/" + dataset_name
    elif dataset_name[:8] == 'toy_pred' or dataset_name[:8] == 'toy_true':
        df_path += "toy/" + dataset_name[:8]
    else:
        df_path += dataset_name
    if method_name == METHOD_NAMES.SR or pt_type == PERTURBATION_TYPE.NONE:
        df_path += "/data/" + dataset_name + "_T_1_eps_0_" + dt_type + ".csv"
    elif (method_name == METHOD_NAMES.BETA or method_name == METHOD_NAMES.ALPHA or method_name == METHOD_NAMES.MAHALANOBIS) and pt_type != PERTURBATION_TYPE.NONE:
        df_path += "/data_perturb_our/" + dataset_name + "_T_" + str(T) + "_eps_" + str(eps) + "_pt_" + pt_type + "_" + dt_type + ".csv"
    elif (method_name == METHOD_NAMES.BETA or method_name == METHOD_NAMES.ALPHA or method_name == METHOD_NAMES.MAHALANOBIS) and pt_type == PERTURBATION_TYPE.NONE:
        df_path += "/data/" + dataset_name + "_T_" + str(T) + "_eps_" + str(eps) + "_" + dt_type + ".csv"
    elif method_name == METHOD_NAMES.ODIN:
        df_path += "/data_perturb/" + dataset_name + "_T_" + str(T) + "_eps_" + str(eps) + "_pt_" + pt_type + "_" + dt_type + ".csv"
    if logits:
        df_path = df_path.replace(".csv", "_logits.csv")
    return df_path

def get_plot_path(dataset_name, plot_type, dt_type='', pt_type='', logits=False):
    if dataset_name[:6] == 'amazon':
        img_path = ROOT + "plots/amazon/" + dataset_name[7:len(dataset_name)] + "/" + dataset_name[7:len(dataset_name)] + "_" + plot_type + ".png"
    elif dataset_name[:3] == 'toy':
        img_path = ROOT + "plots/toy/toy_" + plot_type + ".png"
    else:
        img_path = ROOT + "plots/" + dataset_name + "/" + dataset_name + "_" + plot_type + "_pt_" + pt_type + "_" + dt_type + ".png"
    if logits:
        img_path = img_path.replace(".csv", "_logits.csv")
    return img_path

def get_aurocs_path(dataset_name, aurocs_type, dt_type, pt_type, logits=False):
    if dataset_name[:6] == 'amazon':
        aurocs_path = ROOT + "datasets/" + dataset_name[7:len(dataset_name)] + "/" + dataset_name[7:len(dataset_name)] + "_" + aurocs_type + ".csv"
    else:
        aurocs_path = ROOT + "datasets/" + dataset_name + "/" + dataset_name + "_" + aurocs_type + "_pt_" + pt_type + "_" + dt_type + ".csv"
    if logits:
        aurocs_path = aurocs_path.replace(".csv", "_logits.csv")
    return aurocs_path

def get_df_path_local(method_name, dataset_name, T, eps, dt_type, pt_type, logits=False):
    global ROOT
    ROOT = './'
    return get_df_path(method_name=method_name, dataset_name=dataset_name, T=T, eps=eps, dt_type=dt_type, pt_type=pt_type, logits=logits)

def get_error_path_local(method_name, dataset_name, T, eps, dt_type, pt_type, logits=False, norm=NORMALIZATION_KIND.ALREADY):
    global ROOT
    ROOT = './'
    return get_error_path(method_name=method_name, dataset_name=dataset_name, T=T, eps=eps, dt_type=dt_type, pt_type=pt_type, logits=logits, norm=norm)

def get_plot_path_local(dataset_name, plot_type, dt_type, pt_type, logits=False):
    global ROOT
    ROOT = './'
    return get_plot_path(dataset_name=dataset_name, plot_type=plot_type, dt_type=dt_type, pt_type=pt_type, logits=logits)



