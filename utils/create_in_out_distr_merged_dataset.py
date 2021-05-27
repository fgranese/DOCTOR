import numpy as np
import pandas as pn
import random as rd
import os

def create_in_out_distr_merged_dataset(in_data=None, out_data=None, percentage_rejection_from_in_distribution=.1, iterations=1):
    """

    :param in_data: path to a pandas data-frame where each row corresponds to an entry and the first c columns correspond to
    the soft-probabilities class by class, the second to last column is the arg-max index for the soft-probabilities and the
    last column represents the target label
    :param out_data: path to a pandas data-frame such as the one above: in this case the correspondence between the arg-max
    index for the soft-probabilities and the target label must be ignored since these samples comes from the
    out-distribution
    :param percentage_rejection_from_in_distribution: percentage of samples to be rejected from the in-distribution
    :return: newly build data-frame with data to be discriminated; each row corresponds to an entry and the first c columns
    correspond to the soft-probabilities class by class, the third to last column is the arg-max index for the
    soft-probabilities, the second to last column represents the acceptance/rejection label, and the last column keeps track
    of the fact that the sample comes either from the in-distribution (True) or thee out-distribution (False);
    the perfect discriminator should agree with the acceptance/rejection column

    """

    # copy the data-frames, turn then into numpy arrays of arrays
    in_data = pn.read_csv(filepath_or_buffer=in_data, sep=',')
    out_data = pn.read_csv(filepath_or_buffer=out_data, sep=',')
    out_data = out_data.where(out_data['true_label'] == -1).dropna()

    colnames = list(in_data.columns)[0:-1] + ["acc/rej", "in_distr"]
    
    in_data_cpy_mat = in_data.values
    out_data_cpy_mat = out_data.values

    in_data_cpy_mat_to_be_merged = []  # np.zeros(shape=(in_data_cpy_mat.shape[0], 2))
    out_data_cpy_mat_to_be_merged = []  # np.zeros(shape=(out_data_cpy_mat.shape[0], 2))

    for row in range(in_data_cpy_mat.shape[0]):
        list_tmp = []
        if in_data_cpy_mat[row, -1] == in_data_cpy_mat[row, -2]:
            list_tmp.append(1)
        else:
            list_tmp.append(-1)
        list_tmp.append(True)
        in_data_cpy_mat_to_be_merged.append(list_tmp)

    for row in range(out_data_cpy_mat.shape[0]):
        list_tmp = [-1, False]
        out_data_cpy_mat_to_be_merged.append(list_tmp)

    in_data_cpy_mat_to_be_merged = np.array(in_data_cpy_mat_to_be_merged).reshape((in_data_cpy_mat.shape[0], 2))
    out_data_cpy_mat_to_be_merged = np.array(out_data_cpy_mat_to_be_merged).reshape((out_data_cpy_mat.shape[0], 2))

    in_data_cpy_mat = np.column_stack((in_data_cpy_mat[:, 0:-1], in_data_cpy_mat_to_be_merged))
    out_data_cpy_mat = np.column_stack((out_data_cpy_mat[:, 0:-1], out_data_cpy_mat_to_be_merged))

    in_acceptance_idx = list(np.where(in_data_cpy_mat[:, -2] == 1)[0])
    in_rejections_idx = list(np.where(in_data_cpy_mat[:, -2] == -1)[0])


    out_rejections_idx = [i for i in range(out_data_cpy_mat.shape[0])]
    out_rejection_samples_idx = rd.sample(out_rejections_idx,
                                          int(
                                              percentage_rejection_from_in_distribution * len(in_rejections_idx)))


    selected_accepted_in_samples = in_data_cpy_mat[in_acceptance_idx, :]
    selected_rejected_in_samples = in_data_cpy_mat[in_rejections_idx, :]
    selected_rejected_out_samples = out_data_cpy_mat[out_rejection_samples_idx, :]

    res = np.concatenate((selected_accepted_in_samples, selected_rejected_in_samples, selected_rejected_out_samples))
    res_df = pn.DataFrame(data=res, columns=colnames)

    return res_df

def create_dataset_mixed(in_dataset, out_dataset, method_name, pt_type, t , e, p, i):
    from utils.dataset_utils import save_to_csv
    from utils.files_utils import get_df_path

    df_in_path = get_df_path(method_name=method_name, dataset_name=in_dataset, T=t, eps=e, logits=False, dt_type='test',
                             pt_type=pt_type)
    df_out_path = get_df_path(method_name=method_name, dataset_name=out_dataset, T=t, eps=e, logits=False,
                              dt_type='test', pt_type=pt_type)
    df_out_path_final = df_out_path.replace(out_dataset, out_dataset + "_3_p_" + str(p) + "_it_" + str(i))

    # print('Final path', df_out_path_final)
    # print('Out path', df_out_path)
    # print('In path', df_in_path)

    if not os.path.exists(df_out_path_final):
        print("Does not exist")
        res = create_in_out_distr_merged_dataset(in_data=df_in_path, out_data=df_out_path, percentage_rejection_from_in_distribution=p,
                                             iterations=i)
        print('Len', len(res))
        print('Number of in-distr, correctly classified', len(res.where((res['in_distr'] == 1)  & (res['acc/rej'] == 1)).dropna()))
        print('Number of in-distr, wrongly classified', len(res.where((res['in_distr'] == 1) & (res['acc/rej'] == -1)).dropna()))
        print('Number of out-distr', len(res.where((res['in_distr'] == 0) & (res['acc/rej'] == -1)).dropna()))

        del res['in_distr']
        res = res.rename(columns={'acc/rej':'true_label'})
        save_to_csv(df_out_path_final, res, False)

    return




