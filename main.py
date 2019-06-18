import numpy as np
import pandas as pd

# import rates
import fairness_measures as fm


def bin_col(col, th):
    return (col > th) * 1


def assess(A, Y, R, th_g1, th_g2, eps, print_res=False):

    group1 = (A == 1)
    group2 = (A == 0)

    label_1 = Y[group1]
    label_2 = Y[group2]

    pred_1 = bin_col(R[group1], th_g1)
    pred_2 = bin_col(R[group2], th_g2)

    ind = fm.independence(pred_1, pred_2, eps, print_res)
    sep = fm.separation(pred_1, pred_2, label_1, label_2, eps, print_res)
    suf = fm.sufficiency(pred_1, pred_2, label_1, label_2, eps, print_res)

    # ratios = [ind['ratio_pr'] - 1,
    #           sep['ratio_tpr'] - 1,
    #           sep['ratio_fpr'] - 1,
    #           suf['ratio_ppv'] - 1,
    #           suf['ratio_npv'] - 1]
    # non_nan_ratios = [x for x in ratios if x != np.nan]
    #
    # loss = np.linalg.norm(non_nan_ratios)

    passed = int(ind['ind_fair']) + int(sep['sep_fair']) + \
        int(suf['suf_fair'])

    merged_dictionaries = {'th_g1': th_g1, 'th_g2': th_g2,
                           'tests_passed': passed,
                           **ind, **sep, **suf}

    result = pd.Series(merged_dictionaries)

    return result


def assess_models(A, Y, R, th_values_g1, th_values_g2, eps):

    df_result = pd.DataFrame()

    for i in range(len(th_values_g1)):
        for j in range(len(th_values_g2)):
            th_g1 = th_values_g1[i]
            th_g2 = th_values_g2[j]

            result_s = assess(A, Y, R, th_g1, th_g2, eps)

            # Pandas reorders columns after using append
            df_result = df_result.append(
                result_s, ignore_index=True)[result_s.index.tolist()]

            # if result_s['tests_passed'] > 0:
            #     print('For th_g1 = %.2f and th_g2 =%.2f tests passed: %.2f' %
            #           (th_g1, th_g2, result_s['tests_passed']))
            #     if result_s['ind_fair']:
            #         print('Independence is satisfied')
            #     if result_s['sep_fair']:
            #         print('Separation is satisfied')
            #     if result_s['suf_fair']:
            #         print('Sufficiency is satisfied')
    return df_result


data = np.load('df.npy')
df = pd.DataFrame(data, columns=['A', 'Y', 'r1', 'r2'])

eps = 0.3
th_values_bl = np.arange(21) * 0.1
th_values_wh = np.arange(21) * 0.1

# Studying best models for r1
result_r1 = assess_models(df['A'], df['Y'], df['r1'],
                          th_values_bl, th_values_wh, eps)
result_r1_sorted = result_r1.sort_values('tests_passed', ascending=False)

# Studying best models for r2
result_r2 = assess_models(df['A'], df['Y'], df['r2'],
                          th_values_bl, th_values_wh, eps)
result_r2_sorted = result_r2.sort_values('tests_passed', ascending=False)

# Best model for r2 detailed
res_best_r2 = assess(df['A'], df['Y'], df['r2'], 0.6, 0.4, eps)


