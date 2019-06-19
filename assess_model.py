import numpy as np
import pandas as pd

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

    """
    To ideally satisfy Independence, Separation and Sufficiency, 
    the ratios for pr, fpr, fnr, ppv, npv should be equal to 1.
     Thus, (ratio_x - 1) represents the error, distance to the ideal value,
     it should be as close to 0, as possible. 
    Here I tried to create a non-convex loss function: 
    a MSE of ratios. 
    It didn't work out well, partly because of extreme values of 
    threshold values. 
    For high and low values for thresholds, the predictor 
    classifies all datapoints into 1 category.
    This makes ppv and other measures 1 or 0 for both groups, ratios 
    are equal to 1, the error is equal to 0. The system is fair formally, 
    but such predictor has 0 information gain.
    """
    # ratios = [ind['ratio_pr'] - 1,
    #           sep['ratio_fpr'] - 1,
    #           sep['ratio_fnr'] - 1,
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
