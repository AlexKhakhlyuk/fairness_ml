import numpy as np
import rates


def ratio(x, y):
    big = max(x, y)
    small = min(x, y)

    if small == 0 and big == 0:
        return 1
    elif small == 0 and big != 0:
        return np.inf
    elif small == np.nan or big == np.nan:
        return np.inf
    else:
        return big / small


def independence(col1, col2, eps, print_res=False):
    pr1 = rates.pr(col1)
    pr2 = rates.pr(col2)

    ratio_pr = ratio(pr1, pr2)

    fair = (1 / ratio_pr) >= (1 - eps)

    if print_res:

        print('Independence in group 1: ', pr1)
        print('Independence in group 2: ', pr2)
        print('Ratio: ', ratio_pr, '\n')
        print('Independence fulfilled: ', str(fair).upper())
        print("___________________________________________", '\n\n')

    return {'ind_fair': fair, 'pr1': pr1, 'pr2': pr2,
            'ratio_pr': ratio_pr}


def separation(col1, col2, labels1, labels2, th, print_res=False):
    fpr1 = rates.fpr(col1, labels1)
    fpr2 = rates.fpr(col2, labels2)
    fnr1 = rates.fnr(col1, labels1)
    fnr2 = rates.fnr(col2, labels2)

    ratio_fpr = ratio(fpr1, fpr2)
    ratio_fnr = ratio(fnr1, fnr2)

    # 1/ratio = small / big
    fpr_fair = (1 / ratio_fpr) >= (1 - th)
    fnr_fair = (1 / ratio_fnr) >= (1 - th)
    fair = fpr_fair and fnr_fair

    if print_res:
        print('False positive ratio in group 1: ', fpr1)
        print('False positive ratio in group 2: ', fpr2)
        print('Ratio: ', ratio_fpr, '\n')

        print('False negative ratio in group 1: ', fnr1)
        print('False negative ratio in group 2: ', fnr2)
        print('Ratio: ', ratio_fnr, '\n')

        print('Separation fulfilled: ', str(fair).upper())
        print("___________________________________________", '\n\n')

    return {'sep_fair': fair, 'fpr1': fpr1, 'fpr2': fpr2,
            'fnr1': fnr1, 'fnr2': fnr2,
            'ratio_fpr': ratio_fpr,
            'ratio_fnr': ratio_fnr}


def sufficiency(col1, col2, labels1, labels2, eps, print_res=False):
    ppv1 = rates.ppv(col1, labels1)
    ppv2 = rates.ppv(col2, labels2)
    npv1 = rates.npv(col1, labels1)
    npv2 = rates.npv(col2, labels2)

    ratio_ppv = ratio(ppv1, ppv2)
    ratio_npv = ratio(npv1, npv2)

    # 1/ratio = small / big
    ppv_fair = (1 / ratio_ppv) >= (1 - eps)
    npv_fair = (1 / ratio_npv) >= (1 - eps)
    fair = ppv_fair and npv_fair

    if print_res:
        print('Positive predictive value in group 1: ', ppv1)
        print('Positive predictive value in group 2: ', ppv2)
        print('Ratio: ', ratio_ppv, '\n')

        print('Negative predictive value in group 1: ', npv1)
        print('Negative predictive value in group 2: ', npv2)
        print('Ratio: ', ratio_npv, '\n')

        print('Sufficiency fulfilled: ', str(fair).upper())
        print("___________________________________________", '\n\n')

    return {'suf_fair': fair, 'ppv1': ppv1, 'ppv2': ppv2,
            'npv1': npv1, 'npv2': npv2,
            'ratio_ppv': ratio_ppv,
            'ratio_npv': ratio_npv}
