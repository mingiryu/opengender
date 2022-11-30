from sklearn.metrics import confusion_matrix


def components(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred, ["male", "female", "unknown"])
    mm = M[0][0]
    mf = M[0][1]
    mu = M[0][2]
    fm = M[1][0]
    ff = M[1][1]
    fu = M[1][2]
    return mm, mf, mu, fm, ff, fu


def error_coded(y_true, y_pred):
    mm, mf, mu, fm, ff, fu = components(y_true, y_pred)
    return (fm + mf + mu + fu) / (mm + fm + mf + ff + mu + fu)


def error_coded_without_na(y_true, y_pred):
    mm, mf, mu, fm, ff, fu = components(y_true, y_pred)
    return (fm + mf) / (mm + fm + mf + ff)


def na_coded(y_true, y_pred):
    mm, mf, mu, fm, ff, fu = components(y_true, y_pred)
    return (mu + fu) / (mm + fm + mf + ff + mu + fu)


def errror_gender_bias(y_true, y_pred):
    mm, mf, mu, fm, ff, fu = components(y_true, y_pred)
    return (mf - fm) / (mm + fm + mf + ff)
