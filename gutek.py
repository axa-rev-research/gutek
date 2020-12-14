import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from nltk.tokenize import sent_tokenize


# code for optimal sampling
def update_entries(entries, n_cols, resetter=0):
    # update list of paragraps to remove
    if len(entries) == 1:
        entries[0] += 1
        carry = 0
        if entries[0] >= n_cols:
            carry = 1
            entries[0] = resetter
    else:
        potential_carry = 1 if 1 + entries[0] >= n_cols - len(entries) + 1 else 0
        child_resetter = entries[0] + 2 if not potential_carry == 1 else resetter + 1
        new_entries, old_carry = update_entries(entries[1:], n_cols, resetter=child_resetter)
        carry = potential_carry * old_carry
        entries = [resetter if carry else entries[0] + old_carry] + new_entries
    return entries, carry


def sample_array(n_cols, n_samples):
    # create the boolean array representing the sampling dataset
    n_samples = min(n_samples, 2**n_cols)
    # build the sample without change
    arrays = [np.array([1 for i in range(n_cols)])]
    n_remaining = n_samples - 1
    degree = 1
    while n_remaining > 0:
        entries = list(range(degree))
        carry = 0
        while not carry and n_remaining > 0:
            # build the array
            all_ones = np.ones(n_cols)
            for e in entries:
                all_ones[e] = 0
            arrays.append(all_ones)
            # update the entries to change
            entries, carry = update_entries(entries, n_cols)
            n_remaining -= 1
        # update the degree
        degree += 1
    array = np.vstack(arrays).T
    np.random.shuffle(array)
    return array.T


# interpreters
def interpret_binary(segments, model, n_samples):
    # build the neighborhood
    X = sample_array(len(segments), n_samples)
    # query the model
    texts = [
        ' '.join([segment for segment, presence in zip(segments, x_) if presence]) for x_ in X
    ]
    y = model(texts).flatten()
    # train the surrogate
    try:
        coefs = LogisticRegression().fit(X, y).coef_
    except ValueError:
        # not enough samples to find distinction: dont cheat, just return 0s
        return [0] * len(segments)
    return coefs.flatten()


def interpret_regression(segments, model, n_samples):
    # build the neighborhood
    X = sample_array(len(segments), n_samples)
    # query the model
    texts = [
        ' '.join([segment for segment, presence in zip(segments, x_) if presence]) for x_ in X
    ]
    y = model(texts)[:, 1]
    # train the surrogate
    coefs = LinearRegression().fit(X, y).coef_
    return coefs.flatten()


def gutek_interpreter(model, question, context, binary=True, n_samples=10, batch_size=2):
    # define the predicion function
    f = model.predict if binary else model.predict_proba

    def pred_f(texts):
        if type(texts) == str:
            return f(question, texts)
        else:
            res = []
            for i in range(0, len(texts), batch_size):
                res.append(f(
                    [question] * len(texts[i: i + batch_size]),
                    texts[i: i + batch_size]
                ))
            res = np.vstack(res)
            return res

    # run the interpretation
    if binary:
        return interpret_binary(sent_tokenize(context), pred_f, n_samples)
    else:
        return interpret_regression(sent_tokenize(context), pred_f, n_samples)
