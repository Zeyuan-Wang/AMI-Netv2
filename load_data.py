
import pandas as pd
import numpy as np


# load data and preprocess
def data_preparation(path, type='excel'):

    if type is 'csv':
        df = pd.read_csv(path)
    if type is 'excel':
        df = pd.read_excel(path)

    # x and y
    y = np.array(df['y'])
    x = df.drop(['y'], axis=1)

    # for each patient, convert their containing symptoms to a list of words
    bin_feats = []

    for i in x.columns:
        if len(x[i].unique()) == 2:
            bin_feats.append(i)

    x_bin = x[bin_feats]
    x_bin = x_bin.replace(0, np.nan)
    x_bin_features = bin_gen(x_bin)

    # pad to the maximum length and convert to matrix
    x_bin_features, feats, tokens, feat_max = bin_pad_convert(x_bin_features, bin_feats)

    return x_bin_features, feats, tokens, feat_max, y


# convert to a list of words for binary features
def bin_gen(x):

    x_features = []

    for i in range(x.shape[0]):
        index = x.columns[x.iloc[i, :].notnull()]
        feats = np.array(index)
        x_features.append(feats)

    return np.array(x_features)


# convert to matrix and generate descriptions (binary features)
def bin_pad_convert(txt_features, bin_feat_list):

    bin_feat_max = max([len(feat) for feat in txt_features])
    bin_feats = ['pad'] + bin_feat_list
    tokens = len(bin_feats)

    x_features = np.zeros((len(txt_features), bin_feat_max), dtype='int32')
    feat_index = dict([(char, i) for i, char in enumerate(bin_feats)])

    for i, input_text in enumerate(txt_features):
        for t, char in enumerate(input_text):
            x_features[i, t] = feat_index[char]

    return x_features, bin_feats, tokens, bin_feat_max
