#!/usr/bin/env python

import os
import re
import sys
import warnings
from argparse import ArgumentParser, ArgumentTypeError
from itertools import product
from pprint import pprint
from shutil import rmtree
from tempfile import mkdtemp, gettempdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype,
    is_object_dtype)
import rpy2.rinterface_lib.embedded as r_embedded
r_embedded.set_initoptions(
    ('rpy2', '--quiet', '--no-save', '--max-ppsize=500000'))
import rpy2.robjects as robjects
import seaborn as sns
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='sklearn.utils.deprecation')
from eli5 import explain_weights_df
warnings.filterwarnings('always', category=FutureWarning,
                        module='sklearn.utils.deprecation')
from joblib import Memory, Parallel, delayed, dump, parallel_backend
from natsort import natsorted
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='rpy2.robjects.pandas2ri')
from rpy2.robjects import numpy2ri, pandas2ri
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='rpy2.robjects.pandas2ri')
from rpy2.robjects.packages import importr
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection._base import SelectorMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    auc, average_precision_score, balanced_accuracy_score,
    precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    FunctionTransformer, MinMaxScaler, OneHotEncoder, PowerTransformer,
    RobustScaler, StandardScaler)
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

numpy2ri.activate()
pandas2ri.activate()

from sklearn_extensions.compose import ExtendedColumnTransformer
from sklearn_extensions.ensemble import (
    CachedExtraTreesClassifier, CachedGradientBoostingClassifier,
    CachedRandomForestClassifier)
from sklearn_extensions.feature_selection import (
    ANOVAFScorerClassification, CachedANOVAFScorerClassification,
    CachedChi2Scorer, CachedMutualInfoScorerClassification, CFS, Chi2Scorer,
    ColumnSelector, DESeq2, DreamVoom, EdgeR, EdgeRFilterByExpr, FCBF, Limma,
    LimmaVoom, MutualInfoScorerClassification, ReliefF, RFE, SelectFromModel,
    SelectKBest, VarianceThreshold)
from sklearn_extensions.model_selection import (
    ExtendedGridSearchCV, ExtendedRandomizedSearchCV,
    StratifiedGroupShuffleSplit)
from sklearn_extensions.pipeline import ExtendedPipeline
from sklearn_extensions.preprocessing import (
    DESeq2RLEVST, EdgeRTMMLogCPM, LimmaBatchEffectRemover)
from sklearn_extensions.svm import CachedLinearSVC
from sklearn_extensions.utils import _determine_key_type


def setup_pipe_and_param_grid(cmd_pipe_steps):
    pipe_steps = []
    pipe_param_routing = None
    pipe_step_names = []
    pipe_props = {'has_selector': False, 'uses_rjava': False}
    param_grid = []
    param_grid_dict = {}
    pipe_step_keys = []
    pipe_step_types = []
    for step_idx, step_keys in enumerate(cmd_pipe_steps):
        if any(k.title() == 'None' for k in step_keys):
            pipe_step_keys.append(
                [k for k in step_keys if k.title() != 'None'] + [None])
        else:
            pipe_step_keys.append(step_keys)
        if len(step_keys) > 1:
            pipe_step_names.append('|'.join(step_keys))
        else:
            pipe_step_names.append(step_keys[0])
    for pipe_step_combo in product(*pipe_step_keys):
        params = {}
        for step_idx, step_key in enumerate(pipe_step_combo):
            if step_key:
                if step_key in pipe_config:
                    estimator = pipe_config[step_key]['estimator']
                else:
                    run_cleanup()
                    raise RuntimeError('No pipeline config exists for {}'
                                       .format(step_key))
                if isinstance(estimator, SelectorMixin):
                    step_type = 'slr'
                    pipe_props['has_selector'] = True
                elif isinstance(estimator, TransformerMixin):
                    step_type = 'trf'
                elif isinstance(estimator, ClassifierMixin):
                    step_type = 'clf'
                elif isinstance(estimator, RegressorMixin):
                    step_type = 'rgr'
                else:
                    run_cleanup()
                    raise RuntimeError('Unsupported estimator type {}'
                                       .format(estimator))
                if step_idx < len(pipe_steps):
                    if step_type != pipe_step_types[step_idx]:
                        run_cleanup()
                        raise RuntimeError(
                            'Different step estimator types: {} {}'
                            .format(step_type, pipe_step_types[step_idx]))
                else:
                    pipe_step_types.append(step_type)
                uniq_step_name = '{}{:d}'.format(step_type, step_idx)
                if 'param_grid' in pipe_config[step_key]:
                    for param, param_values in (
                            pipe_config[step_key]['param_grid'].items()):
                        if isinstance(param_values, (list, tuple, np.ndarray)):
                            if (isinstance(param_values, (list, tuple))
                                    and param_values or np.any(param_values)):
                                uniq_step_param = '{}__{}'.format(
                                    uniq_step_name, param)
                                if len(param_values) > 1:
                                    params[uniq_step_param] = param_values
                                    if uniq_step_param not in param_grid_dict:
                                        param_grid_dict[uniq_step_param] = (
                                            param_values)
                                else:
                                    estimator.set_params(
                                        **{param: param_values[0]})
                        elif param_values is not None:
                            estimator.set_params(**{param: param_values})
                if 'param_routing' in pipe_config[step_key]:
                    if pipe_param_routing is None:
                        pipe_param_routing = {}
                    if uniq_step_name in pipe_param_routing:
                        for param in pipe_config[step_key]['param_routing']:
                            if param not in pipe_param_routing[uniq_step_name]:
                                pipe_param_routing[uniq_step_name] = param
                    else:
                        pipe_param_routing[uniq_step_name] = (
                            pipe_config[step_key]['param_routing'])
                if isinstance(estimator, (CFS, FCBF, ReliefF)):
                    pipe_props['uses_rjava'] = True
                if step_idx == len(pipe_steps):
                    if len(pipe_step_keys[step_idx]) > 1:
                        pipe_steps.append((uniq_step_name, None))
                    else:
                        pipe_steps.append((uniq_step_name, estimator))
                if len(pipe_step_keys[step_idx]) > 1:
                    params[uniq_step_name] = [estimator]
                    if uniq_step_name not in param_grid_dict:
                        param_grid_dict[uniq_step_name] = []
                    if estimator not in param_grid_dict[uniq_step_name]:
                        param_grid_dict[uniq_step_name].append(estimator)
            else:
                uniq_step_name = pipe_step_types[step_idx] + str(step_idx)
                params[uniq_step_name] = [None]
                if uniq_step_name not in param_grid_dict:
                    param_grid_dict[uniq_step_name] = []
                if None not in param_grid_dict[uniq_step_name]:
                    param_grid_dict[uniq_step_name].append(None)
        param_grid.append(params)
    pipe = ExtendedPipeline(pipe_steps, memory=memory,
                            param_routing=pipe_param_routing)
    for param, param_values in param_grid_dict.items():
        if any(isinstance(v, BaseEstimator) for v in param_values):
            param_grid_dict[param] = sorted(
                ['.'.join([type(v).__module__, type(v).__qualname__])
                 if isinstance(v, BaseEstimator) else v for v in param_values],
                key=lambda x: (x is None, x))
    return pipe, pipe_step_names, pipe_props, param_grid, param_grid_dict


def load_dataset(dataset_file):
    dataset_name, file_extension = os.path.splitext(
        os.path.split(dataset_file)[1])
    if os.path.isfile(dataset_file) and file_extension in (
            '.Rda', '.rda', '.RData', '.Rdata', '.Rds', '.rds'):
        if file_extension in ('.Rda', '.rda', '.RData', '.Rdata'):
            r_base.load(dataset_file)
            eset = robjects.globalenv[dataset_name]
        else:
            eset = r_base.readRDS(dataset_file)
    else:
        run_cleanup()
        raise IOError('File does not exist/invalid: {}'
                      .format(dataset_file))
    X = pd.DataFrame(r_base.t(r_biobase.exprs(eset)),
                     columns=r_biobase.featureNames(eset),
                     index=r_biobase.sampleNames(eset))
    sample_meta = r_biobase.pData(eset)
    y = np.array(sample_meta['Class'], dtype=int)
    if 'Group' in sample_meta.columns:
        groups = np.array(sample_meta['Group'], dtype=int)
        _, group_indices, group_counts = np.unique(
            groups, return_inverse=True, return_counts=True)
        sample_weights = (np.max(group_counts) / group_counts)[group_indices]
    else:
        groups = None
        sample_weights = None
    try:
        feature_meta = r_biobase.fData(eset)
    except ValueError:
        feature_meta = pd.DataFrame(index=r_biobase.featureNames(eset))
    if args.sample_meta_cols:
        for sample_meta_col in args.sample_meta_cols:
            if sample_meta_col in sample_meta.columns:
                if sample_meta_col not in X.columns:
                    X[sample_meta_col] = sample_meta[sample_meta_col]
                    feature_meta = feature_meta.append(
                        pd.Series(name=sample_meta_col), verify_integrity=True)
                    feature_meta.loc[sample_meta_col].fillna('', inplace=True)
                else:
                    raise RuntimeError('{} column already exists in X'
                                       .format(sample_meta_col))
            else:
                raise RuntimeError('{} column does not exist in sample_meta'
                                   .format(sample_meta_col))
    col_trf_columns = []
    if args.col_trf_patterns:
        for pattern in args.col_trf_patterns:
            col_trf_columns.append(
                X.columns[X.columns.str.contains(pattern, regex=True)]
                .to_numpy(dtype=str))
    elif args.col_trf_dtypes:
        for dtype in args.col_trf_dtypes:
            if dtype == 'int':
                col_trf_columns.append(X.dtypes.apply(is_integer_dtype)
                                       .to_numpy())
            elif dtype == 'float':
                col_trf_columns.append(X.dtypes.apply(is_float_dtype)
                                       .to_numpy())
            elif dtype == 'category':
                col_trf_columns.append(X.dtypes.apply(
                    lambda d: (is_bool_dtype(d) or is_categorical_dtype(d)
                               or is_object_dtype(d))).to_numpy())
    return (dataset_name, X, y, groups, sample_meta, sample_weights,
            feature_meta, col_trf_columns)


def fit_pipeline(X, y, steps, param_routing, params, fit_params):
    pipe = ExtendedPipeline(steps, memory=memory, param_routing=param_routing)
    pipe.set_params(**params)
    pipe.fit(X, y, **fit_params)
    if args.scv_verbose == 0:
        print('.', end='', flush=True)
    return pipe


def calculate_test_scores(pipe, X_test, y_test, pipe_predict_params,
                          test_sample_weights=None):
    scores = {}
    if hasattr(pipe, 'decision_function'):
        y_score = pipe.decision_function(X_test, **pipe_predict_params)
    else:
        y_score = pipe.predict_proba(X_test, **pipe_predict_params)[:, 1]
    for metric in args.scv_scoring:
        if metric == 'roc_auc':
            scores[metric] = roc_auc_score(
                y_test, y_score, sample_weight=test_sample_weights)
            scores['fpr'], scores['tpr'], _ = roc_curve(
                y_test, y_score, pos_label=1,
                sample_weight=test_sample_weights)
        elif metric == 'balanced_accuracy':
            y_pred = pipe.predict(X_test, **pipe_predict_params)
            scores[metric] = balanced_accuracy_score(
                y_test, y_pred, sample_weight=test_sample_weights)
        elif metric == 'average_precision':
            scores[metric] = average_precision_score(
                y_test, y_score, sample_weight=test_sample_weights)
            scores['pre'], scores['rec'], _ = precision_recall_curve(
                y_test, y_score, pos_label=1,
                sample_weight=test_sample_weights)
            scores['pr_auc'] = auc(scores['rec'], scores['pre'])
    return scores


def get_final_feature_meta(pipe, feature_meta):
    final_feature_meta = None
    for estimator in pipe:
        if isinstance(estimator, ColumnTransformer):
            for _, trf_pipe, trf_columns in estimator.transformers_:
                trf_feature_meta = feature_meta.loc[trf_columns]
                for trf_estimator in trf_pipe:
                    if hasattr(trf_estimator, 'get_support'):
                        trf_feature_meta = trf_feature_meta.loc[
                            trf_estimator.get_support()]
                    elif hasattr(trf_estimator, 'get_feature_names'):
                        trf_new_feature_names = (
                            trf_estimator.get_feature_names(
                                input_features=trf_feature_meta.index.values
                            ).astype(str))
                        trf_feature_meta = pd.DataFrame(
                            np.repeat(trf_feature_meta.values, [
                                np.sum(np.char.startswith(
                                    trf_new_feature_names,
                                    '{}_'.format(feature_name)))
                                for feature_name in trf_feature_meta.index],
                                      axis=0),
                            columns=trf_feature_meta.columns,
                            index=trf_new_feature_names)
                if final_feature_meta is None:
                    final_feature_meta = trf_feature_meta
                else:
                    final_feature_meta = pd.concat(
                        [final_feature_meta, trf_feature_meta], axis=0)
        else:
            if final_feature_meta is None:
                final_feature_meta = feature_meta
            if hasattr(estimator, 'get_support'):
                final_feature_meta = final_feature_meta.loc[
                    estimator.get_support()]
            elif hasattr(estimator, 'get_feature_names'):
                new_feature_names = estimator.get_feature_names(
                    input_features=final_feature_meta.index.values
                ).astype(str)
                final_feature_meta = pd.DataFrame(
                    np.repeat(final_feature_meta.values, [
                        np.sum(np.char.startswith(
                            new_feature_names,
                            '{}_'.format(feature_name)))
                        for feature_name in final_feature_meta.index],
                              axis=0),
                    columns=final_feature_meta.columns,
                    index=new_feature_names)
    final_estimator = pipe[-1]
    feature_weights = explain_weights_df(
        final_estimator, feature_names=final_feature_meta.index.values)
    if feature_weights is None and hasattr(final_estimator, 'estimator_'):
        feature_weights = explain_weights_df(
            final_estimator.estimator_,
            feature_names=final_feature_meta.index.values)
    if feature_weights is not None:
        feature_weights.set_index('feature', inplace=True,
                                  verify_integrity=True)
        feature_weights.columns = map(str.title, feature_weights.columns)
        final_feature_meta = final_feature_meta.join(feature_weights,
                                                     how='inner')
    final_feature_meta.index.rename('Feature', inplace=True)
    return final_feature_meta


def add_param_cv_scores(search, param_grid_dict, param_cv_scores=None):
    if param_cv_scores is None:
        param_cv_scores = {}
    for param, param_values in param_grid_dict.items():
        if len(param_values) == 1:
            continue
        param_cv_values = search.cv_results_['param_{}'.format(param)]
        if any(isinstance(v, BaseEstimator) for v in param_cv_values):
            param_cv_values = np.array(
                ['.'.join([type(v).__module__, type(v).__qualname__])
                 if isinstance(v, BaseEstimator) else v
                 for v in param_cv_values])
        if param not in param_cv_scores:
            param_cv_scores[param] = {}
        for metric in args.scv_scoring:
            if metric not in param_cv_scores[param]:
                param_cv_scores[param][metric] = {'scores': [], 'stdev': []}
            param_metric_scores = param_cv_scores[param][metric]['scores']
            param_metric_stdev = param_cv_scores[param][metric]['stdev']
            if args.param_cv_score_meth == 'best':
                for param_value_idx, param_value in enumerate(param_values):
                    mean_cv_scores = (search.cv_results_
                                      ['mean_test_{}'.format(metric)]
                                      [param_cv_values == param_value])
                    std_cv_scores = (search.cv_results_
                                     ['std_test_{}'.format(metric)]
                                     [param_cv_values == param_value])
                    if param_value_idx < len(param_metric_scores):
                        param_metric_scores[param_value_idx] = np.append(
                            param_metric_scores[param_value_idx],
                            mean_cv_scores[np.argmax(mean_cv_scores)])
                        param_metric_stdev[param_value_idx] = np.append(
                            param_metric_stdev[param_value_idx],
                            std_cv_scores[np.argmax(mean_cv_scores)])
                    else:
                        param_metric_scores.append(np.array(
                            [mean_cv_scores[np.argmax(mean_cv_scores)]]))
                        param_metric_stdev.append(np.array(
                            [std_cv_scores[np.argmax(mean_cv_scores)]]))
            elif args.param_cv_score_meth == 'all':
                for param_value_idx, param_value in enumerate(param_values):
                    for split_idx in range(search.n_splits_):
                        split_scores_cv = (search.cv_results_
                                           ['split{:d}_test_{}'
                                            .format(split_idx, metric)]
                                           [param_cv_values == param_value])
                        if param_value_idx < len(param_metric_scores):
                            param_metric_scores[param_value_idx] = np.append(
                                param_metric_scores[param_value_idx],
                                split_scores_cv)
                        else:
                            param_metric_scores.append(split_scores_cv)
    return param_cv_scores


def plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                          param_cv_scores):
    cv_metric_colors = sns.color_palette('hls', len(args.scv_scoring))
    for param in param_cv_scores:
        mean_cv_scores, std_cv_scores = {}, {}
        for metric in args.scv_scoring:
            param_metric_scores = param_cv_scores[param][metric]['scores']
            param_metric_stdev = param_cv_scores[param][metric]['stdev']
            if any(len(l) > 1 for l in param_metric_scores):
                mean_cv_scores[metric], std_cv_scores[metric] = [], []
                for param_value_scores in param_metric_scores:
                    mean_cv_scores[metric].append(np.mean(param_value_scores))
                    std_cv_scores[metric].append(np.std(param_value_scores))
            else:
                mean_cv_scores[metric] = np.ravel(param_metric_scores)
                std_cv_scores[metric] = np.ravel(param_metric_stdev)
        plt.figure(figsize=(args.fig_width, args.fig_height))
        param_type = re.sub(r'^([a-z]+)\d+', r'\1',
                            '__'.join(param.split('__')[-2:]), count=1)
        if param_type in params_num_xticks:
            x_axis = param_grid_dict[param]
            plt.xticks(x_axis)
        elif param_type in params_fixed_xticks:
            x_axis = range(len(param_grid_dict[param]))
            xtick_labels = [v.split('.')[-1]
                            if param_type in pipeline_step_types
                            and not args.long_label_names
                            and v is not None else str(v)
                            for v in param_grid_dict[param]]
            plt.xticks(x_axis, xtick_labels)
        else:
            raise RuntimeError('No ticks config exists for {}'
                               .format(param_type))
        plt.xlim([min(x_axis), max(x_axis)])
        plt.title('{}\n{}\nEffect of {} on CV Performance Metrics'.format(
            dataset_name, pipe_name, param), fontsize=args.title_font_size)
        plt.xlabel(param, fontsize=args.axis_font_size)
        plt.ylabel('CV Score', fontsize=args.axis_font_size)
        for metric_idx, metric in enumerate(args.scv_scoring):
            plt.plot(x_axis, mean_cv_scores[metric],
                     color=cv_metric_colors[metric_idx], lw=2, alpha=0.8,
                     label='Mean {}'.format(metric_label[metric]))
            plt.fill_between(x_axis,
                             [m - s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             [m + s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             alpha=0.1, color=cv_metric_colors[metric_idx],
                             label=(r'$\pm$ 1 std. dev.'
                                    if metric_idx == len(args.scv_scoring) - 1
                                    else None))
        plt.legend(loc='lower right', fontsize='medium')
        plt.tick_params(labelsize=args.axis_font_size)
        plt.grid(True, alpha=0.3)


def run_model_selection():
    pipe, pipe_step_names, pipe_props, param_grid, param_grid_dict = (
        setup_pipe_and_param_grid(args.pipe_steps))
    (dataset_name, X, y, groups, sample_meta, sample_weights, feature_meta,
     col_trf_columns) = load_dataset(args.train_dataset)
    if (isinstance(pipe[0], ColumnTransformer)
            and args.col_trf_pipe_steps is not None):
        col_trf_name, col_trf_estimator = pipe.steps[0]
        col_trf_pipe_names = []
        col_trf_transformers = []
        col_trf_param_grids = []
        col_trf_param_routing = {}
        for trf_idx, trf_pipe_steps in enumerate(args.col_trf_pipe_steps):
            (trf_pipe, trf_pipe_step_names, trf_pipe_props, trf_param_grid,
             trf_param_grid_dict) = setup_pipe_and_param_grid(trf_pipe_steps)
            col_trf_pipe_names.append('->'.join(trf_pipe_step_names))
            uniq_trf_name = 'trf{:d}'.format(trf_idx)
            col_trf_transformers.append((uniq_trf_name, trf_pipe,
                                         col_trf_columns[trf_idx]))
            if trf_param_grid:
                col_trf_param_grids.append(
                    [{'{}__{}__{}'.format(col_trf_name, uniq_trf_name, k): v
                      for k, v in params.items()}
                     for params in trf_param_grid])
                for param, param_value in trf_param_grid_dict.items():
                    param_grid_dict['{}__{}__{}'.format(
                        col_trf_name, uniq_trf_name, param)] = param_value
            if trf_pipe.param_routing is not None:
                col_trf_param_routing[uniq_trf_name] = list(
                    {v for l in trf_pipe.param_routing.values()
                     for v in l})
            for trf_pipe_prop, trf_pipe_prop_value in trf_pipe_props.items():
                if trf_pipe_prop_value:
                    pipe_props[trf_pipe_prop] = trf_pipe_prop_value
        pipe_step_names[0] = '{}({})'.format(pipe_step_names[0],
                                             ','.join(col_trf_pipe_names))
        if col_trf_param_grids:
            final_estimator_param_grid = param_grid.copy()
            param_grid = []
            for param_grid_combo in product(final_estimator_param_grid,
                                            *col_trf_param_grids):
                param_grid.append({k: v for params in param_grid_combo
                                   for k, v in params.items()})
        col_trf_estimator.set_params(
            param_routing=col_trf_param_routing,
            transformers=col_trf_transformers)
        pipe.param_routing[col_trf_name] = list(
            {v for l in col_trf_param_routing.values() for v in l})
    pipe_name = '->'.join(pipe_step_names)
    search_param_routing = ({'cv': 'groups',
                             'estimator': ['sample_weight'],
                             'scoring': ['sample_weight']}
                            if groups is not None else None)
    if pipe.param_routing:
        if search_param_routing is None:
            search_param_routing = {'estimator': [], 'scoring': []}
        for param in [p for l in pipe.param_routing.values() for p in l]:
            if param not in search_param_routing['estimator']:
                search_param_routing['estimator'].append(param)
                search_param_routing['scoring'].append(param)
    scv_refit = (args.scv_refit if args.test_dataset
                 or not pipe_props['uses_rjava'] else False)
    if groups is None:
        cv_splitter = StratifiedShuffleSplit(
            n_splits=args.scv_splits, test_size=args.scv_size,
            random_state=args.random_seed)
    else:
        cv_splitter = StratifiedGroupShuffleSplit(
            n_splits=args.scv_splits, test_size=args.scv_size,
            random_state=args.random_seed)
    if args.scv_type == 'grid':
        search = ExtendedGridSearchCV(
            pipe, cv=cv_splitter, error_score=0, n_jobs=args.n_jobs,
            param_grid=param_grid, param_routing=search_param_routing,
            refit=scv_refit, return_train_score=False,
            scoring=args.scv_scoring, verbose=args.scv_verbose)
    elif args.scv_type == 'rand':
        search = ExtendedRandomizedSearchCV(
            pipe, cv=cv_splitter, error_score=0, n_iter=args.scv_n_iter,
            n_jobs=args.n_jobs, param_distributions=param_grid,
            param_routing=search_param_routing, refit=scv_refit,
            return_train_score=False, scoring=args.scv_scoring,
            verbose=args.scv_verbose)
    if args.verbose > 0:
        print(search.__repr__(N_CHAR_MAX=10000))
        if param_grid_dict:
            print('Param grid dict:')
            pprint(param_grid_dict)
    if args.verbose > 0 or args.scv_verbose > 0:
        print('Train:' if args.test_dataset else 'Dataset:', dataset_name,
              X.shape, end=' ')
        if col_trf_columns:
            print('(', ' '.join(
                ['{}: {:d}'.format(
                    col_trf_estimator.transformers[i][0],
                    np.sum(c) if _determine_key_type(c) == 'bool' else
                    c.shape[0])
                 for i, c in enumerate(col_trf_columns)]), ')', sep='')
        else:
            print()
    if args.verbose > 0 and groups is not None:
        print('Groups:')
        pprint(groups)
        print('Sample weights:')
        pprint(sample_weights)
    if args.load_only:
        run_cleanup()
        sys.exit()
    # train w/ independent test sets
    if args.test_dataset:
        pipe_fit_params = {}
        if 'sample_meta' in search_param_routing['estimator']:
            pipe_fit_params['sample_meta'] = sample_meta
        if 'feature_meta' in search_param_routing['estimator']:
            pipe_fit_params['feature_meta'] = feature_meta
        if 'sample_weight' in search_param_routing['estimator']:
            pipe_fit_params['sample_weight'] = sample_weights
        search_fit_params = pipe_fit_params.copy()
        if groups is not None:
            search_fit_params['groups'] = groups
        with parallel_backend(args.parallel_backend,
                              inner_max_num_threads=inner_max_num_threads):
            search.fit(X, y, **search_fit_params)
        param_cv_scores = add_param_cv_scores(search, param_grid_dict)
        final_feature_meta = get_final_feature_meta(search.best_estimator_,
                                                    feature_meta)
        if args.verbose > 0:
            print('Train:', dataset_name, end=' ')
            for metric in args.scv_scoring:
                print(' {} (CV): {:.4f}'.format(
                    metric_label[metric], search.cv_results_[
                        'mean_test_{}'.format(metric)][search.best_index_]),
                      end=' ')
            print(' Params:', {
                k: ('.'.join([type(v).__module__, type(v).__qualname__])
                    if isinstance(v, BaseEstimator) else v)
                for k, v in search.best_params_.items()})
            if 'Weight' in final_feature_meta.columns:
                print('Feature Ranking:')
                print(tabulate(final_feature_meta.iloc[
                    (-final_feature_meta['Weight'].abs()).argsort()],
                               floatfmt='.6e', headers='keys'))
            else:
                print('Features:')
                print(tabulate(final_feature_meta, headers='keys'))
        if args.save_model:
            dump(search, '{}/{}_search.pkl'.format(args.out_dir,
                                                   dataset_name))
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)
        # plot top-ranked selected features vs test performance metrics
        if 'Weight' in final_feature_meta.columns:
            _, ax_slr = plt.subplots(figsize=(args.fig_width, args.fig_height))
            ax_slr.set_title(('{}\n{}\nEffect of Number of Top-Ranked Features'
                              'Selected on Test Performance Metrics')
                             .format(dataset_name, pipe_name),
                             fontsize=args.title_font_size)
            ax_slr.set_xlabel('Number of top-ranked features selected',
                              fontsize=args.axis_font_size)
            ax_slr.set_ylabel('Test Score', fontsize=args.axis_font_size)
            x_axis = range(1, final_feature_meta.shape[0] + 1)
            ax_slr.set_xlim([min(x_axis), max(x_axis)])
            ax_slr.set_xticks(x_axis)
        # plot roc and pr curves
        if 'roc_auc' in args.scv_scoring:
            _, ax_roc = plt.subplots(figsize=(args.fig_width, args.fig_height))
            ax_roc.set_title('{}\n{}\nROC Curves'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            ax_roc.set_xlabel('False Positive Rate',
                              fontsize=args.axis_font_size)
            ax_roc.set_ylabel('True Positive Rate',
                              fontsize=args.axis_font_size)
            ax_roc.set_xlim([-0.01, 1.01])
            ax_roc.set_ylim([-0.01, 1.01])
        if 'average_precision' in args.scv_scoring:
            _, ax_pre = plt.subplots(figsize=(args.fig_width, args.fig_height))
            ax_pre.set_title('{}\n{}\nPR Curves'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            ax_pre.set_xlabel('Recall', fontsize=args.axis_font_size)
            ax_pre.set_ylabel('Precision', fontsize=args.axis_font_size)
            ax_pre.set_xlim([-0.01, 1.01])
            ax_pre.set_ylim([-0.01, 1.01])
        test_datasets = natsorted(list(
            set(args.test_dataset) - set(args.train_dataset)))
        test_metric_colors = sns.color_palette(
            'hls', len(test_datasets) * len(args.scv_scoring))
        for test_idx, test_dataset in enumerate(test_datasets):
            (test_dataset_name, X_test, y_test, _, test_sample_meta,
             test_sample_weights, test_feature_meta, test_col_trf_columns) = (
                 load_dataset(test_dataset))
            pipe_predict_params = {}
            if 'sample_meta' in pipe_fit_params:
                pipe_predict_params['sample_meta'] = test_sample_meta
            if 'feature_meta' in pipe_fit_params:
                pipe_predict_params['feature_meta'] = test_feature_meta
            test_scores = calculate_test_scores(
                search, X_test, y_test, pipe_predict_params,
                test_sample_weights=test_sample_weights)
            if args.verbose > 0:
                print('Test:', test_dataset_name, end=' ')
                for metric in args.scv_scoring:
                    print(' {}: {:.4f}'.format(
                        metric_label[metric], test_scores[metric]), end=' ')
                    if metric == 'average_precision':
                        print(' PR AUC: {:.4f}'.format(test_scores['pr_auc']),
                              end=' ')
                print()
            if 'Weight' in final_feature_meta.columns:
                tf_pipe_steps = pipe.steps[:-1]
                tf_pipe_steps.append(('slrc', ColumnSelector()))
                tf_pipe_steps.append(pipe.steps[-1])
                tf_pipe_param_routing = (pipe.param_routing
                                         if pipe.param_routing else {})
                tf_pipe_param_routing['slrc'] = (
                    pipe_config['ColumnSelector']['param_routing'])
                tf_name_sets = []
                for feature_name in final_feature_meta.iloc[
                        (-final_feature_meta['Weight'].abs()).argsort()].index:
                    if tf_name_sets:
                        tf_name_sets.append(tf_name_sets[-1] + [feature_name])
                    else:
                        tf_name_sets.append([feature_name])
                tf_pipes = Parallel(
                    n_jobs=args.n_jobs, backend=args.parallel_backend,
                    verbose=args.scv_verbose)(
                        delayed(fit_pipeline)(
                            X, y, tf_pipe_steps, tf_pipe_param_routing,
                            {**search.best_params_,
                             'slrc__cols': feature_names}, pipe_fit_params)
                        for feature_names in tf_name_sets)
                tf_test_scores = {}
                for tf_pipe in tf_pipes:
                    test_scores = calculate_test_scores(
                        tf_pipe, X_test, y_test, pipe_predict_params,
                        test_sample_weights=test_sample_weights)
                    for metric in args.scv_scoring:
                        if metric in test_scores:
                            if metric not in tf_test_scores:
                                tf_test_scores[metric] = []
                            tf_test_scores[metric].append(test_scores[metric])
                for metric_idx, metric in enumerate(tf_test_scores):
                    ax_slr.plot(x_axis, tf_test_scores[metric], alpha=0.8,
                                color=test_metric_colors[
                                    test_idx + metric_idx], lw=2,
                                label='{} {}'.format(test_dataset_name,
                                                     metric_label[metric]))
                ax_slr.legend(loc='lower right', fontsize='medium')
                ax_slr.tick_params(labelsize=args.axis_font_size)
                ax_slr.grid(True, alpha=0.3)
            if 'roc_auc' in args.scv_scoring:
                ax_roc.plot(test_scores['fpr'], test_scores['tpr'], alpha=0.8,
                            color=test_metric_colors[
                                test_idx * len(args.scv_scoring)], lw=3,
                            label='{} ROC (AUC = {:.4f})'.format(
                                test_dataset_name, test_scores['roc_auc']))
                ax_roc.plot([0, 1], [0, 1], alpha=0.2, color='grey',
                            linestyle='--', lw=3, label=(
                                'Chance' if test_idx == len(test_datasets) - 1
                                else None))
                ax_roc.legend(loc='lower right', fontsize='medium')
                ax_roc.tick_params(labelsize=args.axis_font_size)
                ax_roc.grid(False)
            if 'average_precision' in args.scv_scoring:
                ax_pre.step(test_scores['rec'], test_scores['pre'], alpha=0.8,
                            color=test_metric_colors[
                                test_idx * len(args.scv_scoring)], lw=3,
                            label='{} PR (AUC = {:.4f})'.format(
                                test_dataset_name, test_scores['pr_auc']),
                            where='post')
                ax_pre.legend(loc='lower right', fontsize='medium')
                ax_pre.tick_params(labelsize=args.axis_font_size)
                ax_pre.grid(False)
    # train-test nested cv
    else:
        split_results = []
        param_cv_scores = {}
        if groups is None:
            test_splitter = StratifiedShuffleSplit(
                n_splits=args.test_splits, test_size=args.test_size,
                random_state=args.random_seed)
        else:
            test_splitter = StratifiedGroupShuffleSplit(
                n_splits=args.test_splits, test_size=args.test_size,
                random_state=args.random_seed)
        for split_idx, (train_idxs, test_idxs) in enumerate(
                test_splitter.split(X, y, groups)):
            pipe_fit_params = {}
            if 'sample_meta' in search_param_routing['estimator']:
                pipe_fit_params['sample_meta'] = sample_meta.iloc[train_idxs]
            if 'feature_meta' in search_param_routing['estimator']:
                pipe_fit_params['feature_meta'] = feature_meta
            if 'sample_weight' in search_param_routing['estimator']:
                pipe_fit_params['sample_weight'] = (
                    sample_weights[train_idxs] if sample_weights is not None
                    else None)
            search_fit_params = pipe_fit_params.copy()
            if groups is not None:
                search_fit_params['groups'] = groups[train_idxs]
            with parallel_backend(args.parallel_backend,
                                  inner_max_num_threads=inner_max_num_threads):
                search.fit(X.iloc[train_idxs], y[train_idxs],
                           **search_fit_params)
            if pipe_props['uses_rjava']:
                best_index = np.argmin(
                    search.cv_results_['rank_test_{}'.format(args.scv_refit)])
                best_params = search.cv_results_['params'][best_index]
                best_estimator = Parallel(
                    n_jobs=args.n_jobs, backend=args.parallel_backend,
                    verbose=args.scv_verbose)(
                        delayed(fit_pipeline)(
                            X.iloc[train_idxs], y[train_idxs], pipe.steps,
                            pipe.param_routing, pipe_params, pipe_fit_params)
                        for pipe_params in [best_params])[0]
            else:
                best_index = search.best_index_
                best_params = search.best_params_
                best_estimator = search.best_estimator_
            param_cv_scores = add_param_cv_scores(search, param_grid_dict,
                                                  param_cv_scores)
            final_feature_meta = get_final_feature_meta(best_estimator,
                                                        feature_meta)
            split_scores = {'cv': {}}
            for metric in args.scv_scoring:
                split_scores['cv'][metric] = (search.cv_results_
                                              ['mean_test_{}'.format(metric)]
                                              [best_index])
            test_sample_weights = (sample_weights[test_idxs]
                                   if sample_weights is not None else None)
            pipe_predict_params = {}
            if 'sample_meta' in pipe_fit_params:
                pipe_predict_params['sample_meta'] = (
                    sample_meta.iloc[test_idxs])
            if 'feature_meta' in pipe_fit_params:
                pipe_predict_params['feature_meta'] = feature_meta
            split_scores['te'] = calculate_test_scores(
                best_estimator, X.iloc[test_idxs], y[test_idxs],
                pipe_predict_params, test_sample_weights=test_sample_weights)
            if args.verbose > 0:
                print('Dataset:', dataset_name, ' Split: {:>{width}d}'
                      .format(split_idx + 1,
                              width=len(str(args.test_splits))), end=' ')
                for metric in args.scv_scoring:
                    print(' {} (CV / Test): {:.4f} / {:.4f}'.format(
                        metric_label[metric], split_scores['cv'][metric],
                        split_scores['te'][metric]), end=' ')
                    if metric == 'average_precision':
                        print(' PR AUC Test: {:.4f}'.format(
                            split_scores['te']['pr_auc']), end=' ')
                print(' Params:', {
                    k: ('.'.join([type(v).__module__, type(v).__qualname__])
                        if isinstance(v, BaseEstimator) else v)
                    for k, v in best_params.items()})
            if args.verbose > 1:
                if 'Weight' in final_feature_meta.columns:
                    print('Feature Ranking:')
                    print(tabulate(final_feature_meta.iloc[
                        (-final_feature_meta['Weight'].abs()).argsort()],
                                   floatfmt='.6e', headers='keys'))
                else:
                    print('Features:')
                    print(tabulate(final_feature_meta, headers='keys'))
            split_results.append({
                'model': best_estimator if args.save_model else None,
                'feature_meta': final_feature_meta,
                'scores': split_scores})
            # clear cache (can grow too big if not)
            if args.pipe_memory:
                memory.clear(warn=False)
        if args.save_results:
            dump(split_results, '{}/{}_split_results.pkl'.format(
                args.out_dir, dataset_name))
            dump(param_cv_scores, '{}/{}_param_cv_scores.pkl'.format(
                args.out_dir, dataset_name))
        scores = {'cv': {}, 'te': {}}
        num_features = []
        for split_result in split_results:
            for metric in args.scv_scoring:
                if metric not in scores['cv']:
                    scores['cv'][metric] = []
                    scores['te'][metric] = []
                scores['cv'][metric].append(
                    split_result['scores']['cv'][metric])
                scores['te'][metric].append(
                    split_result['scores']['te'][metric])
                if metric == 'average_precision':
                    if 'pr_auc' not in scores['te']:
                        scores['te']['pr_auc'] = []
                    scores['te']['pr_auc'].append(
                        split_result['scores']['te']['pr_auc'])
            num_features.append(split_result['feature_meta'].shape[0])
        print('Dataset:', dataset_name, X.shape, end=' ')
        for metric in args.scv_scoring:
            print(' Mean {} (CV / Test): {:.4f} / {:.4f}'.format(
                metric_label[metric], np.mean(scores['cv'][metric]),
                np.mean(scores['te'][metric])), end=' ')
            if metric == 'average_precision':
                print(' Mean PR AUC Test: {:.4f}'.format(
                    np.mean(scores['te']['pr_auc'])), end=' ')
        if num_features and pipe_props['has_selector']:
            print(' Mean Features: {:.0f}'.format(np.mean(num_features)))
        else:
            print()
        # feature mean rankings and scores
        feature_weights = None
        feature_scores = {}
        for split_idx, split_result in enumerate(split_results):
            if 'Weight' in split_result['feature_meta'].columns:
                if split_idx == 0:
                    feature_weights = (
                        split_result['feature_meta'][['Weight']].copy())
                else:
                    feature_weights = feature_weights.join(
                        split_result['feature_meta'][['Weight']],
                        how='outer')
                feature_weights.rename(columns={'Weight': split_idx},
                                       inplace=True)
            for metric in args.scv_scoring:
                if split_idx == 0:
                    feature_scores[metric] = pd.DataFrame(
                        split_result['scores']['te'][metric], columns=[metric],
                        index=split_result['feature_meta'].index)
                else:
                    feature_scores[metric] = feature_scores[metric].join(
                        pd.DataFrame(split_result['scores']['te'][metric],
                                     columns=[metric],
                                     index=split_result['feature_meta'].index),
                        how='outer')
                feature_scores[metric].rename(columns={metric: split_idx},
                                              inplace=True)
        feature_mean_meta = None
        feature_mean_meta_floatfmt = ['']
        if feature_weights is not None:
            feature_ranks = feature_weights.abs().rank(
                ascending=False, method='min', na_option='keep')
            feature_ranks.fillna(feature_ranks.shape[0], inplace=True)
            feature_weights.fillna(0, inplace=True)
            feature_mean_meta = feature_meta.reindex(index=feature_ranks.index,
                                                     fill_value='')
            feature_mean_meta_floatfmt.extend([''] * feature_meta.shape[1])
            feature_mean_meta['Mean Weight Rank'] = feature_ranks.mean(axis=1)
            feature_mean_meta['Mean Weight'] = feature_weights.mean(axis=1)
            feature_mean_meta_floatfmt.extend(['.1f', '.6e'])
        for metric in args.scv_scoring:
            if metric in ('roc_auc', 'balanced_accuracy', 'average_precision'):
                feature_scores[metric].fillna(0.5, inplace=True)
            else:
                raise RuntimeError('No feature scores fillna value defined '
                                   'for {}'.format(metric))
            if feature_scores[metric].mean(axis=1).nunique() > 1:
                if feature_mean_meta is None:
                    feature_mean_meta = pd.DataFrame({
                        'Mean Test {}'.format(metric_label[metric]):
                            feature_scores[metric].mean(axis=1)})
                else:
                    feature_mean_meta = feature_mean_meta.join(
                        pd.DataFrame({
                            'Mean Test {}'.format(metric_label[metric]):
                                feature_scores[metric].mean(axis=1)}),
                        how='left')
            feature_mean_meta_floatfmt.append('.4f')
        if args.verbose > 0 and feature_mean_meta is not None:
            print('Overall Feature Ranking:')
            if feature_weights is not None:
                print(tabulate(
                    feature_mean_meta.sort_values(by='Mean Weight Rank'),
                    floatfmt=feature_mean_meta_floatfmt, headers='keys'))
            else:
                print(tabulate(
                    feature_mean_meta.sort_values(by='Mean Test {}'.format(
                        metric_label[args.scv_refit]), ascending=False),
                    floatfmt=feature_mean_meta_floatfmt, headers='keys'))
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)
        # plot roc and pr curves
        if 'roc_auc' in args.scv_scoring:
            sns.set_palette(sns.color_palette('hls', 2))
            plt.figure(figsize=(args.fig_width, args.fig_height))
            plt.title('{}\n{}\nROC Curve'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            plt.xlabel('False Positive Rate', fontsize=args.axis_font_size)
            plt.ylabel('True Positive Rate', fontsize=args.axis_font_size)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)
            for split_result in split_results:
                tprs.append(np.interp(mean_fpr,
                                      split_result['scores']['te']['fpr'],
                                      split_result['scores']['te']['tpr']))
                tprs[-1][0] = 0.0
                plt.plot(split_result['scores']['te']['fpr'],
                         split_result['scores']['te']['tpr'], alpha=0.2,
                         color='darkgrey', lw=1)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_roc_auc = np.mean(scores['te']['roc_auc'])
            std_roc_auc = np.std(scores['te']['roc_auc'])
            mean_num_features = np.mean(num_features)
            std_num_features = np.std(num_features)
            plt.plot(mean_fpr, mean_tpr, lw=3, alpha=0.8, label=(
                r'Test Mean ROC (AUC = {:.4f} $\pm$ {:.2f}, '
                r'Features = {:.0f} $\pm$ {:.0f})').format(
                    mean_roc_auc, std_roc_auc, mean_num_features,
                    std_num_features))
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2,
                             color='grey', label=r'$\pm$ 1 std. dev.')
            plt.plot([0, 1], [0, 1], alpha=0.2, color='grey',
                     linestyle='--', lw=3, label='Chance')
            plt.legend(loc='lower right', fontsize='medium')
            plt.tick_params(labelsize=args.axis_font_size)
            plt.grid(False)
        if 'average_precision' in args.scv_scoring:
            sns.set_palette(sns.color_palette('hls', 10))
            plt.figure(figsize=(args.fig_width, args.fig_height))
            plt.title('{}\n{}\nPR Curve'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            plt.xlabel('Recall', fontsize=args.axis_font_size)
            plt.ylabel('Precision', fontsize=args.axis_font_size)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            pres, scores['te']['pr_auc'] = [], []
            mean_rec = np.linspace(0, 1, 100)
            for split_result in split_results:
                scores['te']['pr_auc'].append(
                    split_result['scores']['te']['pr_auc'])
                pres.append(np.interp(mean_rec,
                                      split_result['scores']['te']['rec'],
                                      split_result['scores']['te']['pre']))
                pres[-1][0] = 1.0
                plt.step(split_result['scores']['te']['rec'],
                         split_result['scores']['te']['pre'], alpha=0.2,
                         color='darkgrey', lw=1, where='post')
            mean_pre = np.mean(pres, axis=0)
            mean_pre[-1] = 0.0
            mean_pr_auc = np.mean(scores['te']['pr_auc'])
            std_pr_auc = np.std(scores['te']['pr_auc'])
            mean_num_features = np.mean(num_features)
            std_num_features = np.std(num_features)
            plt.step(mean_rec, mean_pre, alpha=0.8, lw=3, where='post',
                     label=(r'Test Mean PR (AUC = {:.4f} $\pm$ {:.2f}, '
                            r'Features = {:.0f} $\pm$ {:.0f})').format(
                                mean_pr_auc, std_pr_auc, mean_num_features,
                                std_num_features))
            std_pre = np.std(pres, axis=0)
            pres_upper = np.minimum(mean_pre + std_pre, 1)
            pres_lower = np.maximum(mean_pre - std_pre, 0)
            plt.fill_between(mean_rec, pres_lower, pres_upper, alpha=0.2,
                             color='grey', label=r'$\pm$ 1 std. dev.')
            plt.legend(loc='lower right', fontsize='medium')
            plt.tick_params(labelsize=args.axis_font_size)
            plt.grid(False)


def run_cleanup():
    if args.pipe_memory:
        rmtree(cachedir)
    rmtree(r_base.tempdir()[0])


def shifted_log2(X, shift=1):
    return np.log2(X + shift)


def int_list(arg):
    return list(map(int, arg.split(',')))


def str_list(arg):
    return list(map(str, arg.split(',')))


def str_bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


def dir_path(path):
    if os.path.isdir(path):
        return path
    raise ArgumentTypeError('{} is not a valid path'.format(path))


parser = ArgumentParser()
parser.add_argument('--train-dataset', '--dataset', '--train-eset', '--train',
                    type=str, required=True, help='training dataset')
parser.add_argument('--pipe-steps', type=str_list, nargs='+', required=True,
                    help='Pipeline step names')
parser.add_argument('--col-trf-pipe-steps', type=str_list, nargs='+',
                    action='append',
                    help='ColumnTransformer pipeline step names')
parser.add_argument('--col-trf-patterns', type=str, nargs='+',
                    help='ColumnTransformer column patterns')
parser.add_argument('--col-trf-dtypes', type=str, nargs='+',
                    choices=['category', 'float', 'int'],
                    help='ColumnTransformer column dtypes')
parser.add_argument('--sample-meta-cols', type=str, nargs='+',
                    help='sample metadata columns')
parser.add_argument('--test-dataset', '--test-eset', '--test', type=str,
                    nargs='+', help='test datasets')
parser.add_argument('--col-slr-cols', type=str_list, nargs='+',
                    help='ColumnSelector feature or metadata columns')
parser.add_argument('--col-slr-file', type=str, nargs='+',
                    help='ColumnSelector feature or metadata columns file')
parser.add_argument('--col-slr-meta-col', type=str,
                    help='ColumnSelector feature metadata column name')
parser.add_argument('--vrt-slr-thres', type=float, nargs='+',
                    help='VarianceThreshold threshold')
parser.add_argument('--mui-slr-n', type=int, nargs='+',
                    help='MutualInfoScorer n neighbors')
parser.add_argument('--skb-slr-k', type=int, nargs='+',
                    help='SelectKBest k')
parser.add_argument('--skb-slr-k-min', type=int, default=1,
                    help='SelectKBest k min')
parser.add_argument('--skb-slr-k-max', type=int,
                    help='SelectKBest k max')
parser.add_argument('--skb-slr-k-step', type=int, default=1,
                    help='SelectKBest k step')
parser.add_argument('--de-slr-pv', type=float, nargs='+',
                    help='diff expr slr adj p-value')
parser.add_argument('--de-slr-fc', type=float, nargs='+',
                    help='diff expr slr fold change')
parser.add_argument('--de-slr-mb', type=str_bool, nargs='+',
                    help='diff expr slr model batch')
parser.add_argument('--sfm-slr-thres', type=float, nargs='+',
                    help='SelectFromModel threshold')
parser.add_argument('--sfm-slr-svc-ce', type=int, nargs='+',
                    help='SelectFromModel LinearSVC C exp')
parser.add_argument('--sfm-slr-svc-ce-min', type=int,
                    help='SelectFromModel LinearSVC C exp min')
parser.add_argument('--sfm-slr-svc-ce-max', type=int,
                    help='SelectFromModel LinearSVC C exp max')
parser.add_argument('--sfm-slr-svc-cw', type=str, nargs='+',
                    help='SelectFromModel LinearSVC class weight')
parser.add_argument('--sfm-slr-rf-thres', type=float, nargs='+',
                    help='SelectFromModel rf threshold')
parser.add_argument('--sfm-slr-rf-e', type=int, nargs='+',
                    help='SelectFromModel rf n estimators')
parser.add_argument('--sfm-slr-rf-d', type=str, nargs='+',
                    help='SelectFromModel rf max depth')
parser.add_argument('--sfm-slr-rf-f', type=str, nargs='+',
                    help='SelectFromModel rf max features')
parser.add_argument('--sfm-slr-rf-cw', type=str, nargs='+',
                    help='SelectFromModel rf class weight')
parser.add_argument('--sfm-slr-ext-thres', type=float, nargs='+',
                    help='SelectFromModel ext threshold')
parser.add_argument('--sfm-slr-ext-e', type=int, nargs='+',
                    help='SelectFromModel ext n estimators')
parser.add_argument('--sfm-slr-ext-d', type=str, nargs='+',
                    help='SelectFromModel ext max depth')
parser.add_argument('--sfm-slr-ext-f', type=str, nargs='+',
                    help='SelectFromModel ext max features')
parser.add_argument('--sfm-slr-ext-cw', type=str, nargs='+',
                    help='SelectFromModel ext class weight')
parser.add_argument('--sfm-slr-grb-e', type=int, nargs='+',
                    help='SelectFromModel grb n estimators')
parser.add_argument('--sfm-slr-grb-d', type=int, nargs='+',
                    help='SelectFromModel grb max depth')
parser.add_argument('--sfm-slr-grb-f', type=str, nargs='+',
                    help='SelectFromModel grb max features')
parser.add_argument('--rfe-slr-svc-ce', type=int, nargs='+',
                    help='RFE SVC C exp')
parser.add_argument('--rfe-slr-svc-ce-min', type=int,
                    help='RFE SVC C exp min')
parser.add_argument('--rfe-slr-svc-ce-max', type=int,
                    help='RFE SVC C exp max')
parser.add_argument('--rfe-slr-svc-cw', type=str, nargs='+',
                    help='RFE SVC class weight')
parser.add_argument('--rfe-slr-rf-e', type=int, nargs='+',
                    help='RFE rf n estimators')
parser.add_argument('--rfe-slr-rf-d', type=str, nargs='+',
                    help='RFE rf max depth')
parser.add_argument('--rfe-slr-rf-f', type=str, nargs='+',
                    help='RFE rf max features')
parser.add_argument('--rfe-slr-rf-cw', type=str, nargs='+',
                    help='RFE rf class weight')
parser.add_argument('--rfe-slr-ext-e', type=int, nargs='+',
                    help='RFE ext n estimators')
parser.add_argument('--rfe-slr-ext-d', type=str, nargs='+',
                    help='RFE ext max depth')
parser.add_argument('--rfe-slr-ext-f', type=str, nargs='+',
                    help='RFE ext max features')
parser.add_argument('--rfe-slr-ext-cw', type=str, nargs='+',
                    help='RFE ext class weight')
parser.add_argument('--rfe-slr-grb-e', type=int, nargs='+',
                    help='RFE grb n estimators')
parser.add_argument('--rfe-slr-grb-d', type=int, nargs='+',
                    help='RFE grb max depth')
parser.add_argument('--rfe-slr-grb-f', type=str, nargs='+',
                    help='RFE grb max features')
parser.add_argument('--rfe-slr-step', type=float, nargs='+',
                    help='RFE step')
parser.add_argument('--rfe-slr-tune-step-at', type=int,
                    help='RFE tune step at')
parser.add_argument('--rfe-slr-reducing-step', default=False,
                    action='store_true', help='RFE reducing step')
parser.add_argument('--rfe-slr-verbose', type=int, default=0,
                    help='RFE verbosity')
parser.add_argument('--rlf-slr-n', type=int, nargs='+',
                    help='ReliefF n neighbors')
parser.add_argument('--rlf-slr-s', type=int, nargs='+',
                    help='ReliefF sample size')
parser.add_argument('--mms-trf-feature-range', type=int_list, default=(0, 1),
                    help='MinMaxScaler feature range')
parser.add_argument('--pwr-trf-meth', type=str, nargs='+',
                    choices=['box-cox', 'yeo-johnson'],
                    help='PowerTransformer meth')
parser.add_argument('--de-trf-mb', type=str_bool, nargs='+',
                    help='diff expr trf model batch')
parser.add_argument('--svc-clf-ce', type=int, nargs='+',
                    help='SVC/LinearSVC C exp')
parser.add_argument('--svc-clf-ce-min', type=int,
                    help='SVC/LinearSVC C exp min')
parser.add_argument('--svc-clf-ce-max', type=int,
                    help='SVC/LinearSVC C exp max')
parser.add_argument('--svc-clf-cw', type=str, nargs='+',
                    help='SVC/LinearSVC class weight')
parser.add_argument('--svc-clf-kern', type=str, nargs='+',
                    help='SVC kernel')
parser.add_argument('--svc-clf-deg', type=int, nargs='+',
                    help='SVC poly degree')
parser.add_argument('--svc-clf-g', type=str, nargs='+',
                    help='SVC gamma')
parser.add_argument('--lsvc-clf-max-iter', type=int, default=1000,
                    help='LinearSVC max_iter')
parser.add_argument('--lsvc-clf-tol', type=float, default=1e-2,
                    help='LinearSVC tol')
parser.add_argument('--svc-clf-cache', type=int, default=2000,
                    help='SVC cache size')
parser.add_argument('--knn-clf-k', type=int, nargs='+',
                    help='KNeighborsClassifier neighbors')
parser.add_argument('--knn-clf-w', type=str, nargs='+',
                    help='KNeighborsClassifier weights')
parser.add_argument('--dt-clf-d', type=str, nargs='+',
                    help='DecisionTreeClassifier max depth')
parser.add_argument('--dt-clf-f', type=str, nargs='+',
                    help='DecisionTreeClassifier max features')
parser.add_argument('--dt-clf-cw', type=str, nargs='+',
                    help='DecisionTreeClassifier class weight')
parser.add_argument('--rf-clf-e', type=int, nargs='+',
                    help='RandomForestClassifier n estimators')
parser.add_argument('--rf-clf-d', type=str, nargs='+',
                    help='RandomForestClassifier max depth')
parser.add_argument('--rf-clf-f', type=str, nargs='+',
                    help='RandomForestClassifier max features')
parser.add_argument('--rf-clf-cw', type=str, nargs='+',
                    help='RandomForestClassifier class weight')
parser.add_argument('--ext-clf-e', type=int, nargs='+',
                    help='ExtraTreesClassifier n estimators')
parser.add_argument('--ext-clf-d', type=str, nargs='+',
                    help='ExtraTreesClassifier max depth')
parser.add_argument('--ext-clf-f', type=str, nargs='+',
                    help='ExtraTreesClassifier max features')
parser.add_argument('--ext-clf-cw', type=str, nargs='+',
                    help='ExtraTreesClassifier class weight')
parser.add_argument('--ada-clf-e', type=int, nargs='+',
                    help='AdaBoostClassifier n estimators')
parser.add_argument('--ada-clf-lgr-ce', type=int, nargs='+',
                    help='AdaBoostClassifier LogisticRegression C exp')
parser.add_argument('--ada-clf-lgr-ce-min', type=int, nargs='+',
                    help='AdaBoostClassifier LogisticRegression C exp min')
parser.add_argument('--ada-clf-lgr-ce-max', type=int, nargs='+',
                    help='AdaBoostClassifier LogisticRegression C exp max')
parser.add_argument('--ada-clf-lgr-cw', type=str, nargs='+',
                    help='AdaBoostClassifier LogisticRegression class weight')
parser.add_argument('--grb-clf-e', type=int, nargs='+',
                    help='GradientBoostingClassifier n estimators')
parser.add_argument('--grb-clf-d', type=int, nargs='+',
                    help='GradientBoostingClassifier max depth')
parser.add_argument('--grb-clf-f', type=str, nargs='+',
                    help='GradientBoostingClassifier max features')
parser.add_argument('--mlp-clf-hls', type=str, nargs='+',
                    help='MLPClassifier hidden layer sizes')
parser.add_argument('--mlp-clf-act', type=str, nargs='+',
                    help='MLPClassifier activation function')
parser.add_argument('--mlp-clf-slvr', type=str, nargs='+',
                    help='MLPClassifier solver')
parser.add_argument('--mlp-clf-a', type=float, nargs='+',
                    help='MLPClassifier alpha')
parser.add_argument('--mlp-clf-lr', type=str, nargs='+',
                    help='MLPClassifier learning rate')
parser.add_argument('--sgd-clf-ae', type=int, nargs='+',
                    help='SGDClassifier alpha exp')
parser.add_argument('--sgd-clf-ae-min', type=int,
                    help='SGDClassifier alpha exp min')
parser.add_argument('--sgd-clf-ae-max', type=int,
                    help='SGDClassifier alpha exp max')
parser.add_argument('--sgd-clf-l1r', type=float, nargs='+',
                    help='SGDClassifier l1 ratio')
parser.add_argument('--sgd-clf-l1r-min', type=float,
                    help='SGDClassifier l1 ratio min')
parser.add_argument('--sgd-clf-l1r-max', type=float,
                    help='SGDClassifier l1 ratio max')
parser.add_argument('--sgd-clf-l1r-step', type=float, default=0.05,
                    help='SGDClassifier l1 ratio step')
parser.add_argument('--sgd-clf-cw', type=str, nargs='+',
                    help='SGDClassifier class weight')
parser.add_argument('--sgd-clf-loss', type=str, nargs='+',
                    choices=['hinge', 'log', 'modified_huber', 'squared_hinge',
                             'perceptron', 'squared_loss', 'huber',
                             'epsilon_insensitive',
                             'squared_epsilon_insensitive'],
                    help='SGDClassifier loss')
parser.add_argument('--sgd-clf-penalty', type=str,
                    choices=['l1', 'l2', 'elasticnet'], default='l2',
                    help='SGDClassifier penalty')
parser.add_argument('--sgd-clf-max-iter', type=int, default=1000,
                    help='SGDClassifier max_iter')
parser.add_argument('--edger-prior-count', type=int, default=1,
                    help='edger prior count')
parser.add_argument('--limma-robust', default=False, action='store_true',
                    help='limma robust')
parser.add_argument('--limma-trend', default=False, action='store_true',
                    help='limma trend')
parser.add_argument('--limma-model-dupcor', default=False, action='store_true',
                    help='limma model dupcor')
parser.add_argument('--scv-type', type=str,
                    choices=['grid', 'rand'], default='grid',
                    help='scv type')
parser.add_argument('--scv-splits', type=int, default=10,
                    help='scv splits')
parser.add_argument('--scv-size', type=float, default=0.2,
                    help='scv size')
parser.add_argument('--scv-verbose', type=int,
                    help='scv verbosity')
parser.add_argument('--scv-scoring', type=str, nargs='+',
                    choices=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    default=['roc_auc', 'balanced_accuracy'],
                    help='scv scoring metric')
parser.add_argument('--scv-refit', type=str,
                    choices=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    default='roc_auc',
                    help='scv refit scoring metric')
parser.add_argument('--scv-n-iter', type=int, default=100,
                    help='randomized scv num iterations')
parser.add_argument('--test-splits', type=int, default=10,
                    help='num outer splits')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='outer splits test size')
parser.add_argument('--param-cv-score-meth', type=str,
                    choices=['best', 'all'], default='best',
                    help='param cv scores calculation method')
parser.add_argument('--title-font-size', type=int, default=14,
                    help='figure title font size')
parser.add_argument('--axis-font-size', type=int, default=14,
                    help='figure axis font size')
parser.add_argument('--long-label-names', default=False, action='store_true',
                    help='figure long label names')
parser.add_argument('--fig-width', type=float, default=10,
                    help='figure width')
parser.add_argument('--fig-height', type=float, default=10,
                    help='figure height')
parser.add_argument('--fig-format', type=str, nargs='+',
                    choices=['png', 'pdf', 'svg', 'tif'], default=['png'],
                    help='figure format')
parser.add_argument('--save-figs', default=False, action='store_true',
                    help='save figures')
parser.add_argument('--show-figs', default=False, action='store_true',
                    help='show figures')
parser.add_argument('--save-model', default=False, action='store_true',
                    help='save model')
parser.add_argument('--save-results', default=False, action='store_true',
                    help='save results')
parser.add_argument('--n-jobs', type=int, default=-1,
                    help='num parallel jobs')
parser.add_argument('--parallel-backend', type=str, default='loky',
                    help='joblib parallel backend')
parser.add_argument('--pipe-memory', default=False, action='store_true',
                    help='turn on pipeline memory')
parser.add_argument('--out-dir', type=dir_path, default=os.getcwd(),
                    help='output dir')
parser.add_argument('--tmp-dir', type=dir_path, default=gettempdir(),
                    help='tmp dir')
parser.add_argument('--random-seed', type=int, default=777,
                    help='random state seed')
parser.add_argument('--jvm-heap-size', type=int, default=500,
                    help='rjava jvm heap size')
parser.add_argument('--filter-warnings', type=str, nargs='+',
                    choices=['convergence', 'joblib', 'qda'],
                    help='filter warnings')
parser.add_argument('--verbose', type=int, default=1,
                    help='program verbosity')
parser.add_argument('--load-only', default=False, action='store_true',
                    help='set up model selection and load dataset only')
args = parser.parse_args()

if args.test_size >= 1.0:
    args.test_size = int(args.test_size)
if args.scv_size >= 1.0:
    args.scv_size = int(args.scv_size)
if args.scv_verbose is None:
    args.scv_verbose = args.verbose
if args.filter_warnings:
    if args.parallel_backend == 'multiprocessing':
        if 'convergence' in args.filter_warnings:
            # filter LinearSVC convergence warnings
            warnings.filterwarnings(
                'ignore', category=ConvergenceWarning,
                message='^Liblinear failed to converge',
                module='sklearn.svm._base')
            # filter SGDClassifier convergence warnings
            warnings.filterwarnings(
                'ignore', category=ConvergenceWarning,
                message=('^Maximum number of iteration reached before '
                         'convergence'),
                module='sklearn.linear_model._stochastic_gradient')
        if 'joblib' in args.filter_warnings:
            # filter joblib peristence time warnings
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='^Persisting input arguments took')
        if 'qda' in args.filter_warnings:
            # filter QDA collinearity warnings
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='^Variables are collinear',
                module='sklearn.discriminant_analysis')
    else:
        python_warnings = ([os.environ['PYTHONWARNINGS']]
                           if 'PYTHONWARNINGS' in os.environ else [])
        if 'convergence' in args.filter_warnings:
            python_warnings.append(
                'ignore:Liblinear failed to converge:'
                'UserWarning:sklearn.svm._base')
            python_warnings.append(
                'ignore:Maximum number of iteration reached before '
                'convergence:UserWarning:'
                'sklearn.linear_model._stochastic_gradient')
        if 'joblib' in args.filter_warnings:
            python_warnings.append(
                'ignore:Persisting input arguments took:UserWarning')
        if 'qda' in args.filter_warnings:
            python_warnings.append(
                'ignore:Variables are collinear:'
                'UserWarning:sklearn.discriminant_analysis')
        os.environ['PYTHONWARNINGS'] = ','.join(python_warnings)
inner_max_num_threads = 1 if args.parallel_backend in ('loky') else None

# suppress linux conda qt5 wayland warning
if sys.platform.startswith('linux'):
    os.environ['XDG_SESSION_TYPE'] = 'x11'

r_base = importr('base')
r_biobase = importr('Biobase')
robjects.r('set.seed({:d})'.format(args.random_seed))
robjects.r('options(\'java.parameters\'="-Xmx{:d}m")'
           .format(args.jvm_heap_size))

if args.pipe_memory:
    cachedir = mkdtemp(dir=args.tmp_dir)
    memory = Memory(location=cachedir, verbose=0)
    anova_clf_scorer = CachedANOVAFScorerClassification(memory=memory)
    chi2_scorer = CachedChi2Scorer(memory=memory)
    mui_clf_scorer = CachedMutualInfoScorerClassification(
        memory=memory, random_state=args.random_seed)
    lsvc_clf = CachedLinearSVC(
        max_iter=args.lsvc_clf_max_iter, memory=memory,
        random_state=args.random_seed, tol=args.lsvc_clf_tol)
    sfm_lsvc_clf = CachedLinearSVC(
        dual=False, max_iter=args.lsvc_clf_max_iter, memory=memory,
        penalty='l1', random_state=args.random_seed, tol=args.lsvc_clf_tol)
    rf_clf = CachedRandomForestClassifier(
        memory=memory, random_state=args.random_seed)
    ext_clf = CachedExtraTreesClassifier(
        memory=memory, random_state=args.random_seed)
    grb_clf = CachedGradientBoostingClassifier(
        memory=memory, random_state=args.random_seed)
else:
    memory = None
    anova_clf_scorer = ANOVAFScorerClassification()
    chi2_scorer = Chi2Scorer()
    mui_clf_scorer = MutualInfoScorerClassification(
        random_state=args.random_seed)
    lsvc_clf = LinearSVC(
        max_iter=args.lsvc_clf_max_iter, random_state=args.random_seed,
        tol=args.lsvc_clf_tol)
    sfm_lsvc_clf = LinearSVC(
        dual=False, max_iter=args.lsvc_clf_max_iter, penalty='l1',
        random_state=args.random_seed, tol=args.lsvc_clf_tol)
    rf_clf = RandomForestClassifier(
        random_state=args.random_seed)
    ext_clf = ExtraTreesClassifier(
        random_state=args.random_seed)
    grb_clf = GradientBoostingClassifier(
        random_state=args.random_seed)

pipeline_step_types = ('slr', 'trf', 'clf', 'rgr')
cv_params = {k: v for k, v in vars(args).items()
             if '_' in k and k.split('_')[1] in pipeline_step_types}
if cv_params['col_slr_file']:
    for feature_file in cv_params['col_slr_file']:
        if os.path.isfile(feature_file):
            with open(feature_file) as f:
                feature_names = f.read().splitlines()
            feature_names = [n.strip() for n in feature_names]
            if cv_params['col_slr_cols'] is None:
                cv_params['col_slr_cols'] = []
            cv_params['col_slr_cols'].append(feature_names)
        else:
            run_cleanup()
            raise IOError('File does not exist/invalid: {}'
                          .format(feature_file))
for cv_param, cv_param_values in cv_params.copy().items():
    if cv_param_values is None:
        if cv_param in ('sfm_slr_svc_ce', 'rfe_slr_svc_ce', 'svc_clf_ce',
                        'ada_clf_lgr_ce', 'sgd_clf_ae'):
            cv_params[cv_param[:-1]] = None
        continue
    if cv_param in ('col_slr_cols', 'vrt_slr_thres', 'mui_slr_n', 'skb_slr_k',
                    'de_slr_pv', 'de_slr_fc', 'de_slr_mb', 'sfm_slr_thres',
                    'sfm_slr_rf_thres', 'sfm_slr_rf_e', 'sfm_slr_ext_thres',
                    'sfm_slr_ext_e', 'sfm_slr_grb_e', 'sfm_slr_grb_d',
                    'rfe_slr_rf_e', 'rfe_slr_ext_e', 'rfe_slr_grb_e',
                    'rfe_slr_grb_d', 'rfe_slr_step', 'rlf_slr_n', 'rlf_slr_s',
                    'pwr_trf_meth', 'de_trf_mb', 'svc_clf_kern', 'svc_clf_deg',
                    'svc_clf_g', 'knn_clf_k', 'knn_clf_w', 'rf_clf_e',
                    'ext_clf_e', 'ada_clf_e', 'grb_clf_e', 'grb_clf_d',
                    'mlp_clf_hls', 'mlp_clf_act', 'mlp_clf_slvr', 'mlp_clf_a',
                    'mlp_clf_lr', 'sgd_clf_loss', 'sgd_clf_l1r'):
        cv_params[cv_param] = sorted(cv_param_values)
    elif cv_param == 'skb_slr_k_max':
        if cv_params['skb_slr_k_min'] == 1 and cv_params['skb_slr_k_step'] > 1:
            cv_params['skb_slr_k'] = [1] + list(range(
                0, cv_params['skb_slr_k_max'] + cv_params['skb_slr_k_step'],
                cv_params['skb_slr_k_step']))[1:]
        else:
            cv_params['skb_slr_k'] = list(range(
                cv_params['skb_slr_k_min'],
                cv_params['skb_slr_k_max'] + cv_params['skb_slr_k_step'],
                cv_params['skb_slr_k_step']))
    elif cv_param in ('sfm_slr_svc_ce', 'rfe_slr_svc_ce', 'svc_clf_ce',
                      'ada_clf_lgr_ce', 'sgd_clf_ae'):
        cv_params[cv_param[:-1]] = 10 ** cv_param_values
    elif cv_param in ('sfm_slr_svc_ce_max', 'rfe_slr_svc_ce_max',
                      'svc_clf_ce_max', 'ada_clf_lgr_ce_max',
                      'sgd_clf_ae_max'):
        cv_param = '_'.join(cv_param.split('_')[:-1])
        cv_param_v_min = cv_params['{}_min'.format(cv_param)]
        cv_param_v_max = cv_param_values
        cv_params[cv_param[:-1]] = np.logspace(
            cv_param_v_min, cv_param_v_max,
            cv_param_v_max - cv_param_v_min + 1, base=10)
    elif cv_param == 'sgd_clf_l1r_max':
        cv_params['sgd_clf_l1r'] = np.linspace(
            cv_params['sgd_clf_l1r_min'], cv_params['sgd_clf_l1r_max'],
            int(np.floor(
                (cv_params['sgd_clf_l1r_max'] - cv_params['sgd_clf_l1r_min'])
                / cv_params['sgd_clf_l1r_step'])) + 1)
    elif cv_param in ('sfm_slr_svc_cw', 'sfm_slr_rf_cw', 'sfm_slr_ext_cw',
                      'rfe_slr_svc_cw', 'rfe_slr_rf_cw', 'rfe_slr_ext_cw',
                      'sfm_slr_rf_f', 'sfm_slr_ext_f', 'sfm_slr_grb_f',
                      'rfe_slr_rf_f', 'rfe_slr_ext_f', 'rfe_slr_grb_f',
                      'svc_clf_cw', 'dt_clf_f', 'dt_clf_cw', 'rf_clf_f',
                      'rf_clf_cw', 'ext_clf_f', 'ext_clf_cw', 'ada_clf_lgr_cw',
                      'grb_clf_f', 'sgd_clf_cw'):
        cv_params[cv_param] = sorted([None if v.title() == 'None' else v
                                      for v in cv_param_values],
                                     key=lambda x: (x is None, x))
    elif cv_param in ('sfm_slr_rf_d', 'sfm_slr_ext_d', 'rfe_slr_rf_d',
                      'rfe_slr_ext_d', 'dt_clf_d', 'rf_clf_d', 'ext_clf_d'):
        cv_params[cv_param] = sorted([None if v.title() == 'None' else int(v)
                                      for v in cv_param_values],
                                     key=lambda x: (x is None, x))

pipe_config = {
    # feature selectors
    'ColumnSelector': {
        'estimator': ColumnSelector(meta_col=args.col_slr_meta_col),
        'param_grid': {
            'cols': cv_params['col_slr_cols']},
        'param_routing': ['feature_meta']},
    'VarianceThreshold': {
        'estimator':  VarianceThreshold(),
        'param_grid': {
            'threshold': cv_params['vrt_slr_thres']}},
    'SelectKBest-ANOVAFScorerClassification': {
        'estimator': SelectKBest(anova_clf_scorer),
        'param_grid': {
            'k': cv_params['skb_slr_k']}},
    'SelectKBest-Chi2Scorer': {
        'estimator': SelectKBest(chi2_scorer),
        'param_grid': {
            'k': cv_params['skb_slr_k']}},
    'SelectKBest-MutualInfoScorerClassification': {
        'estimator': SelectKBest(mui_clf_scorer),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'score_func__n_neighbors': cv_params['mui_slr_n']}},
    'SelectFromModel-LinearSVC': {
        'estimator': SelectFromModel(sfm_lsvc_clf),
        'param_grid': {
            'estimator__C': cv_params['sfm_slr_svc_c'],
            'estimator__class_weight': cv_params['sfm_slr_svc_cw'],
            'max_features': cv_params['skb_slr_k'],
            'threshold': cv_params['sfm_slr_thres']},
        'param_routing': ['sample_weight']},
    'SelectFromModel-RandomForestClassifier': {
        'estimator': SelectFromModel(rf_clf),
        'param_grid': {
            'estimator__n_estimators': cv_params['sfm_slr_rf_e'],
            'estimator__max_depth': cv_params['sfm_slr_rf_d'],
            'estimator__max_features': cv_params['sfm_slr_rf_f'],
            'estimator__class_weight': cv_params['sfm_slr_rf_cw'],
            'max_features': cv_params['skb_slr_k'],
            'threshold': cv_params['sfm_slr_thres']},
        'param_routing': ['sample_weight']},
    'SelectFromModel-ExtraTreesClassifier': {
        'estimator': SelectFromModel(ext_clf),
        'param_grid': {
            'estimator__n_estimators': cv_params['sfm_slr_ext_e'],
            'estimator__max_depth': cv_params['sfm_slr_ext_d'],
            'estimator__max_features': cv_params['sfm_slr_ext_f'],
            'estimator__class_weight': cv_params['sfm_slr_ext_cw'],
            'max_features': cv_params['skb_slr_k'],
            'threshold': cv_params['sfm_slr_thres']},
        'param_routing': ['sample_weight']},
    'SelectFromModel-GradientBoostingClassifier': {
        'estimator': SelectFromModel(grb_clf),
        'param_grid': {
            'estimator__n_estimators': cv_params['sfm_slr_grb_e'],
            'estimator__max_depth': cv_params['sfm_slr_grb_d'],
            'estimator__max_features': cv_params['sfm_slr_grb_f'],
            'max_features': cv_params['skb_slr_k'],
            'threshold': cv_params['sfm_slr_thres']},
        'param_routing': ['sample_weight']},
    'RFE-LinearSVC': {
        'estimator': RFE(lsvc_clf, tune_step_at=args.rfe_slr_tune_step_at,
                         reducing_step=args.rfe_slr_reducing_step,
                         verbose=args.rfe_slr_verbose),
        'param_grid': {
            'estimator__C': cv_params['rfe_slr_svc_c'],
            'estimator__class_weight': cv_params['rfe_slr_svc_cw'],
            'step': cv_params['rfe_slr_step'],
            'n_features_to_select': cv_params['skb_slr_k']},
        'param_routing': ['sample_weight']},
    'RFE-RandomForestClassifier': {
        'estimator': RFE(rf_clf, tune_step_at=args.rfe_slr_tune_step_at,
                         reducing_step=args.rfe_slr_reducing_step,
                         verbose=args.rfe_slr_verbose),
        'param_grid': {
            'estimator__n_estimators': cv_params['rfe_slr_rf_e'],
            'estimator__max_depth': cv_params['rfe_slr_rf_d'],
            'estimator__max_features': cv_params['rfe_slr_rf_f'],
            'estimator__class_weight': cv_params['rfe_slr_rf_cw'],
            'step': cv_params['rfe_slr_step'],
            'n_features_to_select': cv_params['skb_slr_k']},
        'param_routing': ['sample_weight']},
    'RFE-ExtraTreesClassifier': {
        'estimator': RFE(ext_clf, tune_step_at=args.rfe_slr_tune_step_at,
                         reducing_step=args.rfe_slr_reducing_step,
                         verbose=args.rfe_slr_verbose),
        'param_grid': {
            'estimator__n_estimators': cv_params['rfe_slr_ext_e'],
            'estimator__max_depth': cv_params['rfe_slr_ext_d'],
            'estimator__max_features': cv_params['rfe_slr_ext_f'],
            'estimator__class_weight': cv_params['rfe_slr_ext_cw'],
            'step': cv_params['rfe_slr_step'],
            'n_features_to_select': cv_params['skb_slr_k']},
        'param_routing': ['sample_weight']},
    'RFE-GradientBoostingClassifier': {
        'estimator': RFE(grb_clf, tune_step_at=args.rfe_slr_tune_step_at,
                         reducing_step=args.rfe_slr_reducing_step,
                         verbose=args.rfe_slr_verbose),
        'param_grid': {
            'estimator__n_estimators': cv_params['rfe_slr_grb_e'],
            'estimator__max_depth': cv_params['rfe_slr_grb_d'],
            'estimator__max_features': cv_params['rfe_slr_grb_f'],
            'step': cv_params['rfe_slr_step'],
            'n_features_to_select': cv_params['skb_slr_k']},
        'param_routing': ['sample_weight']},
    'DESeq2': {
        'estimator': DESeq2(memory=memory),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'sv': cv_params['de_slr_pv'],
            'fc': cv_params['de_slr_fc'],
            'model_batch': cv_params['de_slr_mb']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'EdgeR': {
        'estimator': EdgeR(memory=memory, prior_count=args.edger_prior_count),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'pv': cv_params['de_slr_pv'],
            'fc': cv_params['de_slr_fc'],
            'model_batch': cv_params['de_slr_mb']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'EdgeRFilterByExpr': {
        'estimator': EdgeRFilterByExpr(),
        'param_grid': {
            'model_batch': cv_params['de_slr_mb']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'LimmaVoom': {
        'estimator': LimmaVoom(memory=memory,
                               model_dupcor=args.limma_model_dupcor,
                               prior_count=args.edger_prior_count),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'pv': cv_params['de_slr_pv'],
            'fc': cv_params['de_slr_fc'],
            'model_batch': cv_params['de_slr_mb']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'DreamVoom': {
        'estimator': DreamVoom(memory=memory,
                               prior_count=args.edger_prior_count),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'pv': cv_params['de_slr_pv'],
            'fc': cv_params['de_slr_fc'],
            'model_batch': cv_params['de_slr_mb']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'Limma': {
        'estimator': Limma(memory=memory, robust=args.limma_robust,
                           trend=args.limma_trend),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'pv': cv_params['de_slr_pv'],
            'fc': cv_params['de_slr_fc'],
            'model_batch': cv_params['de_slr_mb']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'FCBF': {
        'estimator': FCBF(memory=memory),
        'param_grid': {
            'k': cv_params['skb_slr_k']}},
    'ReliefF': {
        'estimator': ReliefF(memory=memory),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'n_neighbors': cv_params['rlf_slr_n'],
            'sample_size': cv_params['rlf_slr_s']}},
    'CFS': {
        'estimator': CFS()},
    # transformers
    'ColumnTransformer': {
        'estimator': ExtendedColumnTransformer([], n_jobs=1,
                                               remainder='passthrough')},
    'OneHotEncoder': {
        'estimator':  OneHotEncoder(handle_unknown='ignore', sparse=False)},
    'ShiftedLog2Transformer': {
        'estimator':  FunctionTransformer(shifted_log2, check_inverse=False,
                                          validate=True)},
    'PowerTransformer': {
        'estimator': PowerTransformer(),
        'param_grid': {
            'method': cv_params['pwr_trf_meth']}},
    'MinMaxScaler': {
        'estimator': MinMaxScaler(feature_range=args.mms_trf_feature_range)},
    'RobustScaler': {
        'estimator': RobustScaler()},
    'StandardScaler': {
        'estimator': StandardScaler()},
    'DESeq2RLEVST': {
        'estimator': DESeq2RLEVST(memory=memory),
        'param_grid': {
            'model_batch': cv_params['de_trf_mb']},
        'param_routing': ['sample_meta']},
    'EdgeRTMMLogCPM': {
        'estimator': EdgeRTMMLogCPM(memory=memory,
                                    prior_count=args.edger_prior_count),
        'param_routing': ['sample_meta']},
    'LimmaBatchEffectRemover': {
        'estimator': LimmaBatchEffectRemover(),
        'param_routing': ['sample_meta']},
    # classifiers
    'LinearSVC': {
        'estimator': LinearSVC(max_iter=args.lsvc_clf_max_iter,
                               random_state=args.random_seed,
                               tol=args.lsvc_clf_tol),
        'param_grid': {
            'C': cv_params['svc_clf_c'],
            'class_weight': cv_params['svc_clf_cw']},
        'param_routing': ['sample_weight']},
    'SVC': {
        'estimator': SVC(cache_size=args.svc_clf_cache, gamma='scale',
                         random_state=args.random_seed),
        'param_grid': {
            'C': cv_params['svc_clf_c'],
            'class_weight': cv_params['svc_clf_cw'],
            'kernel': cv_params['svc_clf_kern'],
            'degree': cv_params['svc_clf_deg'],
            'gamma': cv_params['svc_clf_g']},
        'param_routing': ['sample_weight']},
    'KNeighborsClassifier': {
        'estimator': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': cv_params['knn_clf_k'],
            'weights': cv_params['knn_clf_w']},
        'param_routing': ['sample_weight']},
    'DecisionTreeClassifier': {
        'estimator': DecisionTreeClassifier(random_state=args.random_seed),
        'param_grid': {
            'max_depth': cv_params['dt_clf_d'],
            'max_features': cv_params['dt_clf_f'],
            'class_weight': cv_params['dt_clf_cw']},
        'param_routing': ['sample_weight']},
    'RandomForestClassifier': {
        'estimator': RandomForestClassifier(random_state=args.random_seed),
        'param_grid': {
            'n_estimators': cv_params['rf_clf_e'],
            'max_depth': cv_params['rf_clf_d'],
            'max_features': cv_params['rf_clf_f'],
            'class_weight': cv_params['rf_clf_cw']},
        'param_routing': ['sample_weight']},
    'ExtraTreesClassifier': {
        'estimator': ExtraTreesClassifier(random_state=args.random_seed),
        'param_grid': {
            'n_estimators': cv_params['ext_clf_e'],
            'max_depth': cv_params['ext_clf_d'],
            'max_features': cv_params['ext_clf_f'],
            'class_weight': cv_params['ext_clf_cw']},
        'param_routing': ['sample_weight']},
    'AdaBoostClassifier-LogisticRegression': {
        'estimator': AdaBoostClassifier(
            LogisticRegression(random_state=args.random_seed),
            random_state=args.random_seed),
        'param_grid': {
            'base_estimator__C': cv_params['ada_clf_lgr_c'],
            'base_estimator__class_weight': cv_params['ada_clf_lgr_cw'],
            'n_estimators': cv_params['ada_clf_e']},
        'param_routing': ['sample_weight']},
    'GradientBoostingClassifier': {
        'estimator': GradientBoostingClassifier(random_state=args.random_seed),
        'param_grid': {
            'n_estimators': cv_params['grb_clf_e'],
            'max_depth': cv_params['grb_clf_d'],
            'max_features': cv_params['grb_clf_f']},
        'param_routing': ['sample_weight']},
    'GaussianNB': {
        'estimator': GaussianNB(),
        'param_routing': ['sample_weight']},
    'GaussianProcessClassifier': {
        'estimator': GaussianProcessClassifier(random_state=args.random_seed)},
    'LinearDiscriminantAnalysis': {
        'estimator': LinearDiscriminantAnalysis()},
    'QuadraticDiscriminantAnalysis': {
        'estimator': QuadraticDiscriminantAnalysis()},
    'MLPClassifier': {
        'estimator': MLPClassifier(random_state=args.random_seed),
        'param_grid': {
            'hidden_layer_sizes': cv_params['mlp_clf_hls'],
            'activation': cv_params['mlp_clf_act'],
            'solver': cv_params['mlp_clf_slvr'],
            'alpha': cv_params['mlp_clf_a'],
            'learning_rate': cv_params['mlp_clf_lr']}},
    'SGDClassifier': {
        'estimator': SGDClassifier(max_iter=args.sgd_clf_max_iter,
                                   penalty=args.sgd_clf_penalty,
                                   random_state=args.random_seed),
        'param_grid': {
            'alpha': cv_params['sgd_clf_a'],
            'loss': cv_params['sgd_clf_loss'],
            'l1_ratio': cv_params['sgd_clf_l1r'],
            'class_weight': cv_params['sgd_clf_cw']},
        'param_routing': ['sample_weight']}}

params_num_xticks = [
    'slr__k',
    'slr__max_features',
    'slr__score_func__n_neighbors',
    'slr__estimator__n_estimators',
    'slr__step',
    'slr__n_features_to_select',
    'slr__n_neighbors',
    'slr__sample_size',
    'clf__degree',
    'clf__l1_ratio',
    'clf__n_neighbors',
    'clf__n_estimators']
params_fixed_xticks = [
    'slr',
    'slr__cols',
    'slr__alpha',
    'slr__estimator__C',
    'slr__estimator__class_weight',
    'slr__estimator__max_depth',
    'slr__estimator__max_features',
    'slr__fc',
    'slr__model_batch',
    'slr__pv',
    'slr__sv',
    'slr__threshold',
    'trf',
    'trf__method',
    'trf__model_batch',
    'clf',
    'clf__alpha',
    'clf__C',
    'clf__class_weight',
    'clf__kernel',
    'clf__loss',
    'clf__gamma',
    'clf__weights',
    'clf__max_depth',
    'clf__base_estimator__C',
    'clf__base_estimator__class_weight',
    'clf__max_features']
metric_label = {
    'roc_auc': 'ROC AUC',
    'balanced_accuracy': 'BCR',
    'average_precision': 'AVG PRE'}

run_model_selection()
if args.show_figs or args.save_figs:
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
        if args.save_figs:
            for fig_fmt in args.fig_format:
                plt.savefig('{}/Figure_{:d}.{}'.format(args.out_dir, fig_num,
                                                       fig_fmt),
                            bbox_inches='tight', format=fig_fmt)
if args.show_figs:
    plt.show()
run_cleanup()
