#!/usr/bin/env python

import glob
import os
import re
import sys
import warnings
from argparse import ArgumentParser
from itertools import product
from pprint import pprint
from shutil import rmtree
from tempfile import mkdtemp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.rinterface_lib.embedded as r_embedded
r_embedded.set_initoptions(
    ('rpy2', '--quiet', '--no-save', '--max-ppsize=500000'))
import rpy2.robjects as robjects
import seaborn as sns
from joblib import Memory, Parallel, delayed, dump, parallel_backend
from natsort import natsorted
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection.base import SelectorMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    auc, average_precision_score, balanced_accuracy_score,
    precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

numpy2ri.activate()
pandas2ri.activate()

from sklearn_extensions.base import (
    TransformerMixin as ExtendedTransformerMixin)
from sklearn_extensions.ensemble import (
    CachedExtraTreesClassifier, CachedGradientBoostingClassifier,
    CachedRandomForestClassifier)
from sklearn_extensions.feature_selection.base import (
    SelectorMixin as ExtendedSelectorMixin)
from sklearn_extensions.feature_selection import (
    ANOVAFScorerClassification, CachedANOVAFScorerClassification,
    CachedChi2Scorer, CachedMutualInfoScorerClassification, CFS, Chi2Scorer,
    ColumnSelector, DESeq2, DreamVoom, EdgeR, EdgeRFilterByExpr, FCBF, Limma,
    LimmaVoom, MutualInfoScorerClassification, ReliefF, RFE, SelectFromModel,
    SelectKBest, VarianceThreshold)
from sklearn_extensions.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedGroupShuffleSplit)
from sklearn_extensions.pipeline import Pipeline
from sklearn_extensions.preprocessing import (
    DESeq2RLEVST, EdgeRTMMLogCPM, LimmaRemoveBatchEffect)
from sklearn_extensions.svm import CachedLinearSVC


def setup_pipe_and_param_grid():
    pipe_steps = []
    pipe_param_routing = None
    pipe_step_names = []
    pipe_props = {'has_selector': False, 'uses_rjava': False}
    param_grid = []
    param_grid_dict = {}
    pipe_step_keys = []
    pipe_step_types = []
    for step_idx, step_keys in enumerate(args.pipe_steps):
        if any(k in ('None', 'none') for k in step_keys):
            pipe_step_keys.append(
                [k for k in step_keys if k not in ('None', 'none')] + [None])
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
                    raise ValueError('No pipeline config exists for {}'
                                     .format(step_key))
                if isinstance(estimator, (SelectorMixin,
                                          ExtendedSelectorMixin)):
                    step_type = 'slr'
                    pipe_props['has_selector'] = True
                elif isinstance(estimator, (TransformerMixin,
                                            ExtendedTransformerMixin)):
                    step_type = 'trf'
                elif isinstance(estimator, ClassifierMixin):
                    step_type = 'clf'
                elif isinstance(estimator, RegressorMixin):
                    step_type = 'rgr'
                else:
                    run_cleanup()
                    raise ValueError('Unsupported estimator type {}'
                                     .format(estimator))
                if step_idx < len(pipe_steps):
                    if step_type != pipe_step_types[step_idx]:
                        run_cleanup()
                        raise ValueError(
                            'Different step estimator types: {} {}'
                            .format(step_type, pipe_step_types[step_idx]))
                else:
                    pipe_step_types.append(step_type)
                uniq_step_name = step_type + str(step_idx)
                if 'param_grid' in pipe_config[step_key]:
                    for param, param_values in (
                            pipe_config[step_key]['param_grid'].items()):
                        if param_values:
                            uniq_step_param = '{}__{}'.format(uniq_step_name,
                                                              param)
                            if len(param_values) > 1:
                                params[uniq_step_param] = param_values
                                if uniq_step_param not in param_grid_dict:
                                    param_grid_dict[uniq_step_param] = (
                                        param_values)
                            else:
                                estimator.set_params(
                                    **{param: param_values[0]})
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
    pipe = Pipeline(pipe_steps, memory=memory,
                    param_routing=pipe_param_routing)
    pipe_name = '->'.join(pipe_step_names)
    for param, param_values in param_grid_dict.items():
        if any(isinstance(v, BaseEstimator) for v in param_values):
            param_grid_dict[param] = sorted([
                '.'.join([type(o).__module__, type(o).__qualname__])
                for o in param_values])
    return (pipe, pipe_steps, pipe_param_routing, pipe_name, pipe_props,
            param_grid, param_grid_dict)


def load_dataset(file):
    dataset_name, file_extension = os.path.splitext(os.path.split(file)[1])
    if os.path.isfile(file) and file_extension in (
            '.Rda', '.rda', '.RData', '.Rdata', '.Rds', '.rds'):
        if file_extension in ('.Rda', '.rda', '.RData', '.Rdata'):
            r_base.load(file)
            eset = robjects.globalenv[dataset_name]
        else:
            eset = r_base.readRDS(file)
    else:
        run_cleanup()
        raise ValueError('File does not exist/invalid: {}'.format(file))
    X = np.array(r_base.t(r_biobase.exprs(eset)), dtype=(
        int if r_base.typeof(r_biobase.exprs(eset))[0] == 'integer'
        else float))
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
    return (dataset_name, X, y, groups, sample_meta, sample_weights,
            feature_meta)


def fit_pipeline(X, y, steps, param_routing, params, fit_params):
    pipe = Pipeline(steps, memory=memory, param_routing=param_routing)
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
            scores['pre'], scores['rec'], _ = (
                precision_recall_curve(y_test, y_score, pos_label=1,
                                       sample_weight=test_sample_weights))
            scores['pr_auc'] = auc(scores['rec'], scores['pre'])
    return scores


def get_feature_idxs_and_weights(pipe, num_total_features):
    feature_idxs = np.arange(num_total_features)
    for step in pipe.named_steps:
        if hasattr(pipe.named_steps[step], 'get_support'):
            feature_idxs = feature_idxs[
                pipe.named_steps[step].get_support(indices=True)]
    feature_weights = np.zeros_like(feature_idxs, dtype=float)
    final_estimator = pipe.steps[-1][1]
    if hasattr(final_estimator, 'coef_'):
        feature_weights = np.square(final_estimator.coef_[0])
    elif hasattr(final_estimator, 'feature_importances_'):
        feature_weights = final_estimator.feature_importances_
    else:
        for estimator in pipe.named_steps.values():
            if hasattr(estimator, 'estimator_'):
                if hasattr(estimator.estimator_, 'coef_'):
                    feature_weights = np.square(estimator.estimator_.coef_[0])
                elif hasattr(estimator.estimator_, 'feature_importances_'):
                    feature_weights = estimator.estimator_.feature_importances_
            elif hasattr(estimator, 'scores_'):
                feature_weights = estimator.scores_
            elif hasattr(estimator, 'feature_importances_'):
                feature_weights = estimator.feature_importances_
    return feature_idxs, feature_weights


def add_param_cv_scores(search, param_grid_dict, param_cv_scores=None):
    if param_cv_scores is None:
        param_cv_scores = {}
    for param, param_values in param_grid_dict.items():
        if len(param_values) == 1:
            continue
        param_cv_values = np.ma.getdata(
            search.cv_results_['param_{}'.format(param)])
        if any(isinstance(v, BaseEstimator) for v in param_cv_values):
            param_cv_values = np.array([
                '.'.join([type(v).__module__, type(v).__qualname__])
                for v in param_cv_values])
        param_cv_values_sort_idxs = np.where(
            np.array(param_values).reshape(len(param_values), 1)
            == param_cv_values)[1]
        new_shape = (len(param_values), int(len(search.cv_results_['params'])
                                            / len(param_values)))
        if param not in param_cv_scores:
            param_cv_scores[param] = {}
        for metric in args.scv_scoring:
            if metric not in param_cv_scores[param]:
                param_cv_scores[param][metric] = {}
            if args.scv_h_plt_meth == 'best':
                mean_cv_scores = np.transpose(np.reshape(
                    search.cv_results_['mean_test_{}'.format(metric)]
                    [param_cv_values_sort_idxs], new_shape))
                std_cv_scores = np.transpose(np.reshape(
                    search.cv_results_['std_test_{}'.format(metric)]
                    [param_cv_values_sort_idxs], new_shape))
                mean_cv_scores_max_idxs = np.argmax(mean_cv_scores, axis=0)
                mean_cv_scores = (
                    mean_cv_scores[mean_cv_scores_max_idxs,
                                   np.arange(mean_cv_scores.shape[1])])
                std_cv_scores = (
                    std_cv_scores[mean_cv_scores_max_idxs,
                                  np.arange(std_cv_scores.shape[1])])
                if 'scores' in param_cv_scores[param][metric]:
                    param_cv_scores[param][metric]['scores'] = np.vstack(
                        (param_cv_scores[param][metric]['scores'],
                         mean_cv_scores))
                    param_cv_scores[param][metric]['stdev'] = np.vstack(
                        (param_cv_scores[param][metric]['stdev'],
                         std_cv_scores))
                else:
                    param_cv_scores[param][metric]['scores'] = mean_cv_scores
                    param_cv_scores[param][metric]['stdev'] = std_cv_scores
            elif args.scv_h_plt_meth == 'all':
                for split_idx in range(search.n_splits_):
                    split_scores_cv = np.transpose(np.reshape(
                        search.cv_results_
                        ['split{:d}_test_{}'.format(split_idx, metric)]
                        [param_cv_values_sort_idxs], new_shape))
                    if 'scores' in param_cv_scores[param][metric]:
                        param_cv_scores[param][metric]['scores'] = np.vstack(
                            (param_cv_scores[param][metric]['scores'],
                             split_scores_cv))
                    else:
                        param_cv_scores[param][metric]['scores'] = (
                            split_scores_cv)
    return param_cv_scores


def plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                          param_cv_scores):
    sns.set_palette(sns.color_palette('hls', len(args.scv_scoring)))
    for param in param_cv_scores:
        mean_cv_scores, std_cv_scores = {}, {}
        for metric in args.scv_scoring:
            if param_cv_scores[param][metric]['scores'].ndim > 1:
                mean_cv_scores[metric] = np.mean(
                    param_cv_scores[param][metric]['scores'], axis=0)
                std_cv_scores[metric] = np.std(
                    param_cv_scores[param][metric]['scores'], axis=0)
            else:
                mean_cv_scores[metric] = (
                    param_cv_scores[param][metric]['scores'])
                std_cv_scores[metric] = (
                    param_cv_scores[param][metric]['stdev'])
        plt.figure()
        nonuniq_param = re.sub(r'^(\w+)\d+', r'\1', param)
        if nonuniq_param in params_num_xticks:
            x_axis = param_grid_dict[param]
            plt.xticks(x_axis)
        elif nonuniq_param in params_fixed_xticks:
            x_axis = range(len(param_grid_dict[param]))
            xtick_labels = [v.split('.')[-1]
                            if nonuniq_param in pipeline_step_types
                            and not args.long_label_names
                            else str(v) for v in param_grid_dict[param]]
            plt.xticks(x_axis, xtick_labels)
        plt.xlim([min(x_axis), max(x_axis)])
        plt.title('{}\n{}\nEffect of {} on CV Performance Metrics'.format(
            dataset_name, pipe_name, param), fontsize=args.title_font_size)
        plt.xlabel(param, fontsize=args.axis_font_size)
        plt.ylabel('CV Score', fontsize=args.axis_font_size)
        for metric_idx, metric in enumerate(args.scv_scoring):
            plt.plot(x_axis, mean_cv_scores[metric], lw=2, alpha=0.8,
                     label='Mean {}'.format(metric_label[metric]))
            plt.fill_between(x_axis,
                             [m - s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             [m + s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             alpha=0.2, color='grey', label=(
                                 r'$\pm$ 1 std. dev.'
                                 if metric_idx == len(args.scv_scoring) - 1
                                 else None))
        plt.legend(loc='lower right', fontsize='medium')
        plt.tick_params(labelsize=args.axis_font_size)
        plt.grid(True, alpha=0.3)


def run_model_selection():
    (pipe, pipe_steps, pipe_param_routing, pipe_name, pipe_props, param_grid,
     param_grid_dict) = setup_pipe_and_param_grid()
    dataset_name, X, y, groups, sample_meta, sample_weights, feature_meta = (
        load_dataset(args.train_dataset))
    search_param_routing = ({'cv': 'groups',
                             'estimator': ['sample_weight'],
                             'scoring': ['sample_weight']}
                            if groups is not None else None)
    if pipe_param_routing:
        if search_param_routing is None:
            search_param_routing = {'estimator': [], 'scoring': []}
        for param in [p for l in pipe_param_routing.values() for p in l]:
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
        search = GridSearchCV(
            pipe, cv=cv_splitter, error_score=0, iid=False, n_jobs=args.n_jobs,
            param_grid=param_grid, param_routing=search_param_routing,
            refit=scv_refit, return_train_score=False,
            scoring=args.scv_scoring, verbose=args.scv_verbose)
    elif args.scv_type == 'rand':
        search = RandomizedSearchCV(
            pipe, cv=cv_splitter, error_score=0, iid=False,
            n_iter=args.scv_n_iter, n_jobs=args.n_jobs,
            param_distributions=param_grid, param_routing=search_param_routing,
            refit=scv_refit, return_train_score=False,
            scoring=args.scv_scoring, verbose=args.scv_verbose)
    if args.verbose > 0:
        print('{}:'.format(type(search).__name__))
        pprint({k: vars(v) if k == 'estimator' else v
                for k, v in vars(search).items()})
    if args.verbose > 1 and param_grid_dict:
        print('Param grid dict:')
        pprint(param_grid_dict)
    if args.verbose > 0 or args.scv_verbose > 0:
        print('Train:' if args.test_dataset else 'Dataset:', dataset_name,
              X.shape)
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
        search_fit_params = pipe_fit_params
        if groups is not None:
            search_fit_params['groups'] = groups
        with parallel_backend(args.parallel_backend):
            search.fit(X, y, **search_fit_params)
        param_cv_scores = add_param_cv_scores(search, param_grid_dict)
        feature_idxs, feature_weights = get_feature_idxs_and_weights(
            search.best_estimator_, X.shape[1])
        selected_feature_meta = feature_meta.iloc[feature_idxs].copy()
        if np.any(feature_weights):
            selected_feature_meta['Weight'] = feature_weights
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
            if np.any(feature_weights):
                print('Feature Ranking:')
                print(tabulate(selected_feature_meta.sort_values(
                    by='Weight', ascending=False), floatfmt='.8f',
                               headers='keys'))
            else:
                print('Features:')
                print(tabulate(selected_feature_meta, headers='keys'))
        if args.save_model:
            os.makedirs(args.results_dir, mode=0o755, exist_ok=True)
            dump(search, args.results_dir + '/' + dataset_name + '_search.pkl')
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)
        # plot top-ranked selected features vs test performance metrics
        if np.any(feature_weights):
            _, ax_slr = plt.subplots()
            ax_slr.set_title(('{}\n{}\nEffect of Number of Top-Ranked Features'
                              'Selected on Test Performance Metrics')
                             .format(dataset_name, pipe_name),
                             fontsize=args.title_font_size)
            ax_slr.set_xlabel('Number of top-ranked features selected',
                              fontsize=args.axis_font_size)
            ax_slr.set_ylabel('Test Score', fontsize=args.axis_font_size)
            x_axis = range(1, feature_idxs.size + 1)
            ax_slr.set_xlim([min(x_axis), max(x_axis)])
            ax_slr.set_xticks(x_axis)
        # plot roc and pr curves
        if 'roc_auc' in args.scv_scoring:
            _, ax_roc = plt.subplots()
            ax_roc.set_title('{}\n{}\nROC Curves'.format(
                dataset_name, pipe_name), fontsize=args.title_font_size)
            ax_roc.set_xlabel('False Positive Rate',
                              fontsize=args.axis_font_size)
            ax_roc.set_ylabel('True Positive Rate',
                              fontsize=args.axis_font_size)
            ax_roc.set_xlim([-0.01, 1.01])
            ax_roc.set_ylim([-0.01, 1.01])
        if 'average_precision' in args.scv_scoring:
            _, ax_pre = plt.subplots()
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
             test_sample_weights, test_feature_meta) = (
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
            if np.any(feature_weights):
                tf_pipe_steps = pipe_steps[:-1]
                tf_pipe_steps.append(('slrc', ColumnSelector()))
                tf_pipe_steps.append(pipe_steps[-1])
                tf_pipe_param_routing = (pipe_param_routing
                                         if pipe_param_routing else {})
                tf_pipe_param_routing['slrc'] = (
                    pipe_config['ColumnSelector']['param_routing'])
                tf_name_sets = []
                for feature_name in selected_feature_meta.sort_values(
                        by='Weight', ascending=False).index:
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
            search_fit_params = pipe_fit_params
            if groups is not None:
                search_fit_params['groups'] = groups[train_idxs]
            with parallel_backend(args.parallel_backend):
                search.fit(X[train_idxs], y[train_idxs], **search_fit_params)
            if pipe_props['uses_rjava']:
                best_index = np.argmin(
                    search.cv_results_['rank_test_{}'.format(args.scv_refit)])
                best_params = search.cv_results_['params'][best_index]
                best_estimator = Parallel(
                    n_jobs=args.n_jobs, backend=args.parallel_backend,
                    verbose=args.scv_verbose)(
                        delayed(fit_pipeline)(
                            X[train_idxs], y[train_idxs], pipe_steps,
                            pipe_param_routing, pipe_params,
                            pipe_fit_params)
                        for pipe_params in [best_params])[0]
            else:
                best_index = search.best_index_
                best_params = search.best_params_
                best_estimator = search.best_estimator_
            param_cv_scores = add_param_cv_scores(search, param_grid_dict,
                                                  param_cv_scores)
            feature_idxs, feature_weights = get_feature_idxs_and_weights(
                best_estimator, X[train_idxs].shape[1])
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
                best_estimator, X[test_idxs], y[test_idxs],
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
            selected_feature_meta = feature_meta.iloc[feature_idxs].copy()
            if np.any(feature_weights):
                selected_feature_meta['Weight'] = feature_weights
            if args.verbose > 1:
                if np.any(feature_weights):
                    print('Feature Ranking:')
                    print(tabulate(selected_feature_meta.sort_values(
                        by='Weight', ascending=False), floatfmt='.8f',
                                   headers='keys'))
                else:
                    print('Features:')
                    print(tabulate(selected_feature_meta, headers='keys'))
            split_results.append({
                'model': best_estimator if args.save_model else None,
                'feature_idxs': feature_idxs,
                'feature_weights': feature_weights,
                'scores': split_scores})
            # clear cache (can grow too big if not)
            if args.pipe_memory:
                memory.clear(warn=False)
        if args.save_results:
            os.makedirs(args.results_dir, mode=0o755, exist_ok=True)
            dump(split_results, args.results_dir + '/' + dataset_name
                 + '_split_results.pkl')
            dump(param_cv_scores, args.results_dir + '/' + dataset_name
                 + '_param_cv_scores.pkl')
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
            num_features.append(split_result['feature_idxs'].size)
        print('Dataset:', dataset_name, X.shape, end=' ')
        for metric in args.scv_scoring:
            print(' Mean {} (CV / Test): {:.4f} / {:.4f}'.format(
                metric_label[metric], np.mean(scores['cv'][metric]),
                np.mean(scores['te'][metric])), end=' ')
            if metric == 'average_precision':
                print(' PR AUC Test: {:.4f}'.format(
                    np.mean(scores['te']['pr_auc'])), end=' ')
        if num_features and pipe_props['has_selector']:
            print(' Mean Features: {:.0f}'.format(np.mean(num_features)))
        else:
            print()
        # calculate overall feature ranking
        feature_idxs = []
        for split_result in split_results:
            feature_idxs.extend(split_result['feature_idxs'])
        feature_idxs = sorted(list(set(feature_idxs)))
        feature_matrix_idx = {}
        for idx, feature_idx in enumerate(feature_idxs):
            feature_matrix_idx[feature_idx] = idx
        weights_matrix = np.zeros(
            (len(feature_idxs), len(split_results)), dtype=float)
        scores_cv_matrix = {}
        for metric in args.scv_scoring:
            scores_cv_matrix[metric] = np.zeros(
                (len(feature_idxs), len(split_results)), dtype=float)
        for split_idx, split_result in enumerate(split_results):
            for idx, feature_idx in enumerate(split_result['feature_idxs']):
                (weights_matrix[feature_matrix_idx[feature_idx]]
                 [split_idx]) = split_result['feature_weights'][idx]
                for metric in args.scv_scoring:
                    (scores_cv_matrix[metric]
                     [feature_matrix_idx[feature_idx]][split_idx]) = (
                         split_result['scores']['cv'][metric])
        feature_mean_weights, feature_mean_scores = [], []
        for idx in range(len(feature_idxs)):
            feature_mean_weights.append(np.mean(weights_matrix[idx]))
            feature_mean_scores.append(np.mean(
                scores_cv_matrix[args.scv_refit][idx]))
        if args.verbose > 0:
            print('Overall Feature Ranking:')
            selected_feature_meta = feature_meta.iloc[feature_idxs].copy()
            if args.feature_rank_meth == 'weight':
                selected_feature_meta['Mean Weight'] = feature_mean_weights
                print(tabulate(selected_feature_meta.sort_values(
                    by='Mean Weight', ascending=False), floatfmt='.8f',
                               headers='keys'))
            elif args.feature_rank_meth == 'score':
                header = 'Mean {}'.format(metric_label[args.scv_refit])
                selected_feature_meta[header] = feature_mean_scores
                print(tabulate(selected_feature_meta.sort_values(
                    by=header, ascending=False), floatfmt='.4f',
                               headers='keys'))
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)
        # plot roc and pr curves
        if 'roc_auc' in args.scv_scoring:
            sns.set_palette(sns.color_palette('hls', 2))
            plt.figure()
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
            plt.figure()
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
    if glob.glob('/tmp/Rtmp*'):
        for rtmp in glob.glob('/tmp/Rtmp*'):
            rmtree(rtmp)


def int_list(arg):
    return list(map(int, arg.split(',')))


def str_list(arg):
    return list(map(str, arg.split(',')))


parser = ArgumentParser()
parser.add_argument('--train-dataset', '--dataset', '--train-eset', '--train',
                    type=str, required=True, help='training dataset')
parser.add_argument('--pipe-steps', type=str_list, nargs='+', required=True,
                    help='pipeline step names')
parser.add_argument('--test-dataset', '--test-eset', '--test', type=str,
                    nargs='+', help='test datasets')
parser.add_argument('--slr-col-names', type=str_list,
                    nargs='+', help='slr feature or metadata names')
parser.add_argument('--slr-col-meta', type=str,
                    help='slr feature metadata column name')
parser.add_argument('--slr-vrt-thres', type=float,
                    nargs='+', help='slr vrt threshold')
parser.add_argument('--slr-mi-n', type=int, nargs='+',
                    help='slr mi n neighbors')
parser.add_argument('--slr-skb-k', type=int,
                    nargs='+', help='slr skb k')
parser.add_argument('--slr-skb-k-min', type=int,
                    help='slr skb k min')
parser.add_argument('--slr-skb-k-max', type=int,
                    help='slr skb k max')
parser.add_argument('--slr-skb-k-step', type=int, default=1,
                    help='slr skb k step')
parser.add_argument('--slr-de-pv', type=float, nargs='+',
                    help='slr diff expr adj p-value')
parser.add_argument('--slr-de-fc', type=float, nargs='+',
                    help='slr diff expr fold change')
parser.add_argument('--slr-sfm-svm-thres', type=float,
                    nargs='+', help='slr sfm svm threshold')
parser.add_argument('--slr-sfm-svm-c', type=float,
                    nargs='+', help='slr sfm svm c')
parser.add_argument('--slr-sfm-svm-cw', type=str,
                    nargs='+', help='slr sfm svm class weight')
parser.add_argument('--slr-sfm-rf-thres', type=float,
                    nargs='+', help='slr sfm rf threshold')
parser.add_argument('--slr-sfm-rf-e', type=int,
                    nargs='+', help='slr sfm rf n estimators')
parser.add_argument('--slr-sfm-rf-d', type=str,
                    nargs='+', help='slr sfm rf max depth')
parser.add_argument('--slr-sfm-rf-f', type=str,
                    nargs='+', help='slr sfm rf max features')
parser.add_argument('--slr-sfm-rf-cw', type=str,
                    nargs='+', help='slr sfm rf class weight')
parser.add_argument('--slr-sfm-ext-thres', type=float,
                    nargs='+', help='slr sfm ext threshold')
parser.add_argument('--slr-sfm-ext-e', type=int,
                    nargs='+', help='slr sfm ext n estimators')
parser.add_argument('--slr-sfm-ext-d', type=str,
                    nargs='+', help='slr sfm ext max depth')
parser.add_argument('--slr-sfm-ext-f', type=str,
                    nargs='+', help='slr sfm ext max features')
parser.add_argument('--slr-sfm-ext-cw', type=str,
                    nargs='+', help='slr sfm ext class weight')
parser.add_argument('--slr-sfm-grb-e', type=int,
                    nargs='+', help='slr sfm grb n estimators')
parser.add_argument('--slr-sfm-grb-d', type=int,
                    nargs='+', help='slr sfm grb max depth')
parser.add_argument('--slr-sfm-grb-f', type=str,
                    nargs='+', help='slr sfm grb max features')
parser.add_argument('--slr-rfe-svm-c', type=float,
                    nargs='+', help='slr rfe svm c')
parser.add_argument('--slr-rfe-svm-cw', type=str,
                    nargs='+', help='slr rfe svm class weight')
parser.add_argument('--slr-rfe-rf-e', type=int,
                    nargs='+', help='slr rfe rf n estimators')
parser.add_argument('--slr-rfe-rf-d', type=str,
                    nargs='+', help='slr rfe rf max depth')
parser.add_argument('--slr-rfe-rf-f', type=str,
                    nargs='+', help='slr rfe rf max features')
parser.add_argument('--slr-rfe-rf-cw', type=str,
                    nargs='+', help='slr rfe rf class weight')
parser.add_argument('--slr-rfe-ext-e', type=int,
                    nargs='+', help='slr rfe ext n estimators')
parser.add_argument('--slr-rfe-ext-d', type=str,
                    nargs='+', help='slr rfe ext max depth')
parser.add_argument('--slr-rfe-ext-f', type=str,
                    nargs='+', help='slr rfe ext max features')
parser.add_argument('--slr-rfe-ext-cw', type=str,
                    nargs='+', help='slr rfe ext class weight')
parser.add_argument('--slr-rfe-grb-e', type=int,
                    nargs='+', help='slr rfe grb n estimators')
parser.add_argument('--slr-rfe-grb-d', type=int,
                    nargs='+', help='slr rfe grb max depth')
parser.add_argument('--slr-rfe-grb-f', type=str,
                    nargs='+', help='slr rfe grb max features')
parser.add_argument('--slr-rfe-step', type=float,
                    nargs='+', help='slr rfe step')
parser.add_argument('--slr-rfe-tune-step-at', type=int, default=None,
                    help='slr rfe tune step at')
parser.add_argument('--slr-rfe-reducing-step', default=False,
                    action='store_true', help='slr rfe reducing step')
parser.add_argument('--slr-rfe-verbose', type=int, default=0,
                    help='slr rfe verbosity')
parser.add_argument('--slr-rlf-n', type=int, nargs='+',
                    help='slr rlf n neighbors')
parser.add_argument('--slr-rlf-s', type=int, nargs='+',
                    help='slr rlf sample size')
parser.add_argument('--trf-mms-fr', type=int_list,
                    nargs='+', help='trf mms fr')
parser.add_argument('--clf-svm-c', type=float, nargs='+',
                    help='clf svm c')
parser.add_argument('--clf-svm-c-min', type=float,
                    help='clf svm c min')
parser.add_argument('--clf-svm-c-max', type=float,
                    help='clf svm c max')
parser.add_argument('--clf-svm-cw', type=str, nargs='+',
                    help='clf svm class weight')
parser.add_argument('--clf-svm-kern', type=str,
                    nargs='+', help='clf svm kernel')
parser.add_argument('--clf-svm-deg', type=int,
                    nargs='+', help='clf svm poly degree')
parser.add_argument('--clf-svm-g', type=str,
                    nargs='+', help='clf svm gamma')
parser.add_argument('--clf-svm-tol', type=float, default=1e-4,
                    help='clf svm tol')
parser.add_argument('--clf-svm-cache', type=int, default=2000,
                    help='libsvm cache size')
parser.add_argument('--clf-knn-k', type=int, nargs='+',
                    help='clf knn neighbors')
parser.add_argument('--clf-knn-w', type=str,
                    nargs='+', help='clf knn weights')
parser.add_argument('--clf-dt-d', type=str, nargs='+',
                    help='clf dt max depth')
parser.add_argument('--clf-dt-f', type=str, nargs='+',
                    help='clf dt max features')
parser.add_argument('--clf-dt-cw', type=str, nargs='+',
                    help='clf dt class weight')
parser.add_argument('--clf-rf-e', type=int, nargs='+',
                    help='clf rf n estimators')
parser.add_argument('--clf-rf-d', type=str, nargs='+',
                    help='clf rf max depth')
parser.add_argument('--clf-rf-f', type=str, nargs='+',
                    help='clf rf max features')
parser.add_argument('--clf-rf-cw', type=str, nargs='+',
                    help='clf rf class weight')
parser.add_argument('--clf-ext-e', type=int, nargs='+',
                    help='clf ext n estimators')
parser.add_argument('--clf-ext-d', type=str, nargs='+',
                    help='clf ext max depth')
parser.add_argument('--clf-ext-f', type=str, nargs='+',
                    help='clf ext max features')
parser.add_argument('--clf-ext-cw', type=str, nargs='+',
                    help='clf ext class weight')
parser.add_argument('--clf-ada-e', type=int, nargs='+',
                    help='clf ada n estimators')
parser.add_argument('--clf-ada-lgr-c', type=float,
                    nargs='+', help='clf ada lgr c')
parser.add_argument('--clf-ada-lgr-cw', type=str,
                    nargs='+', help='clf ada lgr class weight')
parser.add_argument('--clf-grb-e', type=int, nargs='+',
                    help='clf grb n estimators')
parser.add_argument('--clf-grb-d', type=int, nargs='+',
                    help='clf grb max depth')
parser.add_argument('--clf-grb-f', type=str, nargs='+',
                    help='clf grb max features')
parser.add_argument('--clf-mlp-hls', type=str, nargs='+',
                    help='clf mlp hidden layer sizes')
parser.add_argument('--clf-mlp-act', type=str, nargs='+',
                    help='clf mlp activation function')
parser.add_argument('--clf-mlp-slvr', type=str,
                    nargs='+', help='clf mlp solver')
parser.add_argument('--clf-mlp-a', type=float,
                    nargs='+', help='clf mlp alpha')
parser.add_argument('--clf-mlp-lr', type=str, nargs='+',
                    help='clf mlp learning rate')
parser.add_argument('--clf-sgd-penalty', type=str,
                    help='clf sgd penalty')
parser.add_argument('--clf-sgd-loss', type=str, nargs='+',
                    help='clf sgd loss')
parser.add_argument('--clf-sgd-l1r', type=float, nargs='+',
                    help='clf sgd l1 ratio')
parser.add_argument('--clf-sgd-cw', type=str, nargs='+',
                    help='clf sgd class weight')
parser.add_argument('--edger-prior-count', type=int, default=1,
                    help='edger prior count')
parser.add_argument('--limma-robust', default=False, action='store_true',
                    help='limma robust')
parser.add_argument('--limma-trend', default=False, action='store_true',
                    help='limma trend')
parser.add_argument('--model-batch', default=False, action='store_true',
                    help='model batch')
parser.add_argument('--voom-model-dupcor', default=False, action='store_true',
                    help='limma-voom model dupcor')
parser.add_argument('--scv-type', type=str, choices=['grid', 'rand'],
                    default='grid', help='scv type')
parser.add_argument('--scv-splits', type=int, default=10,
                    help='scv splits')
parser.add_argument('--scv-size', type=float, default=0.2,
                    help='scv size')
parser.add_argument('--scv-verbose', type=int, default=1,
                    help='scv verbosity')
parser.add_argument('--scv-scoring', type=str, nargs='+',
                    choices=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    default=['roc_auc', 'balanced_accuracy'],
                    help='scv scoring metric')
parser.add_argument('--scv-refit', type=str, default='roc_auc',
                    choices=['roc_auc', 'balanced_accuracy',
                             'average_precision'],
                    help='scv refit scoring metric')
parser.add_argument('--scv-n-iter', type=int, default=100,
                    help='randomized scv num iterations')
parser.add_argument('--scv-h-plt-meth', type=str, choices=['best', 'all'],
                    default='best', help='scv hyperparam plot method')
parser.add_argument('--test-splits', type=int, default=10,
                    help='num outer splits')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='outer splits test size')
parser.add_argument('--feature-rank-meth', type=str,
                    choices=['weight', 'score'], default='weight',
                    help='feature rank method')
parser.add_argument('--title-font-size', type=int, default=14,
                    help='figure title font size')
parser.add_argument('--axis-font-size', type=int, default=14,
                    help='figure axis font size')
parser.add_argument('--long-label-names', default=False, action='store_true',
                    help='figure long label names')
parser.add_argument('--save-figs', default=False, action='store_true',
                    help='save figures')
parser.add_argument('--show-figs', default=False, action='store_true',
                    help='show figures')
parser.add_argument('--save-model', default=False, action='store_true',
                    help='save model')
parser.add_argument('--save-results', default=False, action='store_true',
                    help='save results')
parser.add_argument('--results-dir', type=str, default='results',
                    help='results dir')
parser.add_argument('--n-jobs', type=int, default=-1,
                    help='num parallel jobs')
parser.add_argument('--parallel-backend', type=str, default='loky',
                    help='joblib parallel backend')
parser.add_argument('--pipe-memory', default=False, action='store_true',
                    help='turn on pipeline memory')
parser.add_argument('--cache-dir', type=str, default='/tmp',
                    help='cache dir')
parser.add_argument('--random-seed', type=int, default=19825791,
                    help='random state seed')
parser.add_argument('--jvm-heap-size', type=int, default=500,
                    help='rjava jvm heap size')
parser.add_argument('--filter-warnings', type=str, nargs='+',
                    choices=['lsvc', 'qda', 'joblib'],
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
if args.filter_warnings:
    if args.parallel_backend == 'multiprocessing':
        if 'lsvc' in args.filter_warnings:
            # ignore LinearSVC convergence warnings
            warnings.filterwarnings('ignore', category=ConvergenceWarning,
                                    message='^Liblinear failed to converge',
                                    module='sklearn.svm.base')
        if 'qda' in args.filter_warnings:
            # ignore QDA collinearity warnings
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='^Variables are collinear',
                                    module='sklearn.discriminant_analysis')
        if 'joblib' in args.filter_warnings:
            # ignore joblib peristence time warnings
            warnings.filterwarnings('ignore', category=UserWarning,
                                    message='^Persisting input arguments took',
                                    module='sklearn_extensions.pipeline')
    else:
        python_warnings = ([os.environ['PYTHONWARNINGS']]
                           if 'PYTHONWARNINGS' in os.environ else [])
        if 'lsvc' in args.filter_warnings:
            python_warnings.append(
                'ignore:Liblinear failed to converge:'
                'UserWarning:sklearn.svm.base')
        if 'qda' in args.filter_warnings:
            python_warnings.append(
                'ignore:Variables are collinear:'
                'UserWarning:sklearn.discriminant_analysis')
        if 'joblib' in args.filter_warnings:
            python_warnings.append(
                'ignore:Persisting input arguments took:'
                'UserWarning:sklearn_extensions.pipeline')
        os.environ['PYTHONWARNINGS'] = ','.join(python_warnings)

# suppress linux conda qt5 wayland warning
if sys.platform.startswith('linux'):
    os.environ['XDG_SESSION_TYPE'] = 'x11'

r_base = importr('base')
r_biobase = importr('Biobase')
robjects.r('set.seed(' + str(args.random_seed) + ')')
robjects.r('options(\'java.parameters\'="-Xmx' + str(args.jvm_heap_size)
           + 'm")')

if args.pipe_memory:
    cachedir = mkdtemp(dir=args.cache_dir)
    memory = Memory(location=cachedir, verbose=0)
    slr_anova_scorer = CachedANOVAFScorerClassification(memory=memory)
    slr_chi2_scorer = CachedChi2Scorer(memory=memory)
    slr_mi_scorer = CachedMutualInfoScorerClassification(
        memory=memory, random_state=args.random_seed)
    slr_svm_estimator = CachedLinearSVC(
        memory=memory, random_state=args.random_seed, tol=args.clf_svm_tol)
    slr_sfm_svm_estimator = CachedLinearSVC(
        memory=memory, penalty='l1', dual=False,
        random_state=args.random_seed, tol=args.clf_svm_tol)
    slr_rf_estimator = CachedRandomForestClassifier(
        memory=memory, random_state=args.random_seed)
    slr_ext_estimator = CachedExtraTreesClassifier(
        memory=memory, random_state=args.random_seed)
    slr_grb_estimator = CachedGradientBoostingClassifier(
        memory=memory, random_state=args.random_seed)
else:
    memory = None
    slr_anova_scorer = ANOVAFScorerClassification()
    slr_chi2_scorer = Chi2Scorer()
    slr_mi_scorer = MutualInfoScorerClassification(
        random_state=args.random_seed)
    slr_svm_estimator = LinearSVC(
        random_state=args.random_seed, tol=args.clf_svm_tol)
    slr_sfm_svm_estimator = LinearSVC(
        penalty='l1', dual=False, random_state=args.random_seed,
        tol=args.clf_svm_tol)
    slr_rf_estimator = RandomForestClassifier(
        random_state=args.random_seed)
    slr_ext_estimator = ExtraTreesClassifier(
        random_state=args.random_seed)
    slr_grb_estimator = GradientBoostingClassifier(
        random_state=args.random_seed)

# specify params in sort order
# (needed by code dealing with *SearchCV cv_results_)
pipeline_step_types = ('slr', 'trf', 'clf', 'rgr')
cv_params = {k: v for k, v in vars(args).items()
             if k.startswith(pipeline_step_types)}
for cv_param, cv_param_values in cv_params.items():
    if cv_param_values is None:
        continue
    if cv_param == 'trf_mms_fr':
        cv_params[cv_param] = sorted(tuple(v) for v in cv_param_values)
    elif cv_param in ('slr_col_names', 'slr_vrt_thres', 'slr_mi_n',
                      'slr_skb_k', 'slr_de_pv', 'slr_de_fc',
                      'slr_sfm_svm_thres', 'slr_sfm_svm_c', 'slr_sfm_rf_thres',
                      'slr_sfm_rf_e', 'slr_sfm_ext_thres', 'slr_sfm_ext_e',
                      'slr_sfm_grb_e', 'slr_sfm_grb_d', 'slr_rfe_svm_c',
                      'slr_rfe_rf_e', 'slr_rfe_ext_e', 'slr_rfe_grb_e',
                      'slr_rfe_grb_d', 'slr_rfe_step', 'slr_rlf_n',
                      'slr_rlf_s', 'clf_svm_c', 'clf_svm_kern', 'clf_svm_deg',
                      'clf_svm_g', 'clf_knn_k', 'clf_knn_w', 'clf_rf_e',
                      'clf_ext_e', 'clf_ada_e', 'clf_ada_lgr_c', 'clf_grb_e',
                      'clf_grb_d', 'clf_mlp_hls', 'clf_mlp_act',
                      'clf_mlp_slvr', 'clf_mlp_a', 'clf_mlp_lr',
                      'clf_sgd_loss', 'clf_sgd_l1r'):
        cv_params[cv_param] = sorted(cv_param_values)
    elif cv_param in ('slr_skb_k_min', 'slr_skb_k_max'):
        if cv_params['slr_skb_k_min'] == 1 and cv_params['slr_skb_k_step'] > 1:
            cv_params['slr_skb_k'] = [1] + list(range(
                0, cv_params['slr_skb_k_max'] + cv_params['slr_skb_k_step'],
                cv_params['slr_skb_k_step']))[1:]
        else:
            cv_params['slr_skb_k'] = list(range(
                cv_params['slr_skb_k_min'],
                cv_params['slr_skb_k_max'] + cv_params['slr_skb_k_step'],
                cv_params['slr_skb_k_step']))
    elif cv_param in ('clf_svm_c_min', 'clf_svm_c_max'):
        svm_c_log_start = int(np.floor(np.log10(abs(
            cv_params['clf_svm_c_min']))))
        svm_c_log_end = int(np.floor(np.log10(abs(
            cv_params['clf_svm_c_max']))))
        cv_params['clf_svm_c'] = list(np.logspace(
            svm_c_log_start, svm_c_log_end,
            svm_c_log_end - svm_c_log_start + 1))
    elif cv_param in ('slr_sfm_svm_cw', 'slr_sfm_rf_cw', 'slr_sfm_ext_cw',
                      'slr_rfe_svm_cw', 'slr_rfe_rf_cw', 'slr_rfe_ext_cw',
                      'slr_sfm_rf_f', 'slr_sfm_ext_f', 'slr_sfm_grb_f',
                      'slr_rfe_rf_f', 'slr_rfe_ext_f', 'slr_rfe_grb_f',
                      'clf_svm_cw', 'clf_dt_f', 'clf_dt_cw', 'clf_rf_f',
                      'clf_rf_cw', 'clf_ext_f', 'clf_ext_cw', 'clf_ada_lgr_cw',
                      'clf_grb_f', 'clf_sgd_cw'):
        cv_params[cv_param] = sorted(
            [None if v in ('None', 'none') else v
             for v in cv_param_values], key=lambda x: (x is not None, x))
    elif cv_param in ('slr_sfm_rf_d', 'slr_sfm_ext_d', 'slr_rfe_rf_d',
                      'slr_rfe_ext_d', 'clf_dt_d', 'clf_rf_d', 'clf_ext_d'):
        cv_params[cv_param] = sorted(
            [None if v in ('None', 'none') else int(v)
             for v in cv_param_values], key=lambda x: (x is not None, x))

pipe_config = {
    # feature selectors
    'ColumnSelector': {
        'estimator': ColumnSelector(meta_col=args.slr_col_meta),
        'param_grid': {
            'cols': cv_params['slr_col_names']},
        'param_routing': ['feature_meta']},
    'VarianceThreshold': {
        'estimator':  VarianceThreshold(),
        'param_grid': {
            'threshold': cv_params['slr_vrt_thres']}},
    'SelectKBest-ANOVAFScorerClassification': {
        'estimator': SelectKBest(slr_anova_scorer),
        'param_grid': {
            'k': cv_params['slr_skb_k']}},
    'SelectKBest-Chi2Scorer': {
        'estimator': SelectKBest(slr_chi2_scorer),
        'param_grid': {
            'k': cv_params['slr_skb_k']}},
    'SelectKBest-MutualInfoScorerClassification': {
        'estimator': SelectKBest(slr_mi_scorer),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'score_func__n_neighbors': cv_params['slr_mi_n']}},
    'SelectFromModel-LinearSVC': {
        'estimator': SelectFromModel(slr_sfm_svm_estimator),
        'param_grid': {
            'estimator__C': cv_params['slr_sfm_svm_c'],
            'estimator__class_weight': cv_params['slr_sfm_svm_cw'],
            'k': cv_params['slr_skb_k']}},
    'SelectFromModel-RandomForestClassifier': {
        'estimator': SelectFromModel(slr_rf_estimator),
        'param_grid': {
            'estimator__n_estimators': cv_params['slr_sfm_rf_e'],
            'estimator__max_depth': cv_params['slr_sfm_rf_d'],
            'estimator__max_features': cv_params['slr_sfm_rf_f'],
            'estimator__class_weight': cv_params['slr_sfm_rf_cw'],
            'k': cv_params['slr_skb_k']}},
    'SelectFromModel-ExtraTreesClassifier': {
        'estimator': SelectFromModel(slr_ext_estimator),
        'param_grid': {
            'estimator__n_estimators': cv_params['slr_sfm_ext_e'],
            'estimator__max_depth': cv_params['slr_sfm_ext_d'],
            'estimator__max_features': cv_params['slr_sfm_ext_f'],
            'estimator__class_weight': cv_params['slr_sfm_ext_cw'],
            'k': cv_params['slr_skb_k']}},
    'SelectFromModel-GradientBoostingClassifier': {
        'estimator': SelectFromModel(slr_grb_estimator),
        'param_grid': {
            'estimator__n_estimators': cv_params['slr_sfm_grb_e'],
            'estimator__max_depth': cv_params['slr_sfm_grb_d'],
            'estimator__max_features': cv_params['slr_sfm_grb_f'],
            'k': cv_params['slr_skb_k']}},
    'RFE-LinearSVC': {
        'estimator': RFE(slr_svm_estimator,
                         tune_step_at=args.slr_rfe_tune_step_at,
                         reducing_step=args.slr_rfe_reducing_step,
                         verbose=args.slr_rfe_verbose),
        'param_grid': {
            'estimator__C': cv_params['slr_rfe_svm_c'],
            'estimator__class_weight': cv_params['slr_rfe_svm_cw'],
            'step': cv_params['slr_rfe_step'],
            'n_features_to_select': cv_params['slr_skb_k']}},
    'RFE-RandomForestClassifier': {
        'estimator': RFE(slr_rf_estimator,
                         tune_step_at=args.slr_rfe_tune_step_at,
                         reducing_step=args.slr_rfe_reducing_step,
                         verbose=args.slr_rfe_verbose),
        'param_grid': {
            'estimator__n_estimators': cv_params['slr_rfe_rf_e'],
            'estimator__max_depth': cv_params['slr_rfe_rf_d'],
            'estimator__max_features': cv_params['slr_rfe_rf_f'],
            'estimator__class_weight': cv_params['slr_rfe_rf_cw'],
            'step': cv_params['slr_rfe_step'],
            'n_features_to_select': cv_params['slr_skb_k']}},
    'RFE-ExtraTreesClassifier': {
        'estimator': RFE(slr_ext_estimator,
                         tune_step_at=args.slr_rfe_tune_step_at,
                         reducing_step=args.slr_rfe_reducing_step,
                         verbose=args.slr_rfe_verbose),
        'param_grid': {
            'estimator__n_estimators': cv_params['slr_rfe_ext_e'],
            'estimator__max_depth': cv_params['slr_rfe_ext_d'],
            'estimator__max_features': cv_params['slr_rfe_ext_f'],
            'estimator__class_weight': cv_params['slr_rfe_ext_cw'],
            'step': cv_params['slr_rfe_step'],
            'n_features_to_select': cv_params['slr_skb_k']}},
    'RFE-GradientBoostingClassifier': {
        'estimator': RFE(slr_grb_estimator,
                         tune_step_at=args.slr_rfe_tune_step_at,
                         reducing_step=args.slr_rfe_reducing_step,
                         verbose=args.slr_rfe_verbose),
        'param_grid': {
            'estimator__n_estimators': cv_params['slr_rfe_grb_e'],
            'estimator__max_depth': cv_params['slr_rfe_grb_d'],
            'estimator__max_features': cv_params['slr_rfe_grb_f'],
            'step': cv_params['slr_rfe_step'],
            'n_features_to_select': cv_params['slr_skb_k']}},
    'DESeq2': {
        'estimator': DESeq2(memory=memory, model_batch=args.model_batch),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'sv': cv_params['slr_de_pv'],
            'fc': cv_params['slr_de_fc']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'EdgeR': {
        'estimator': EdgeR(memory=memory, prior_count=args.edger_prior_count,
                           model_batch=args.model_batch),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'pv': cv_params['slr_de_pv'],
            'fc': cv_params['slr_de_fc']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'EdgeRFilterByExpr': {
        'estimator': EdgeRFilterByExpr(model_batch=args.model_batch),
        'param_routing': ['sample_meta', 'feature_meta']},
    'LimmaVoom': {
        'estimator': LimmaVoom(memory=memory, model_batch=args.model_batch,
                               model_dupcor=args.voom_model_dupcor,
                               prior_count=args.edger_prior_count),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'pv': cv_params['slr_de_pv'],
            'fc': cv_params['slr_de_fc']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'DreamVoom': {
        'estimator': DreamVoom(memory=memory, model_batch=args.model_batch,
                               prior_count=args.edger_prior_count),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'pv': cv_params['slr_de_pv'],
            'fc': cv_params['slr_de_fc']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'Limma': {
        'estimator': Limma(memory=memory, model_batch=args.model_batch,
                           robust=args.limma_robust, trend=args.limma_trend),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'pv': cv_params['slr_de_pv'],
            'fc': cv_params['slr_de_fc']},
        'param_routing': ['sample_meta', 'feature_meta']},
    'FCBF': {
        'estimator': FCBF(memory=memory),
        'param_grid': {
            'k': cv_params['slr_skb_k']}},
    'ReliefF': {
        'estimator': ReliefF(memory=memory),
        'param_grid': {
            'k': cv_params['slr_skb_k'],
            'n_neighbors': cv_params['slr_rlf_n'],
            'sample_size': cv_params['slr_rlf_s']}},
    'CFS': {
        'estimator': CFS()},
    # transformers
    'MinMaxScaler': {
        'estimator': MinMaxScaler(),
        'param_grid': {
            'feature_range': cv_params['trf_mms_fr']}},
    'StandardScaler': {
        'estimator': StandardScaler()},
    'RobustScaler': {
        'estimator': RobustScaler()},
    'DESeq2RLEVST': {
        'estimator': DESeq2RLEVST(memory=memory, model_batch=args.model_batch),
        'param_routing': ['sample_meta']},
    'EdgeRTMMLogCPM': {
        'estimator': EdgeRTMMLogCPM(memory=memory,
                                    prior_count=args.edger_prior_count),
        'param_routing': ['sample_meta']},
    'LimmaRemoveBatchEffect': {
        'estimator': LimmaRemoveBatchEffect(),
        'param_routing': ['sample_meta']},
    # classifiers
    'LinearSVC': {
        'estimator': LinearSVC(random_state=args.random_seed,
                               tol=args.clf_svm_tol),
        'param_grid': {
            'C': cv_params['clf_svm_c'],
            'class_weight': cv_params['clf_svm_cw']},
        'param_routing': ['sample_weight']},
    'SVC': {
        'estimator': SVC(cache_size=args.clf_svm_cache, gamma='scale',
                         random_state=args.random_seed),
        'param_grid': {
            'C': cv_params['clf_svm_c'],
            'class_weight': cv_params['clf_svm_cw'],
            'kernel': cv_params['clf_svm_kern'],
            'degree': cv_params['clf_svm_deg'],
            'gamma': cv_params['clf_svm_g']},
        'param_routing': ['sample_weight']},
    'KNeighborsClassifier': {
        'estimator': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': cv_params['clf_knn_k'],
            'weights': cv_params['clf_knn_w']}},
    'DecisionTreeClassifier': {
        'estimator': DecisionTreeClassifier(random_state=args.random_seed),
        'param_grid': {
            'max_depth': cv_params['clf_dt_d'],
            'max_features': cv_params['clf_dt_f'],
            'class_weight': cv_params['clf_dt_cw']}},
    'RandomForestClassifier': {
        'estimator': RandomForestClassifier(random_state=args.random_seed),
        'param_grid': {
            'n_estimators': cv_params['clf_rf_e'],
            'max_depth': cv_params['clf_rf_d'],
            'max_features': cv_params['clf_rf_f'],
            'class_weight': cv_params['clf_rf_cw']}},
    'ExtraTreesClassifier': {
        'estimator': ExtraTreesClassifier(random_state=args.random_seed),
        'param_grid': {
            'n_estimators': cv_params['clf_ext_e'],
            'max_depth': cv_params['clf_ext_d'],
            'max_features': cv_params['clf_ext_f'],
            'class_weight': cv_params['clf_ext_cw']}},
    'AdaBoostClassifier-LogisticRegression': {
        'estimator': AdaBoostClassifier(
            LogisticRegression(random_state=args.random_seed),
            random_state=args.random_seed),
        'param_grid': {
            'base_estimator__C': cv_params['clf_ada_lgr_c'],
            'base_estimator__class_weight': cv_params['clf_ada_lgr_cw'],
            'n_estimators': cv_params['clf_ada_e']}},
    'GradientBoostingClassifier': {
        'estimator': GradientBoostingClassifier(random_state=args.random_seed),
        'param_grid': {
            'n_estimators': cv_params['clf_grb_e'],
            'max_depth': cv_params['clf_grb_d'],
            'max_features': cv_params['clf_grb_f']}},
    'GaussianNB': {
        'estimator': GaussianNB()},
    'GaussianProcessClassifier': {
        'estimator': GaussianProcessClassifier(random_state=args.random_seed)},
    'LinearDiscriminantAnalysis': {
        'estimator': LinearDiscriminantAnalysis()},
    'QuadraticDiscriminantAnalysis': {
        'estimator': QuadraticDiscriminantAnalysis()},
    'MLPClassifier': {
        'estimator': MLPClassifier(random_state=args.random_seed),
        'param_grid': {
            'hidden_layer_sizes': cv_params['clf_mlp_hls'],
            'activation': cv_params['clf_mlp_act'],
            'solver': cv_params['clf_mlp_slvr'],
            'alpha': cv_params['clf_mlp_a'],
            'learning_rate': cv_params['clf_mlp_lr']}},
    'SGDClassifier': {
        'estimator': SGDClassifier(penalty=args.clf_sgd_penalty,
                                   random_state=args.random_seed),
        'param_grid': {
            'loss': cv_params['clf_sgd_loss'],
            'l1_ratio': cv_params['clf_sgd_l1r'],
            'class_weight': cv_params['clf_sgd_cw']},
        'param_routing': ['sample_weight']}}

params_num_xticks = [
    'slr__k',
    'slr__score_func__n_neighbors',
    'slr__estimator__n_estimators',
    'slr__n_neighbors',
    'slr__sample_size',
    'slr__n_features_to_select',
    'clf__degree',
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
    'slr__pv',
    'slr__sv',
    'slr__threshold',
    'trf',
    'clf',
    'clf__C',
    'clf__class_weight',
    'clf__kernel',
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
if args.show_figs:
    plt.show()
run_cleanup()
