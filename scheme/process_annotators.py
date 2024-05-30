"""
Functions used to manage the multiple annotations per sample in the dataset, and the different annotator pools involved
in the annotation process for Development and Evaluation data.

"""

import numpy as np
import random


def random_draw(loader, annotator_pool=None, n_annotators=1):
    '''
    Randomly drawing annotators for each sample of a dataset

    :param loader: DevDataLoader object
    :param annotator_pool: List of annotators
    :param n_annotators: Number of annotators per sample
    :return drawn_annotators: Positions of the randomly drawn annotators in the annotation array
    '''

    # If a pool of annotators was given, select annotators among the pool only
    if annotator_pool is not None:
        all_annotators = annotator_pool
    else:
        # else: take all the possible annotators within the dataset
        all_annotators = loader.dataset.annotators

    # Random draw
    drawn_annotators = np.empty([len(loader.dataset), n_annotators])
    for k in range(len(loader.dataset)):
        drawn_annotators[k, :] = np.array(random.sample(tuple(all_annotators), n_annotators))
    drawn_annotators = drawn_annotators.astype(int)

    return drawn_annotators


def get_annotator_pool(dev_loader, eval_loader, data_type, pool_id):

    # Look for common and uncommon annotators
    dev_annotators = dev_loader.dataset.annotators
    eval_annotators = eval_loader.dataset.annotators
    pool_a = eval_annotators[~np.isin(eval_annotators, dev_annotators)]
    pool_b = eval_annotators[np.isin(eval_annotators, dev_annotators)]
    pool_c = dev_annotators[~np.isin(dev_annotators, eval_annotators)]

    # Return the annotator pool
    if data_type == 'dev' and pool_id in ['pool_b', 'pool_c']:
        return eval(pool_id)
    elif data_type == 'eval' and pool_id in ['pool_a', 'pool_b']:
        return eval(pool_id)
    else:
        raise ValueError("Incorrect data_type or pool_id. Variable data_type must either be dev' or 'eval'. "
                         "In case data_type is 'dev', variable pool_id must either be 'pool_b' or 'pool_c. "
                         "In case data_type is 'eval', variable pool_id must either be 'pool_a', or 'pool_b'.")


def get_annotator_indices(loader, selected_annotators, data_type):
    '''
    Transforming the annotator IDs into their respective indices within the list of annotators

    :param loader: DevDataLoader or EvalDataLoader object
    :param selected_annotators: List of the selected annotators represented by their integer IDs
    :param data_type: 'dev' or 'eval'
    :return indices: Indices of the selected annotators within the list of annotators in the dataset
    '''

    if data_type == 'dev':
        # For 'dev' data, annotators are selected for each sample. We operate a simple conversion from IDs to indices.
        find_index = np.vectorize(lambda x: list(loader.dataset.annotators).index(x))
        indices = find_index(selected_annotators)
    elif data_type == 'eval':
        # For 'eval' data, the annotators are selected once for the whole subset (all evaluation samples are annotated
        # by the same annotators). We convert IDs to indices and return the indices for each sample.
        indices = np.repeat([(loader.dataset.annotators[:, None] == selected_annotators).argmax(axis=0).tolist()],
                            len(loader.dataset), axis=0)
    else:
        raise ValueError("Variable data_type must either be 'dev' or 'eval'.")

    return indices
