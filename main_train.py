"""
Example of a script that can be used to train models, similar to what was done in the paper.

"""

import os
from torch import optim
from predicting_alarm_audibility.scheme import *
from tqdm import trange


def main(config):

    # Saving parameters
    run_name = config['run_name']
    save_fd = 'trained_models/main_train'
    model_file_path = f'./{save_fd}/{run_name}.tar'
    if not os.path.exists(f'./{save_fd}'):
        os.makedirs(f'./{save_fd}')

    # Make config
    model, dev_dl, eval_dl, optimizer = make(config)

    # Define evaluation annotators
    pool_a = get_annotator_pool(dev_dl, eval_dl, data_type='eval', pool_id=config['eval_annotators'])
    pool_a_indices = get_annotator_indices(eval_dl, pool_a, data_type='eval')
    # Random draw : 1 annotator per training sample
    drawn_annotators = random_draw(dev_dl, n_annotators=config['n_annotators'])
    drawn_annotator_indices = get_annotator_indices(dev_dl, drawn_annotators, data_type='dev')

    # Initialize progress metrics
    progress_metrics = {'train_acc': 0, 'valid_acc': 0, 'train_loss': 0, 'valid_loss': 0,
                        'best_valid_acc': 0, 'best_epoch': 0}

    # Train model
    with trange(config['epochs'], unit='epoch') as progressbar:
        for epoch in progressbar:
            progressbar.set_description(f"Epoch {epoch}: ")
            progress_metrics = train_log(model, dev_dl, drawn_annotator_indices, optimizer, epoch, progress_metrics,
                                         model_file_path)
            progressbar.set_postfix(train_acc=progress_metrics['train_acc'],
                                    valid_acc=progress_metrics['valid_acc'],
                                    best_epoch=progress_metrics['best_epoch'],
                                    best_valid_acc=progress_metrics['best_valid_acc'])

            for data, target in dev_dl.generate(dev_subset='training', annotator_indices=drawn_annotator_indices):
                train_batch(model, data, target, optimizer)

    # When training done, evaluate model
    best_model = load_weights(model, directory=save_fd, model_name=run_name)
    # ... on training data
    auc_train, f1_train = evaluate(best_model, dev_dl, dev_subset='training', annotator_indices=drawn_annotator_indices)
    print(f"\ttrain_auc: {auc_train:.2f}, train_f1: {f1_train:.2f}")
    # ... on validation data
    auc_valid, f1_valid = evaluate(best_model, dev_dl,
                                   dev_subset='validation',
                                   annotator_indices=drawn_annotator_indices)
    print(f"\tvalid_auc: {auc_valid:.2f}, valid_f1: {f1_valid:.2f}")
    # ... on evaluation data
    auc_eval, f1_eval = evaluate(best_model, eval_dl, annotator_indices=pool_a_indices)
    print(f"\teval_auc: {auc_eval:.2f}, eval_f1: {f1_eval:.2f}")
    print("")
    pass


def make(config):
    dev_ds_path = './features/dev/mel-spectrogram.h5'
    indices_csv_path = './data/annotations/dev/dev_train_valid_split.csv'
    dev_ds = Dataset(dev_ds_path, dev=True, split_indices_csv_path=indices_csv_path)
    dev_dl = DevDataLoader(dev_ds, config['batch_size'])

    eval_ds_path = './features/eval/mel-spectrogram.h5'
    eval_ds = Dataset(eval_ds_path, dev=False, label_type='apf')
    eval_dl = EvalDataLoader(eval_ds, config['batch_size'], training_mean=dev_dl.mean, training_std=dev_dl.std)

    model = CNN(in_channels=1, dropout=config['dropout'], norm_type=config['norm_type'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    return model, dev_dl, eval_dl, optimizer


def train_log(model, loader, annotator_indices, optimizer, epoch, progress_metrics, model_file_path):

    train_acc, train_loss = get_dev_metrics(model, loader, dev_subset='training', annotator_indices=annotator_indices)
    valid_acc, valid_loss = get_dev_metrics(model, loader, dev_subset='validation', annotator_indices=annotator_indices)

    if valid_acc > progress_metrics['best_valid_acc']:
        if os.path.exists(model_file_path):
            os.remove(model_file_path)

        best_valid_acc, best_epoch = valid_acc, epoch
        save_out_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_out_path = model_file_path
        torch.save(save_out_dict, save_out_path)
    else:
        best_valid_acc, best_epoch = progress_metrics['best_valid_acc'], progress_metrics['best_epoch']

    progress_metrics = {'train_acc': train_acc, 'valid_acc': valid_acc,
                        'train_loss': train_loss, 'valid_loss': valid_loss,
                        'best_valid_acc': best_valid_acc, 'best_epoch': best_epoch}

    return progress_metrics


if __name__ == '__main__':

    # Number of runs in the experiment
    nb_runs = 10

    # Run the experiment
    for k in range(nb_runs):
        print(f'*** Training Run {k+1} ***')
        params = {'run_name': f'run_{k+1}',
                  'n_annotators': 1,
                  'eval_annotators': 'pool_a',
                  'batch_size': 16,
                  'dropout': 0.25,
                  'learning_rate': 1e-4,
                  'weight_decay': 1e-4,
                  'norm_type': 5,
                  'epochs': 251}

        main(config=params)
