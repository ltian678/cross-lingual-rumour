import numpy as np
import pandas as pd
import argparse
import logging
import yaml
import time
import datetime
import json
import os
import glob
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return load_from_config_dict(config)

def save_ckp(epoch, model, optimizer, scheduler, model_dir, chkpt_name):
    checkpoint_fpath = str(model_dir / chkpt_name)
    logging.info(f"Saving to checkpoint {checkpoint_fpath}")
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_fpath)


def load_ckp(checkpoint_fpath, model, optimizer=None, scheduler=None, reset_optimizer=False):
    logging.info(f"Loading from checkpoint {checkpoint_fpath}")
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    epoch = 0
    if not reset_optimizer:
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = int(checkpoint_fpath.split('epoch')[1].split('.')[0])
    return epoch



def save_and_report_results(args, results, logger):
    logger.info("\t*** Final Results ***")
    for res, values in results.items():
        logger.info("\n{}:\t{}".format(res, values))
    savepath = os.path.join(args.logdir, 'results.pkl')
    logger.info('Saving results at {}'.format(savepath))
    txt_savepath = os.path.join(args.logdir, 'results.txt')

    args_savepath = os.path.join(args.logdir, 'args.json')
    with open(args_savepath, 'w') as f:
        args.device = -1
        json.dump(vars(args), f)
    logger.info('Saved report at {}'.format(txt_savepath))
    return



def summarize_results(basefolder, from_dataset, target_dataset):
    savefile = os.path.join(basefolder, 'all_res.txt')
    with open(savefile, 'w') as f:
        all_teacher_perf = []
        all_student_perf = []
        all_teacher_dev_perf = []
        all_student_dev_perf = []
        best_students = []
        best_teachers = []
        student_iters = []
        teacher_iters = []
        best_students_acc = []
        best_teachers_acc = []
        for seedfolder in seedfolders:
            print(seedfolder)
            resfolders = glob.glob(seedfolder + '/*')
            if len(resfolders) > 1:
                print('found more than 1 folders.. multiple experiments..')
                print(resfolders)
                return "multiple folders"

            resfolder = resfolders[0]
            res = get_results(resfolder)
            all_teacher_perf.append(res['teacher_test_perf'])
            all_student_perf.append(res['student_test_perf'])
            all_teacher_perf.append(res['teacher_test_perf'])
            all_student_perf.append(res['student_test_perf'])
            all_teacher_dev_perf.append(res['teacher_dev_perf'])
            all_student_dev_perf.append(res['student_dev_perf'])
            metric = res['metric']

            print("results for {}".format(seedfolder.split('/')[-1]))
            best_student_iter = np.argmax(res['student_dev_perf'])
            best_teacher_iter = np.argmax(res['teacher_dev_perf'])
            best_student = res['student_test_perf'][best_student_iter]
            best_teacher = res['teacher_test_perf'][best_teacher_iter]
            print("Best Student: {:.2f} (iter={})".format(best_student, best_student_iter))
            print("Best Teacher: {:.2f} (iter={})".format(best_teacher, best_teacher_iter))
            best_students.append(best_student)
            best_teachers.append(best_teacher)
            best_students_acc.append(res['student_test_acc'][best_student_iter])
            best_teachers_acc.append(res['teacher_test_acc'][best_teacher_iter])
            student_iters.append(best_student_iter)
            teacher_iters.append(best_teacher_iter)

        avg_teacher_perf = np.average(all_teacher_perf, axis=0)
        std_teacher_perf = np.std(all_teacher_perf, axis=0)
        avg_student_perf = np.average(all_student_perf, axis=0)
        std_student_perf = np.std(all_student_perf, axis=0)

        avg_teacher_dev_perf = np.average(all_teacher_dev_perf, axis=0)
        avg_student_dev_perf = np.average(all_student_dev_perf, axis=0)

        num_iter = len(avg_student_perf)

        best_student_iter = np.argmax(avg_student_dev_perf)
        best_teacher_iter = np.argmax(avg_teacher_dev_perf)

    return {
        'student_perf': avg_student_perf[best_student_iter],
        'student_std': std_student_perf[best_student_iter],
        'teacher_perf': avg_teacher_perf[best_teacher_iter],
        'teacher_std': std_teacher_perf[best_teacher_iter],
    }

def plot_res(dataset,avg_teacher_perf, std_teacher_perf, avg_student_perf, std_student_perf):
    plt.figure()
    plt.title(dataset)
    if len(avg_teacher_perf) > 1:
        plt.errorbar(iters, avg_teacher_perf, std_teacher_perf, linestyle='-', marker='^', label='teacher')
        plt.fill_between(iters, avg_teacher_perf - std_teacher_perf, avg_teacher_perf + std_teacher_perf,
                         facecolor='#F0F8FF', edgecolor='#8F94CC', alpha=1.0)
    plt.errorbar(iters, avg_student_perf, std_student_perf, linestyle='-', marker='^', label='student')
    plt.fill_between(iters, avg_student_perf - std_student_perf, avg_student_perf + std_student_perf,
                     facecolor='#F0F8FF', edgecolor='#bc5a45', alpha=1.0)
    plt.legend(loc='lower right')
    plt.xlabel('iter')
    plt.ylabel(metric)
    plt.show()
