import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertTokenizer, BertModel

from rumourClassifier import *
from transformer import *
from utils import *
from logger import *



logger = logging.getLogger(__name__)



#train
def run_rumour(args, logger):
    """
        Self-training for rumour detection
        Leverages labeled, unlabeled data and filtering rules for training a neural network
    """

    teacher_dev_res_list = []
    teacher_test_res_list = []
    teacher_train_res_list = []
    dev_res_list = []
    test_res_list = []
    train_res_list = []
    results = {}

    student_pred_list = []

    logger.info("building student: {}".format(args.student_name))
    student = Student(args)

    logger.info("building teacher")
    teacher = Teacher(args)

    logger.info("loading data")
    teacher_train_dataloader, teacher_validation_dataloader, teacher_prediction_dataloader = teacher.init_dataloader(args.teacher_train, args.teacher_test)
    student_train_dataloader, student_validation_dataloader, student_prediction_dataloader = student.init_dataloader(args.student_train, args.student_test)

    # Initialize Teacher
    logger.info("*** Initialise the teacher model with golds labels ***")
    init_fine_tune_teacher_res = teacher.finetune(teacher_train_dataloader, teacher_validation_dataloader)
    results['teacher_init'] = init_fine_tune_teacher_res
    teacher.save('init_supervised_teacher')

    student_test_dataframe = pd.read_pickle(args.student_test)

    # Self-Training Loop
    for iter in range(args.self_training_iter):
        logger.info("*** We are starting the self-training loop {} ***".format(iter))

        # Apply the teacher model on unlabelled data to gen silver labels
        unlabeled_dataset = load_dataset(args.unlabeled_dataset)
        unlabeled_dataframe = pd.read_pickle(args.unlabeled_dataset)
        # Apply Teacher on unlabeled data
        teacher_preds, teacher_conf = teacher.predict(dataset=unlabeled_dataset)


        # Update ublabeled data with teacher's predictions
        unlabeled_dataframe['teacher_labels'] = teacher_preds
        unlabeled_dataframe['teacher_conf'] = teacher_conf

        # Based on p, filter out the data to keep into next loop
        next_iter_unlabeled_dataframe = unlabeled_dataframe[unlabeled_dataframe['techer_conf'] > args.self_training_confidence_thread_p]

        combo_dataset = pd.concat([gold_dataframe, next_iter_unlabeled_dataframe])

        # Re-train student model with teacher's silver labels and gold labels
        logger.info('training student on silver labeled instances provided by the teacher')
        train_res = student.finetune_raw(
            train_data=combo_dataset,
            test_data=student_test_dataframe
        )

        # Evaluate student performance and update records
        test_preds, test_conf, test_res = student.predict_acc(test_data = student_test_dataframe)
        logger.info("Student Test performance on iter {}: {}".format(iter, test_acc))

        test_res_list.append(test_res)
        student_pred_list.append(test_preds)



    # Store the final results
    logger.info("Final Results")

    results['student_test_iter'] = test_res_list
    results['student_pred_lst'] = student_pred_list

    # Save models and results
    student.save("student_final")
    teacher.save("teacher_final")
    return results



#batch
class Batch():
    def __init__(self, data, idx, batch_size, device):
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor([x[0] for x in cur_batch])
        seg = torch.tensor([x[1] for x in cur_batch])
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src != 0)

        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src


#predict
def prediction(dataset, model, args):
    preds = []
    golds = []
    model.eval()
    for i in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(dataset, i, args.batch_size, args.device).get()
        preds += model.predict(src, seg, mask_src)
        golds += label.cpu().data.numpy().tolist()
    return f1_score(golds, preds, average='macro'), preds

def main():
    args_parser = argparse.ArgumentParser()

    # Main Arguments
    parser.add_argument('--from_dataset', help='Source Dataset name', type=str, default='Twitter1516')
    parser.add_argument('--target_dataset', help='Target Dataset name', type=str, default='WEIBO')
    parser.add_argument('--unlabeled_dataset', help='Unlabeled Dataset', type=str, default='Extra_CHN')
    parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default='../data')
    parser.add_argument("--student_model", help="Student base model", type=str, default='CHNBERT')
    parser.add_argument("--teacher_model", help="Teacher base model", type=str, default='MBERT')
    parser.add_argument('--student_train', type=str, default='../student_train')
    parser.add_argument('--student_test', type=str, default='../student_test')
    parser.add_argument('--teacher_train', type=str, default='../teacher_train')
    parser.add_argument('--teacher_test', type=str, default='../teacher_test')

    #Arguments for experiments
    args_parser.add_argument('--base_model', default='MBERT', choices=['MBERT', 'XLMR','BERT','CHNBERT'], help='select one of models')
    args_parser.add_argument('--pretrain', default=True, help='whether to use further pretained model and vocab')
    args_parser.add_argument('--num_labels', type=int, default=2, help='currently we focus on binary classification')
    args_parser.add_argument('--lower_case', default=False, help='whether do lower case when tokenizing')
    args_parser.add_argument('--pretrain_data_path', default='fp_model/',help='path to further pretrained base models')
    args_parser.add_argument('--max_length', type=int, default=384, help='maximum token length for each input sequence')
    args_parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    args_parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
    args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
    args_parser.add_argument('--num_train_epochs', type=int, default=5, help='total epoch')
    args_parser.add_argument('--warmup_steps', type=int, default=0, help='warmup_steps, the default value is 10% of total steps')
    args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
    args_parser.add_argument('--seed', type=int, default=42, help='set up a seed for reproductibility')
    args_parser.add_argument('--self_training_iter', type=int, default=5, help='number of iteration for running self-training loop')
    args_parser.add_argument('--self_training_confidence_thread_p', type=float, default=0.95, help='confidence thread for self-training loop')
    args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    args_parser.add_argument('--gpu', default=True, help='whether to use GPU')
    args_parser.add_argument('--logdir', help='Directory for store logs', type=str, default='logs/')
    args_parser.add_argument('--output_dir',help='Directory for store output results', type=str, default='res/')
    args = args_parser.parse_args()


    #setupCUDA
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Start Experiment
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H_%M")

    args.experiment_folder = os.path.join(args.experiment_folder, args.dataset)
    args.logdir = os.path.join(args.experiment_folder, args.logdir)
    experiment_dir = str(Path(args.logdir).parent.absolute())


    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if args.debug:
        args.logdir = os.path.join(args.experiment_folder, 'debug')
    else:
        args.logdir = args.logdir + "/" + date_time + "_st{}".format(args.student_name.upper())

    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger(logfile=os.path.join(args.logdir, 'log.log'))

    logger.info("*** EXPERIMENT Start *** with args={}".format(args))
    run_rumour(args, logger=logger)
    close(logger)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = summarize_results(args.output_dir, args.from_dataset, args.target_dataset)
    print("*** Results summary (metric={}): {} ***".format(args.metric, all_results))

if __name__ == "__main__":
    main()
