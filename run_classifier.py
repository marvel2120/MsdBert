import argparse
import logging
import os
import random

import numpy as np
import torch
import torchvision
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer

from models import MsdBERT, BertOnly, ResNetOnly, Res_BERT
from optimizer import BertAdam
from resnet_utils import myResnet
from utils import Processer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str)
    parser.add_argument("--image_dir",
                        default='../image',
                        type=str)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=77,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_hashtag_length",
                        default=12,
                        type=int,
                        help="The maximum total hashtag input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--model_select', default='MsdBERT',
                        help='model select')  # select from MsdBERT, ResNetOnly, BertOnly, Res_BERT
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    logger.info("************** Using: " + torch.cuda.get_device_name(0) + " ******************")
    # set seed val
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of 'do_train' or 'do_test' must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = Processer(args.data_dir, args.image_dir, args.model_select, args.max_seq_length,
                          args.max_hashtag_length)

    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    train_examples = None
    num_train_steps = None
    eval_examples = None
    if args.do_train:
        train_examples = processor.get_train_examples()
        eval_examples = processor.get_eval_examples()
        num_train_steps = int((len(train_examples) * args.num_train_epochs) / args.train_batch_size)

    if args.model_select == 'BertOnly':
        model = BertOnly()
    elif args.model_select == 'ResNetOnly':
        model = ResNetOnly()
    elif args.model_select == 'Res_BERT':
        model = Res_BERT()
    elif args.model_select == 'MsdBERT':
        model = MsdBERT()
    else:
        raise ValueError("A model must be given.")

    model.to(device)
    net = torchvision.models.resnet152(pretrained=True)
    encoder = myResnet(net).to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = torch.nn.DataParallel(encoder)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': encoder.parameters()}
    ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

    train_loss = 0
    if args.do_train:
        train_features = processor.convert_mm_examples_to_features(train_examples, label_list, tokenizer)
        eval_features = processor.convert_mm_examples_to_features(eval_examples, label_list, tokenizer)

        train_input_ids, train_input_mask, train_added_input_mask, train_img_feats, \
        train_hashtag_input_ids, train_hashtag_input_mask, train_label_ids = train_features
        train_data = TensorDataset(train_input_ids, train_input_mask, train_added_input_mask, train_img_feats, \
                                   train_hashtag_input_ids, train_hashtag_input_mask, train_label_ids)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.train_batch_size)

        eval_input_ids, eval_input_mask, eval_added_input_mask, eval_img_feats, \
        eval_hashtag_input_ids, eval_hashtag_input_mask, eval_label_ids = eval_features
        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_added_input_mask, eval_img_feats, \
                                  eval_hashtag_input_ids, eval_hashtag_input_mask, eval_label_ids)
        eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=args.eval_batch_size)

        max_acc = 0.0
        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx + 1) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()
            encoder.train()
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                train_input_ids, train_input_mask, train_added_input_mask, train_img_feats, \
                train_hashtag_input_ids, train_hashtag_input_mask, train_label_ids = batch
                imgs_f, img_mean, train_img_att = encoder(train_img_feats)

                loss = model(train_input_ids, train_img_att, train_input_mask, train_added_input_mask,
                             train_hashtag_input_ids,
                             train_hashtag_input_mask, train_label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            encoder.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []
            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                eval_input_ids, eval_input_mask, eval_added_input_mask, eval_img_feats, \
                eval_hashtag_input_ids, eval_hashtag_input_mask, eval_label_ids = batch
                imgs_f, img_mean, eval_img_att = encoder(eval_img_feats)
                with torch.no_grad():
                    tmp_eval_loss = model(eval_input_ids, eval_img_att, eval_input_mask, eval_added_input_mask,
                                          eval_hashtag_input_ids,
                                          eval_hashtag_input_mask, eval_label_ids)
                    logits = model(eval_input_ids, eval_img_att, eval_input_mask, eval_added_input_mask,
                                   eval_hashtag_input_ids,
                                   eval_hashtag_input_mask)
                logits = logits.detach().cpu().numpy()
                label_ids = eval_label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += eval_input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            train_loss = tr_loss / nb_tr_steps if args.do_train else None

            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = macro_f1(true_label, pred_outputs)
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'precision': precision,
                      'recall': recall,
                      'f_score': F_score,
                      'train_loss': train_loss}
            logger.info("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if eval_accuracy > max_acc:
                torch.save(model.state_dict(), output_model_file)
                torch.save(encoder.state_dict(), output_encoder_file)
                logger.info("better model")
                max_acc = eval_accuracy

    if args.do_test:
        model.load_state_dict(torch.load(output_model_file))
        encoder.load_state_dict(torch.load(output_encoder_file))
        model.to(device)
        encoder.to(device)
        model.eval()
        encoder.eval()

        test_examples = processor.get_test_examples()
        logger.info("***** Running evaluation on Test Set*****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        test_features = processor.convert_mm_examples_to_features(test_examples, label_list, tokenizer)

        test_input_ids, test_input_mask, test_added_input_mask, test_img_feats, \
        test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids = test_features
        test_data = TensorDataset(test_input_ids, test_input_mask, test_added_input_mask, test_img_feats, \
                                  test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids)
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.eval_batch_size)

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        true_label_list = []
        pred_label_list = []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            test_input_ids, test_input_mask, test_added_input_mask, test_img_feats, \
            test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids = batch
            imgs_f, img_mean, test_img_att = encoder(test_img_feats)
            with torch.no_grad():
                tmp_eval_loss = model(test_input_ids,test_img_att, test_input_mask, test_added_input_mask, \
                                      test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids)
                logits = model(test_input_ids, test_img_att, test_input_mask, test_added_input_mask, \
                               test_hashtag_input_ids, test_hashtag_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = test_label_ids.to('cpu').numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += test_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = train_loss if args.do_train else None

        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)

        precision, recall, F_score = macro_f1(true_label, pred_outputs)
        result = {'test_loss': eval_loss,
                  'test_accuracy': eval_accuracy,
                  'precision': precision,
                  'recall': recall,
                  'f_score': F_score,
                  'train_loss': loss}

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
        fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')
        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + '\n')
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + '\n')

        fout_p.close()
        fout_t.close()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
