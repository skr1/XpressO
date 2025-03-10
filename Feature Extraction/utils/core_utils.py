import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
import inspect

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_confidence_interval(metric_func, y_true, y_pred, n_iterations=1000, alpha=0.05):
    """Calculate the confidence interval for a given metric function."""
    np.random.seed(0)  # For reproducibility
    metrics = []
    n_samples = len(y_true)

    # Check if the metric function accepts the zero_division argument
    func_params = inspect.signature(metric_func).parameters
    accepts_zero_division = "zero_division" in func_params
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_bootstrap = np.array(y_true)[indices]
        y_pred_bootstrap = np.array(y_pred)[indices]

        # Check if there are at least two classes in y_true_bootstrap
        if len(np.unique(y_true_bootstrap)) > 1:
            if accepts_zero_division:
                metric = metric_func(y_true_bootstrap, y_pred_bootstrap, average="weighted", zero_division=0)
            else:
                metric = metric_func(y_true_bootstrap, y_pred_bootstrap)
            metrics.append(metric)
        else:
            metrics.append(0.0)  # Use a default value if only one class is present

    # Convert to numpy array
    metrics = np.array(metrics)

    # Filter out NaN values
    metrics = metrics[~np.isnan(metrics)]
    
    # Calculate confidence intervals
    lower_bound = np.percentile(metrics, 100 * alpha / 2)
    upper_bound = np.percentile(metrics, 100 * (1 - alpha / 2))
    
    return lower_bound, upper_bound

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.y_true = []
        self.y_pred = []

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.y_true.append(Y)
        self.y_pred.append(Y_hat)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        self.y_true.extend(Y)
        self.y_pred.extend(Y_hat)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

    def get_metrics(self):
        # precision = precision_score(self.y_true, self.y_pred, zero_division=0, average='weighted')
        # recall = recall_score(self.y_true, self.y_pred, zero_division=0, average='weighted')
        # f1 = f1_score(self.y_true, self.y_pred, zero_division=0, average='weighted')
        precision = calculate_confidence_interval(precision_score, self.y_true, self.y_pred)
        recall = calculate_confidence_interval(recall_score, self.y_true, self.y_pred)
        f1 = calculate_confidence_interval(f1_score, self.y_true, self.y_pred)
    
        return precision, recall, f1

def extract_metric_value(metric):
    if isinstance(metric, tuple):
        return (metric[0] + metric[1]) / 2  # Use the mean of the confidence interval
    return metric

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_precision, val_recall, val_f1 = summary(model, val_loader, args.n_classes)

    val_precision = extract_metric_value(val_precision)
    val_recall = extract_metric_value(val_recall)
    val_f1 = extract_metric_value(val_f1)
    val_auc = extract_metric_value(val_auc)

    # Print the results using the extracted scalar values
    print('Val error: {:.4f}, ROC AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format
        (val_error, val_auc, val_precision, val_recall, val_f1)
    )
    results_dict, test_error, test_auc, acc_logger, test_precision, test_recall, test_f1 = summary(model, test_loader, args.n_classes)
    test_precision = extract_metric_value(val_precision)
    test_recall = extract_metric_value(val_recall)
    test_f1 = extract_metric_value(val_f1)
    test_auc = extract_metric_value(test_auc)
    
    print('Test error: {:.4f}, ROC AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_precision, test_recall, test_f1))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/val_precision', val_precision, 0)
        writer.add_scalar('final/val_recall', val_recall, 0)
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/test_precision', test_precision, 0)
        writer.add_scalar('final/test_recall', test_recall, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_precision, val_precision, test_recall, val_recall, test_f1, val_f1

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(f'batch {batch_idx}, loss: {loss_value:.4f}, instance_loss: {instance_loss_value:.4f}, weighted_loss: {total_loss.item():.4f}, label: {label.item()}, bag_size: {data.size(0)}')

        error = calculate_error(Y_hat, label)
        train_error += error
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(f'class {i} clustering acc {acc}: correct {correct}/{count}')

    precision, recall, f1 = acc_logger.get_metrics()
    
    # Extract metric values if they are tuples
    if isinstance(precision, tuple):
        precision_value, precision_ci = precision
    else:
        precision_value = precision
    
    if isinstance(recall, tuple):
        recall_value, recall_ci = recall
    else:
        recall_value = recall
    
    if isinstance(f1, tuple):
        f1_value, f1_ci = f1
    else:
        f1_value = f1

    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_clustering_loss: {train_inst_loss:.4f}, train_error: {train_error:.4f}, precision: {precision_value:.4f}, recall: {recall_value:.4f}, f1: {f1_value:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer and acc is not None:
            writer.add_scalar(f'train/class_{i}_acc', acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        writer.add_scalar('train/precision', precision_value, epoch)
        writer.add_scalar('train/recall', recall_value, epoch)
        writer.add_scalar('train/f1', f1_value, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(f'batch {batch_idx}, loss: {loss_value:.4f}, label: {label.item()}, bag_size: {data.size(0)}')
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)

    precision, recall, f1 = acc_logger.get_metrics()
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_error: {train_error:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer:
            writer.add_scalar(f'train/class_{i}_acc', acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/precision', precision, epoch)
        writer.add_scalar('train/recall', recall, epoch)
        writer.add_scalar('train/f1', f1, epoch)

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count = 0
    
    prob = np.zeros((len(loader.dataset), n_classes))
    labels = np.zeros(len(loader.dataset))
    
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            val_inst_loss += instance_loss.item()

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = Y_prob.cpu().numpy()
            labels[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = label.cpu().numpy()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        aucs = []
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(auc(fpr, tpr))
        auc = np.mean(aucs)

    precision, recall, f1 = acc_logger.get_metrics()
    
    # Extract numeric values, handling tuples
    precision_value = precision[0] if isinstance(precision, tuple) else precision
    recall_value = recall[0] if isinstance(recall, tuple) else recall
    f1_value = f1[0] if isinstance(f1, tuple) else f1

    # Handle NaN and Inf values
    precision_value = np.nan_to_num(precision_value, nan=0.0, posinf=0.0, neginf=0.0)
    recall_value = np.nan_to_num(recall_value, nan=0.0, posinf=0.0, neginf=0.0)
    f1_value = np.nan_to_num(f1_value, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'\nVal Set, val_loss: {val_loss:.4f}, val_clustering_loss: {val_inst_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}, precision: {precision_value:.4f}, recall: {recall_value:.4f}, f1: {f1_value:.4f}')
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc:.4f}, correct {correct}/{count}')
        if writer and acc is not None:
            writer.add_scalar(f'val/class_{i}_acc', acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/clustering_loss', val_inst_loss, epoch)
        writer.add_scalar('val/precision', precision_value, epoch)
        writer.add_scalar('val/recall', recall_value, epoch)
        writer.add_scalar('val/f1', f1_value, epoch)
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        stop = early_stopping.early_stop
    else:
        stop = False

    return stop


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, _ = model(data)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(auc(fpr, tpr))
        auc = np.mean(aucs)

    precision, recall, f1 = acc_logger.get_metrics()
    print(f'\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer and acc is not None:
            writer.add_scalar(f'val/class_{i}_acc', acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/precision', precision, epoch)
        writer.add_scalar('val/recall', recall, epoch)
        writer.add_scalar('val/f1', f1, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        stop = early_stopping.early_stop
    else:
        stop = False

    return stop

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        # auc = roc_auc_score(all_labels, all_probs[:, 1])
        auc = calculate_confidence_interval(roc_auc_score, all_labels, all_probs[:,1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                # fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                fpr, tpr, _ = calculate_confidence_interval(roc_curve, binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    # precision = precision_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')
    # recall = recall_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')
    # f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')
    
    precision, recall, f1 = acc_logger.get_metrics()

    return patient_results, test_error, auc, acc_logger, precision, recall, f1

