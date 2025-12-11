import torch
from collections import defaultdict, Counter
from tqdm.auto import tqdm

def get_predictor(model, predictor_cls, score_fn, device):
    model = model.to(device)
    model.eval()
    predictor = predictor_cls(score_fn, model)
    return predictor

def calibrate_predictor(predictor, cal_loader, alpha):
    predictor.calibrate(cal_loader, alpha=alpha)

def eval_predictor(predictor, test_dl):
    return predictor.evaluate(test_dl)

def get_conformal_predictions(predictor, test_ds, train_dataset):
    pred_inds = []
    pred_classes = []
    for x, _ in tqdm(test_ds, desc="Getting Predictions"):
        pred = predictor.predict(x.unsqueeze(0))
        indices = torch.nonzero(pred.flatten() == 1).flatten()
        pred_inds.append(indices)
        clss = [train_dataset.class_names[id.item()] for id in indices.detach().cpu()]
        pred_classes.append(clss)
    return pred_inds, pred_classes

def get_per_sample_set_size(pred_classes):
    per_sample_pred_set_size_dict = defaultdict(int)
    for i, clss in enumerate(pred_classes):
        per_sample_pred_set_size_dict[i] = len(clss)
    return per_sample_pred_set_size_dict

def get_set_size_counter(per_sample_pred_set_size_dict):
    return Counter(per_sample_pred_set_size_dict.values())