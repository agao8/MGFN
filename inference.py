from torch.utils.data import DataLoader
import option
import torch
from tqdm import tqdm
args=option.parse_args()
from config import *
from models.mgfn import mgfn as Model
from datasets.dataset import Dataset
import time


classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

MODEL_LOCATION = 'saved_models/'
MODEL_NAME = 'triplet_deform_best'
MODEL_EXTENSION = '.pkl'

if __name__ == '__main__':

    start_time = time.time()

    args = option.parse_args()
    args.test_rgb_list = None
    args.test_rgb_list = input('Patch to file list: ')
    outfile = input('Name of output file(default: output.txt): ') or "output.txt"

    
    device = torch.device("cuda")
    model_load_time = time.time()
    model = Model()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=2 * args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION).items()})

    print()
    print("Time to load model: {:.2f} seconds".format(time.time() - model_load_time))
    print()

    model_start_time = time.time()

    with torch.no_grad():
        model.eval()
        pred = []
        labels = []
        y_scores = []
        class_results = {}
        for _, inputs in tqdm(enumerate(test_loader)):
            labels += inputs[1].cpu().detach().tolist()
            atypes = inputs[2].cpu().detach().tolist()
            input = inputs[0].to(device)
            scores, _, _ = model(input)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            pred_ = scores.cpu().detach().tolist()
            pred += pred_

    model_time = time.time() - model_start_time
    num_samples = len(test_loader.dataset)

    print()
    print("Time to perform inference on {} samples: {:.2f} seconds".format(num_samples, model_time))
    print("Average time per sample: {:.2f} ms".format(1000 * model_time / num_samples))

    print()
    with open("predictions_" + outfile, 'w') as file:
        for prediction in pred:
            file.write(f"{round(prediction)}\n")

    print("Predictions written to " + "predictions_" + outfile)

    with open("scores_" + outfile, 'w') as file:
        for prediction in pred:
            file.write(f"{prediction}\n")

    print("Scores written to " + "scores_" + outfile)

    print()
    print("Total Time: {:.2f} seconds".format(time.time() - start_time))