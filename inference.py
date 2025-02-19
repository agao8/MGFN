from torch.utils.data import DataLoader
import option
import torch
from tqdm import tqdm
args=option.parse_args()
from config import *
from MGFN.models.model import mgfn as Model
from datasets.dataset import Dataset
import time
import gradio as gr
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

MODEL_LOCATION = 'saved_models/'
MODEL_NAME = 'triplet_deform_best'
MODEL_EXTENSION = '.pkl'

args = option.parse_args()
args.test_rgb_list = None

def run(mode, file_path, outfile):
    start_time = time.time()
    args.test_rgb_list = file_path
    outfile = outfile

    device = torch.device("cuda")
    model_load_time = time.time()
    model = Model()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=2 * args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION).items()})

    outtext = "Time to load model: {:.2f} seconds".format(time.time() - model_load_time) + "\n"
    yield outtext + "Running Model...", None

    model_start_time = time.time()

    with torch.no_grad():
        model.eval()
        pred = []
        labels = []
        for _, inputs in enumerate(test_loader):
            labels += inputs[1].cpu().detach().tolist()
            input = inputs[0].to(device)
            scores, _, _ = model(input)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            pred_ = scores.cpu().detach().tolist()
            pred += pred_

    model_time = time.time() - model_start_time
    num_samples = len(test_loader.dataset)

    outtext += "\n" + "Time to perform inference on {} samples: {:.2f} seconds".format(num_samples, model_time) + "\n"
    outtext += "Average time per sample: {:.2f} ms".format(1000 * model_time / num_samples) + "\n"
    yield outtext, None

    fpr, tpr, threshold = roc_curve(labels, pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, th = precision_recall_curve(labels, pred)
    pr_auc = auc(recall, precision)

    with open("predictions_" + outfile, 'w') as file:
        for prediction in pred:
            file.write(f"{round(prediction)}\n")

    outtext += "\n" + "Predictions written to " + "predictions_" + outfile  + "\n"
    yield outtext, None

    with open("scores_" + outfile, 'w') as file:
        for prediction in pred:
            file.write(f"{prediction}\n")

    outtext += "Scores written to " + "scores_" + outfile  + "\n"
    yield outtext, None

    outtext += "\n" + "Total Time: {:.2f} seconds".format(time.time() - start_time)  + "\n"
    yield outtext, None

    if mode == "Evaluation":
        plt.figure()  
        plt.plot(fpr, tpr, label='ROC Curve (Area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for UCF-Crime Anomaly Detection')
        plt.legend()
    else:
        pred = np.array(pred)
        plt.figure()  
        plt.bar(["Normal", "Anomaly"], [np.sum(pred < 0.5), np.sum(pred >= 0.5)], label=["Normal", "Anomaly"], color=["tab:blue", "tab:red"])
        plt.xlabel('Class')
        plt.ylabel('Number of Classifications')
        plt.title('Number of Classifications per Class')
        plt.legend()

    yield outtext, plt

    plt.close()



if __name__ == '__main__':

    iface = gr.Interface(
        fn=run, 
        inputs=[
            gr.Dropdown(["Evaluation", "Inference"], label="Evaluation or Inference"),
            gr.Textbox(label="Path to file list"),
            gr.Textbox(label="Name of output file")
        ],
        outputs=[
            gr.Textbox(label="Output"),
            gr.Plot(label="Plot")
        ]
    )


    iface.launch()

