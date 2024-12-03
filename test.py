from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
args=option.parse_args()
from config import *
from models.mgfn import mgfn as Model
from datasets.dataset import Dataset
import train
from sklearn.metrics import top_k_accuracy_score

classes = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

MODEL_LOCATION = 'saved_models/'
MODEL_NAME = 'triplet_deform_best'
MODEL_NAME2 = 'contrastive_deform_best'
MODEL_NAME3 = 'new_loss_deform_best'
MODEL_LABEL = 'Triplet Loss'
MODEL_LABEL2 = 'MC Loss'
MODEL_LABEL3 = 'Experimental Loss'
MODEL_EXTENSION = '.pkl'

def test(dataloader, model, args, device, name = MODEL_NAME):
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = []
        labels = []
        y_scores = []
        class_results = {}
        for _, inputs in tqdm(enumerate(dataloader)):
            labels += inputs[1].cpu().detach().tolist()
            atypes = inputs[2].cpu().detach().tolist()
            input = inputs[0].to(device)
            scores, _, _ = model(input)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            # scores = torch.nn.Softmax(dim=1)(scores)
            # y_scores += scores.cpu().detach().tolist()
            #pred_ = torch.argmax(scores, dim=1)
            #print(scores)
            pred_ = scores.cpu().detach().tolist()
            #print(pred_)
            pred += pred_
            for i, atype in enumerate(atypes):
                #print(atype, pred_[i])
                class_results[atype] = np.append(class_results.get(atype, np.array([])), pred_[i])
        print()
        for i, at in enumerate(classes):
            if class_results.get(i, None) is None:
                continue
            #print(at + ": " + str(np.sum(class_results[i] == i)) + "/" + str(len(class_results[i])))
            if i == 0:
                print(at + ": " + str(np.sum(class_results[i] <= 0.5)) + "/" + str(len(class_results[i])))
            else:
                print(at + ": " + str(np.sum(class_results[i] > 0.5)) + "/" + str(len(class_results[i])))
        fpr, tpr, threshold = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(labels, pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('roc_auc : ' + str(roc_auc))
        # top1 = top_k_accuracy_score(labels, y_scores, k=1, labels=range(14))
        # top3 = top_k_accuracy_score(labels, y_scores, k=3, labels=range(14))
        # top5 = top_k_accuracy_score(labels, y_scores, k=5, labels=range(14))
        # print("Top-1 Accuracy: " + str(top1))
        # print("Top-3 Accuracy: " + str(top3))
        # print("Top-5 Accuracy: " + str(top5))

        #return top1, top3, top5

        plt.figure()  
        plt.plot(fpr, tpr, label='ROC Curve (Area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for UCF-Crime Anomaly Detection')
        plt.legend()
        plt.savefig('plots/' + name + '_roc.png', bbox_inches='tight')
        plt.close()

        plt.figure()  
        plt.plot(recall, precision, label='PR Curve (Area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for UCF-Crime Anomaly Detection')
        plt.legend()
        plt.savefig('plots/' + name + '_pr.png', bbox_inches='tight')
        plt.close()

        if MODEL_NAME2 is None:
            return roc_auc, pr_auc
        
        else:
            return fpr, tpr, roc_auc, recall, precision, pr_auc


if __name__ == '__main__':
    args = option.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    model = Model()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=2 * args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    if MODEL_NAME2 is None:
        model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION).items()})
        auc = test(test_loader, model, args, device)
    elif MODEL_NAME3 is None:
        model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION).items()})
        model1_results = test(test_loader, model, args, device, name=MODEL_NAME)
        model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME2 + MODEL_EXTENSION).items()})
        model2_results = test(test_loader, model, args, device, name=MODEL_NAME2)
        plt.figure()  
        plt.plot(model1_results[0], model1_results[1], label=MODEL_LABEL + ' ROC Curve (Area = %0.2f)' % model1_results[2])
        plt.plot(model2_results[0], model2_results[1], label=MODEL_LABEL2 + ' ROC Curve (Area = %0.2f)' % model2_results[2])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for UCF-Crime Anomaly Detection')
        plt.legend()
        plt.savefig('plots/' + MODEL_NAME + '_' + MODEL_NAME2 + '_roc.png', bbox_inches='tight')
        plt.close()

        plt.figure()  
        plt.plot(model1_results[3], model1_results[4], label=MODEL_LABEL + ' PR Curve (Area = %0.2f)' % model1_results[5])
        plt.plot(model2_results[3], model2_results[4], label=MODEL_LABEL + ' PR Curve (Area = %0.2f)' % model2_results[5])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for UCF-Crime Anomaly Detection')
        plt.legend()
        plt.savefig('plots/' + MODEL_NAME + '_' + MODEL_NAME2 + '_pr.png', bbox_inches='tight')
        plt.close()
    else:
        model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION).items()})
        model1_results = test(test_loader, model, args, device, name=MODEL_NAME)
        model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME2 + MODEL_EXTENSION).items()})
        model2_results = test(test_loader, model, args, device, name=MODEL_NAME2)
        model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOCATION + MODEL_NAME3 + MODEL_EXTENSION).items()})
        model3_results = test(test_loader, model, args, device, name=MODEL_NAME3)
        plt.figure()  
        plt.plot(model1_results[0], model1_results[1], label=MODEL_LABEL + ' ROC Curve (Area = %0.2f)' % model1_results[2])
        plt.plot(model2_results[0], model2_results[1], label=MODEL_LABEL2 + ' ROC Curve (Area = %0.2f)' % model2_results[2])
        plt.plot(model3_results[0], model3_results[1], label=MODEL_LABEL3 + ' ROC Curve (Area = %0.2f)' % model3_results[2])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for UCF-Crime Anomaly Detection')
        plt.legend()
        plt.savefig('plots/' + MODEL_NAME + '_' + MODEL_NAME2 + '_' + MODEL_NAME3 + '_roc.png', bbox_inches='tight')
        plt.close()

        plt.figure()  
        plt.plot(model1_results[3], model1_results[4], label=MODEL_LABEL + ' PR Curve (Area = %0.2f)' % model1_results[5])
        plt.plot(model2_results[3], model2_results[4], label=MODEL_LABEL2 + ' PR Curve (Area = %0.2f)' % model2_results[5])
        plt.plot(model3_results[3], model3_results[4], label=MODEL_LABEL3 + ' PR Curve (Area = %0.2f)' % model3_results[5])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for UCF-Crime Anomaly Detection')
        plt.legend()
        plt.savefig('plots/' + MODEL_NAME + '_' + MODEL_NAME2 + '_' + MODEL_NAME3 + '_pr.png', bbox_inches='tight')
        plt.close()



