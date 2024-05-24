import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from Constants import EPOCHS
from Constants import LR
from Constants import WEIGHT_DECAY
from Constants import CATEGORIES_DICT
from Predict import get_sentiment
from Constants import text
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Obtiene las funciones de perdida y optimizadores
def get_loss_opt(model):
    #Funcion de pérdida
    criterio = torch.nn.CrossEntropyLoss()

    #Optimizador
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = LR,
        weight_decay = WEIGHT_DECAY
    )
    return criterio, optimizer


# Funcion que cuenta los parametros de un modelo
def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

# Funcion de entrenamiento del modelo
def entrena(data_loader, modelo, optimizer, criterio):
    '''
    Function to train the model for one epoch.

    Args:
    data_loader: DataLoader object for loading the training data.
    modelo: The model to be trained.
    epoch (int): Current epoch number.

    Returns:
    float: Average accuracy for the epoch.
    float: Average loss for the epoch.
    '''
    #Poner el modelo en modo train
    modelo.train()

    # Inicializa el accuracy, count y loss para cada epoch
    epoch_acc = 0
    epoch_loss = 0
    total_count = 0

    for idx, data in enumerate(data_loader):
        attention_mask = data['attention_mask']
        input_ids = data['input_ids']
        labels = data['label']

        # reestablece los gradientes despues de cada batch
        optimizer.zero_grad()

        #optener predicciones del modelo
        prediccion = modelo(input_ids, attention_mask)

        #optener la perdida
        loss = criterio(prediccion, labels)

        # Backpropagate la perdida y calcular los gradientes
        loss.backward()

        acc = (prediccion.argmax(1) == labels).sum()

        # evitar que los gradientes sean demasiado grandes
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 0.1)

        #Actualizacion de pesos
        optimizer.step()

        #Llevar el conteo de pérdida y accuracy
        epoch_acc += acc.item()
        epoch_loss += loss.item()
        total_count += labels.size(0)
    return epoch_acc/total_count, epoch_loss/total_count

#Funcion de evaluacion del modelo
def evalua(data_loader, modelo, criterio):
    '''
        Function to evaluate the model on validation or test data for one epoch.

        Args:
        data_loader: DataLoader object for loading the validation or test data.
        modelo: The model to be evaluated.
        epoch (int): Current epoch number.

        Returns:
        float: Average accuracy for the epoch.
        float: Average loss for the epoch.
    '''
    modelo.eval()
    epoch_acc = 0
    total_count = 0
    epoch_loss = 0

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            attention_mask = data['attention_mask']
            input_ids = data['input_ids']
            labels = data['label']
            #Obtener prediccion
            prediccion = modelo(input_ids, attention_mask)

            #Perdida y accuracy
            loss = criterio(prediccion, labels)
            acc = (prediccion.argmax(1) == labels).sum()

            # Llevar el conteo de la perdida y acc para el epoch
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            total_count += labels.size(0)

    return epoch_acc/total_count, epoch_loss/total_count

def train(model, path: str, train_loader, val_loader, optimizer, criterio):
    # Obten la mejor perdida
    major_loss_validation = float('inf')
    
    major_loss_training = float('inf')
    major_accuracy_training = 0

    #entrenar
    for epoch in tqdm(range(1, EPOCHS + 1)):
        #Entrenamiento
        entrenamiento_acc, entrenamiento_loss = entrena(train_loader, model, optimizer, criterio)
        if entrenamiento_acc > major_accuracy_training: major_accuracy_training = entrenamiento_acc 
        if entrenamiento_loss > major_loss_training: major_loss_training = entrenamiento_loss

        #Validacion
        validacion_acc, validacion_loss = evalua(val_loader, model, criterio)

        if(epoch % 20 == 0):
            tqdm.write(f"\nEpoch: {epoch} | Training acc: {entrenamiento_acc} | Training loss: {entrenamiento_loss} | Val acc: {validacion_acc} | Val loss: {validacion_loss}")
            tqdm.write("-"*150)

        # Guardar el mejor modelo
        if validacion_loss < major_loss_validation:
            model_info = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validacion_loss,
            }
            best_valid_loss = validacion_loss
            torch.save(model_info, path)
    return major_loss_training, major_accuracy_training


def get_predictions (model, data_loader):
    model = model.eval ()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            texts = d["texts"]
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            labels = d["label"]

            outputs = model(input_ids, attention_mask)

            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).to(DEVICE)
    prediction_probs = torch.stack(prediction_probs).to(DEVICE)
    real_values = torch.stack(real_values).to(DEVICE)
    return review_texts, predictions, prediction_probs, real_values

def conf_matrix(confus_matrix):
    hmap = sns.heatmap(confus_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted Sentiment')

def conf_matrix(confus_matrix):
    hmap = sns.heatmap(confus_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.show()


def verify_model(model, data_loader, tokenizer):
    review_texts, predictions, prediction_probs, real_values = get_predictions(model, data_loader)

    class_names = list(CATEGORIES_DICT.values())

    print("Statistics: ")
    print(classification_report(real_values.to('cpu').numpy(), predictions.to('cpu').numpy(), target_names=class_names))

    cm = confusion_matrix(real_values.cpu().numpy(), predictions.cpu().numpy())
    df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
    conf_matrix(df_cm)

    print(get_sentiment(model, text, tokenizer))
    to_ret = pd.DataFrame({
        'Review Text': review_texts,
        'Prediction': predictions.to('cpu'),
        'Probability Negative': prediction_probs.to('cpu')[:,0],
        'Probability Neutral': prediction_probs.to('cpu')[:,1],
        'Probability Positive': prediction_probs.to('cpu')[:,2],
        'Real Value': real_values.to('cpu')
    })
    return to_ret