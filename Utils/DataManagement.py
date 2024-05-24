#@title Importación de librerías
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

BATCH_SIZE = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Label encoding para las clases
def convert_to_numeric(category: str) -> int:
    '''
    This function converts categorical labels into numeric representations.

    Parameters:
    category (str): The categorical label to be converted.

    Returns:
    int: The numeric representation of the input category.
    '''

    if category == 'NEGATIVO': return int(0)
    if category == 'NEUTRO': return int(1)
    if category == 'POSITIVO': return int(2)
    if category == 'IRONÍA': return int(3)

#Tiene la informacion de los datos de entrenamiento
def preparar_datos(path):
    '''
        This function prepares training data by loading it from a CSV file,
        encoding it properly, and transforming categorical labels into numeric representations.

        Parameters:
        path (str): The path to the CSV file containing the training data.
                    Default value is TRAINING_DATA.

        Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed training data.
    '''
    datos = pd.read_csv(
        path,
    )[['text', 'TAG']]
    datos['TAG_num'] = datos['TAG'].apply(lambda x: convert_to_numeric(x))
    datos = datos[datos['TAG_num'] != 3]
    datos = datos.drop('TAG', axis = 1)
    return datos


#Clase para contruir el dataloader
class PeriodisticDataset(Dataset):
    def __init__(self, _text: list[str], _classification: list[int], _tokenizer):
        self.text: list[str] = _text
        self.classification: list[int] = _classification
        self.tokenizer = _tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
        )
        return {
            'text': text,
            'input_ids': torch.tensor(encoding['input_ids']).to(DEVICE),
            'attention_mask': torch.tensor(encoding['attention_mask']).to(DEVICE),
            'label': torch.tensor(self.classification[idx]).to(DEVICE)
        }

# Define la funcion de collate
def collate_fn(batch):
    # Extracting input_ids, attention_mask, and target tensors from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    target = [item['label'] for item in batch]
    texts = [item['text'] for item in batch]

    # Padding sequences within each batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Returning padded sequences and targets
    return {
        'texts': texts,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'label': torch.stack(target)
    }

def createDataloaders(train_df, val_df, test_df, tokenizer):
    #Dataset de entrenamiento
    training_data = PeriodisticDataset(
        train_df.text.to_numpy(),
        train_df.TAG_num.to_numpy(),
        tokenizer
    )

    #Dataset de validacion
    val_data = PeriodisticDataset(
        val_df.text.to_numpy(),
        val_df.TAG_num.to_numpy(),
        tokenizer
    )

    #Dataset de test
    test_data = PeriodisticDataset(
        test_df.text.to_numpy(),
        test_df.TAG_num.to_numpy(),
        tokenizer
    )

    #Dataloader de entrenamiento
    train_loader = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    #Dataloader de validacion
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    #Dataloader de test
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader