import re
import unicodedata
import torch
from Constants import CATEGORIES_DICT
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(input_string: str) -> str:
    '''
        Remove accents and special characters from a given string, leaving only characters from the Latin alphabet,
        punctuation marks, spaces, and common symbols.

        Parameters:
            input_string (str): The input string containing characters with accents and special characters.

        Returns:
            str: The input string with accents and special characters removed.
    '''
    # Replace accents with their non-accented versions
    normalized_string = unicodedata.normalize('NFKD', input_string)
    without_accents = ''.join([c for c in normalized_string if not unicodedata.combining(c)])

    # Remove characters that are not in the latin alphabet and its punctuation signs
    cleaned_string = re.sub(r'[^\w\s.,!?¿¡]', '', without_accents)

    return cleaned_string


def get_sentiment (model, sample_review, tokenizer):
    # get input ids and attention mask
    encoded_text = tokenizer(
        preprocess(sample_review),
        truncation=True,
        return_tensors = 'pt'
    )

    input_ids = encoded_text['input_ids'].to(DEVICE)
    attention_mask = encoded_text['attention_mask'].to(DEVICE)

    output = None
    with torch.no_grad():
        model.to(DEVICE)
        output = model(input_ids, attention_mask)

    _, prediction = torch.max(output, dim=1)
    pred_prods = F.softmax(output, dim=1).cpu().numpy().reshape(4)

    to_ret = {
        'NEGATIVO': pred_prods[0],
        'NEUTRO': pred_prods[1],
        'POSTIVO': pred_prods[2],
        #'IRONIA': pred_prods[3]
    }
    return to_ret, CATEGORIES_DICT[int(prediction)]