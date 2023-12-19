# Util functions
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def embed_sentence(sentence, model="paraphrase-MiniLM-L6-v2"):
    # From https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

    return sentence_embedding




if __name__ == '__main__':


    sentence = ['This is an example sentence']

    print(embed_sentence(sentence))