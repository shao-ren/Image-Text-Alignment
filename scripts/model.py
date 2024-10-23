from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
import torch
import torch.nn as nn

# Load pre-trained models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer

# Projection layer to align image embedding dimensions (2048 -> 768)
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# Initialize projection layer (ResNet output is 2048, we reduce it to 768)
image_projection = ProjectionLayer(input_dim=2048, output_dim=768)

def extract_text_features(caption):
    """Extract BERT embeddings for a given caption."""
    inputs = tokenizer(caption, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def extract_image_features(image):
    """Extract ResNet embeddings for a given image and project them to match text embedding dimensions."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        resnet_embedding = resnet(image_tensor).squeeze()  # 2048-dimensional embedding
        projected_embedding = image_projection(resnet_embedding)  # Project to 768 dimensions
        return projected_embedding