from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_image
from model import extract_text_features, extract_image_features

def calculate_similarity(image_embedding, caption_embedding):
    """Calculate cosine similarity between image and caption embeddings."""
    return cosine_similarity(image_embedding.detach().numpy().reshape(1, -1), 
                             caption_embedding.detach().numpy().reshape(1, -1))[0][0]

def align_image_caption(image_id, image_id_to_captions, threshold=0.5, actual=1):
    """Align an image with its captions and calculate similarity."""
    image = load_image(image_id)
    image_embedding = extract_image_features(image)
    results = []

    for caption in image_id_to_captions[image_id]:
        caption_embedding = extract_text_features(caption)
        similarity_score = calculate_similarity(image_embedding, caption_embedding)

        # Determine predicted label based on threshold
        predicted_label = 1 if similarity_score >= threshold else 0
        
        # Store the similarity score, actual label (1 for correct alignment), and prediction
        results.append({
            'caption': caption,
            'similarity': similarity_score,
            'actual': actual,  
            'predicted': predicted_label
        })
        print(f'Image ID: {image_id}, Caption: {caption}, Similarity: {similarity_score}, Predicted: {predicted_label}')
    
    return results