from preprocess import load_annotations
from alignment import align_image_caption
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random

# Load annotations
annotation_file = './data/annotations/captions_train2014.json'
image_id_to_captions = load_annotations(annotation_file)

# Function to create negative samples by shuffling captions
def create_negative_samples(image_id_to_captions, num_negatives=5):
    """Create negative samples by shuffling captions across images."""
    all_image_ids = list(image_id_to_captions.keys())
    negative_samples = []

    for _ in range(num_negatives):
        # Randomly select two different images
        image_id_1, image_id_2 = random.sample(all_image_ids, 2)
        # Take a caption from image 2 and pair it with image 1
        random_caption = random.choice(image_id_to_captions[image_id_2])
        negative_samples.append((image_id_1, random_caption))
    
    return negative_samples


# Store similarity scores and results
similarity_scores = []
y_true = []
y_pred = []

# Step 1: Process Aligned (Correct) Samples
sample_image_ids = list(image_id_to_captions.keys())[:10]  # Use the first 10 images as a sample
for image_id in sample_image_ids:
    results = align_image_caption(image_id, image_id_to_captions, threshold=0.04)
    for result in results:
        # Collect similarity scores, true labels, and predicted labels
        similarity_scores.append(result['similarity'])
        y_true.append(result['actual'])
        y_pred.append(result['predicted'])

# Step 2: Generate and Process Negative (Misaligned) Samples
negative_samples = create_negative_samples(image_id_to_captions, num_negatives=10)
for image_id, incorrect_caption in negative_samples:
    # Wrap the incorrect caption in a dictionary for processing
    results = align_image_caption(image_id, {image_id: [incorrect_caption]}, threshold=0.04, actual=0)
    for result in results:
        print(f"Misaligned - Image ID: {image_id}, Caption: {incorrect_caption}, Similarity: {result['similarity']}, Predicted: {result['predicted']}")
        similarity_scores.append(result['similarity'])
        y_true.append(result['actual'])
        y_pred.append(result['predicted'])

print("\nDebug: Lengths of `y_true` and `y_pred`")
print(f"Length of y_true: {len(y_true)}")
print(f"Length of y_pred: {len(y_pred)}")
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
# Calculate performance metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("\nPerformance Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[1,0])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Aligned', 'Predicted Not Aligned'],
            yticklabels=['Actual Aligned', 'Actual Not Aligned'])
plt.title('Confusion Matrix')
plt.show()

# Bar Chart: Mean Similarity Scores
average_correct = np.mean([s for s, t in zip(similarity_scores, y_true) if t == 1])
average_incorrect = np.mean([s for s, t in zip(similarity_scores, y_true) if t == 0])

fig, ax = plt.subplots()
labels = ['Correct Alignments', 'Incorrect Alignments']
means = [average_correct, average_incorrect]
ax.bar(labels, means)
ax.set_ylabel('Mean Similarity Score')
ax.set_title('Comparison of Mean Similarity Scores')
plt.show()