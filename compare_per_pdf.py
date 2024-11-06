import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

class PDFAnalyzer:
    def __init__(self, main_folder_path, max_segments=1000, segment_size=(1000, 1000), max_image_size=2000):
        self.pdf_paths = self.get_all_pdf_paths(main_folder_path)
        self.segmented_images = {}
        self.max_segments = max_segments
        self.segment_size = segment_size
        self.max_image_size = max_image_size

        # Load pre-trained model for image feature extraction
        self.cnn_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn_model.eval()
        
    def get_all_pdf_paths(self, main_folder_path):
        pdf_paths = []
        for root, _, files in os.walk(main_folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
        return pdf_paths

    def downscale_image(self, image):
        height, width = image.shape[:2]
        if max(height, width) > self.max_image_size:
            scale_factor = self.max_image_size / max(height, width)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def segment_images(self, image):
        height, width = image.shape[:2]
        segment_height, segment_width = self.segment_size
        segments = [
            image[y:y + segment_height, x:x + segment_width]
            for y in range(0, height, segment_height)
            for x in range(0, width, segment_width)
        ]
        return segments[:self.max_segments]  # Limit to max_segments

    def extract_image_features(self, image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            features = self.cnn_model(input_batch)
        return features.numpy().flatten()  # Flatten to 1D array for easier comparison

    def analyze_pdf(self):
        for pdf_path in self.pdf_paths:
            print(f"Processing {pdf_path}...")
            pdf_name = os.path.basename(pdf_path)
            images = convert_from_path(pdf_path)

            for page_num, image in enumerate(images, start=1):
                print(f"  Processing page {page_num} of {pdf_name}...")
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image_cv = self.downscale_image(image_cv)
                segments = self.segment_images(image_cv)
                print(f"    Total segments on page {page_num}: {len(segments)}")
                
                # Store page segments under the document name and page number
                if pdf_name not in self.segmented_images:
                    self.segmented_images[pdf_name] = {}
                self.segmented_images[pdf_name][page_num] = [self.extract_image_features(segment) for segment in segments]

    def compare_pdfs(self):
        pdf_names = list(self.segmented_images.keys())
        
        for pdf_name in pdf_names:
            # Compare each document with itself
            print(f"\nComparing pages within the same document: {pdf_name}")
            pages = list(self.segmented_images[pdf_name].keys())
            intra_similarity_scores = []

            for i in range(len(pages)):
                for j in range(i + 1, len(pages)):
                    page1, page2 = pages[i], pages[j]
                    features1, features2 = self.segmented_images[pdf_name][page1], self.segmented_images[pdf_name][page2]
                    
                    # Compare image features using cosine similarity
                    for f1 in features1:
                        for f2 in features2:
                            similarity = self.cosine_similarity(f1, f2)
                            intra_similarity_scores.append(similarity)
            
            # Calculate overall average similarity for the same document
            avg_intra_similarity = np.mean(intra_similarity_scores) if intra_similarity_scores else 0
            print(f"Overall Average Intra-document Similarity for {pdf_name}: {avg_intra_similarity:.4f}")

            # Compare with other documents
            for other_pdf_name in pdf_names:
                if pdf_name == other_pdf_name:
                    continue
                
                print(f"\nComparing {pdf_name} with {other_pdf_name}")
                inter_similarity_scores = []
                for page1_features in self.segmented_images[pdf_name].values():
                    for page2_features in self.segmented_images[other_pdf_name].values():
                        for f1 in page1_features:
                            for f2 in page2_features:
                                similarity = self.cosine_similarity(f1, f2)
                                inter_similarity_scores.append(similarity)
                
                # Calculate overall average similarity between documents
                avg_inter_similarity = np.mean(inter_similarity_scores) if inter_similarity_scores else 0
                print(f"Overall Average Inter-document Similarity between {pdf_name} and {other_pdf_name}: {avg_inter_similarity:.4f}")

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

# Example usage:
main_folder_path = r"C:\Users\Lenovo\Sample"  # Replace with your main folder path
analyzer = PDFAnalyzer(main_folder_path)
analyzer.analyze_pdf()
analyzer.compare_pdfs()
