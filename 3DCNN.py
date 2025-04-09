import os
import rasterio
import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from thop import profile
import torchsummary
from shapely.geometry import Point
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score

# Set the data path
data_path = r"C:\Users\Eric\Desktop\thesis\0612model\cnn3dmodel"
os.chdir(data_path)

def load_raster_images(raster_paths):
    raster_imgs = []
    for raster_path in raster_paths:
        try:
            with rasterio.open(raster_path) as src:
                raster_imgs.append(src.read())
        except Exception as e:
            print(f"Error loading {raster_path}: {e}")
    return np.array(raster_imgs)

# Function to extract raster values
def extract_raster_values(raster_paths, shapefile_path, class_column='GEWASGROEP'):
    features = []
    labels = []
    valid_indices = []
    try:
        with rasterio.open(raster_paths[0]) as src:
            raster_img = src.read()
            affine = src.transform
            raster_crs = src.crs
            raster_bounds = src.bounds
            print("Loading shapefile data from:", shapefile_path)
            print("Raster CRS:", raster_crs)
            print("Raster bounds:", raster_bounds)
            shape_data = gpd.read_file(shapefile_path)
            shape_data = shape_data.to_crs(rasterio.open(raster_paths[0]).crs)

        for index, row in shape_data.iterrows():
            if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                centroid = row.geometry.centroid
                point_coords = (centroid.x, centroid.y)
                feature_vector = []

                for raster_path in raster_paths:
                    with rasterio.open(raster_path) as src:
                        raster_bounds = src.bounds
                        if (raster_bounds.left <= point_coords[0] <= raster_bounds.right) and \
                           (raster_bounds.bottom <= point_coords[1] <= raster_bounds.top):
                            val = np.array([v for v in src.sample([point_coords])]).squeeze()
                            feature_vector.append(val)
                        else:
                            print(f"Centroid {point_coords} is out of raster bounds.")
                            feature_vector.append(np.zeros(src.count))  # Append zero vector if out of bounds

                features.append(np.stack(feature_vector, axis=0))  # Stack along the first dimension (time steps)
                labels.append(row[class_column])
                valid_indices.append(index)

        features = np.array(features)
        labels = np.array(labels)
        print("Number of data points:", len(features))
        print("Unique class labels:", np.unique(labels))
        if len(features) > 0:
            print("Shape of a single feature set (time_num, band_num):", features[0].shape)
        return features, labels, valid_indices
    except Exception as e:
        print("Error in extracting raster values:", e)
        return np.array(features), np.array(labels), valid_indices

# Dataset class
class SatelliteDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # Expected shape based on your input: (band_num, time_num, height, width)
        # Assuming the height and width are 1 for the current data
        feature = feature.reshape((feature.shape[1], feature.shape[0], 1, 1))

        if self.transform:
            feature = self.transform(feature)

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 3D CNN model
class CNN3D(nn.Module):
    def __init__(self, time_num, band_num, class_num):
        super(CNN3D, self).__init__()
        self.time_num = time_num
        self.band_num = band_num
        self.class_num = class_num
        channels = [32, 64, 128]

        self.Tlayer1 = nn.Sequential(
            nn.Conv3d(self.band_num, channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d((self.time_num, 1, 1))
        )

        self.Tlayer2 = nn.Sequential(
            nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d((self.time_num, 1, 1))
        )

        self.Tlayer3 = nn.Sequential(
            nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d((self.time_num, 1, 1))
        )

        self.FC = nn.Sequential(
            nn.Linear(channels[2]* self.time_num * 1* 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.class_num),
        )

    def forward(self, x):
        x = self.Tlayer1(x)
        #print(f"After Tlayer1: {x.shape}")
        x = self.Tlayer2(x)
        #print(f"After Tlayer2: {x.shape}")
        x = self.Tlayer3(x)
        #print(f"After Tlayer3: {x.shape}")
        x = x.view(x.size(0), -1)
        #print(f"After flattening: {x.shape}")
        x = self.FC(x)
        return x

def train_model(model, train_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    print("Training complete")

def evaluate_model(model, data_loader):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.cuda()
            labels = labels.cuda()
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    precision = precision_score(true_labels, preds, average='weighted')
    recall = recall_score(true_labels, preds, average='weighted')
    f1 = f1_score(true_labels, preds, average='weighted')
    return np.array(preds), np.array(true_labels), precision, recall, f1

def save_predictions_to_shapefile(predictions, shapefile_path, output_path, valid_indices):
    shape_data = gpd.read_file(shapefile_path)
    shape_data = shape_data.iloc[valid_indices]  # Filter valid indices
    if len(predictions) != len(shape_data):
        raise ValueError(f"Length of predictions ({len(predictions)}) does not match length of filtered shape_data ({len(shape_data)}).")
    shape_data['predictions'] = predictions
    shape_data.to_file(output_path, driver='ESRI Shapefile')
    print(f"Predictions saved to {output_path}")

def generate_accuracy_assessment(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    oa = accuracy_score(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    return cm, oa, kappa

if __name__ == '__main__':
    time_num = 3
    band_num = 230
    class_num = 8
    model = CNN3D(time_num, band_num, class_num).cuda()
    input_tensor = torch.randn(1, band_num, time_num, 1, 1).cuda()
    flops, params = profile(model, inputs=(input_tensor,))
    model.cuda()
    torchsummary.summary(model, (band_num, time_num, 1, 1), 1)
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))

    # Paths to the three satellite images
    raster_paths = ['202204.tif', '202207.tif', '202208.tif']

    # Extract features and labels for training data
    train_features, train_labels, train_valid_indices = extract_raster_values(raster_paths, 'trainingsamples.shp')
    print(f"Extracted {len(train_features)} training features. Shape: {train_features.shape}")
    # Extract features and labels for test data
    test_features, test_labels, test_valid_indices = extract_raster_values(raster_paths, 'validationsamples.shp')
    print(f"Extracted {len(test_features)} test features. Shape: {test_features.shape}")

    # Debugging statements to print the number of features and labels
    print(f"Extracted {len(train_features)} training features and {len(train_labels)} training labels.")
    print(f"Extracted {len(test_features)} test features and {len(test_labels)} test labels.")

    # Check if extracted features and labels are not empty
    if len(train_features) == 0 or len(train_labels) == 0:
        raise ValueError("Training features or labels are empty. Please check the data extraction step.")
    if len(test_features) == 0 or len(test_labels) == 0:
        raise ValueError("Test features or labels are empty. Please check the data extraction step.")

    # Create a label map to convert string labels to integers
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label mapping:", label_map)

    # Convert string labels to integer labels using the label map
    train_labels = np.array([label_map[label] for label in train_labels], dtype=np.int64)
    test_labels = np.array([label_map[label] for label in test_labels], dtype=np.int64)

    # Verify that labels are integers
    assert train_labels.dtype == np.int64, "Train labels are not integers"
    assert test_labels.dtype == np.int64, "Test labels are not integers"

    # Verify that features and labels are extracted correctly
    print("Train features shape:", train_features.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test features shape:", test_features.shape)
    print("Test labels shape:", test_labels.shape)

    # Create datasets
    train_dataset = SatelliteDataset(train_features, train_labels)
    test_dataset = SatelliteDataset(test_features, test_labels)

    print("Length of train_dataset:", len(train_dataset))
    print("Length of test_dataset:", len(test_dataset))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Train the model
    train_model(model, train_loader, num_epochs=300)

    # Generate predictions for the training dataset
    train_preds, train_true_labels, train_precision, train_recall, train_f1 = evaluate_model(model, train_loader)
    print(f"Length of train_preds: {len(train_preds)}")
    print(f"Length of train_true_labels: {len(train_true_labels)}")
    confusion_mat_train, overall_accuracy_train, kappa_coeff_train = generate_accuracy_assessment(train_true_labels, train_preds)

    # Print accuracy assessment results for training data
    print("Training Data - Confusion Matrix:")
    print(confusion_mat_train)
    print(f"Training Data - Overall Accuracy: {overall_accuracy_train:.4f}")
    print(f"Training Data - Kappa Coefficient: {kappa_coeff_train:.4f}")
    # Print precision, recall, and F1-score for training data
    print(f"Training Data - Precision: {train_precision:.4f}")
    print(f"Training Data - Recall: {train_recall:.4f}")
    print(f"Training Data - F1-Score: {train_f1:.4f}")

    # Save training predictions to shapefile
    # train_output_path = r"C:/Users/Eric/Desktop/thesis/0612model/cnn3dmodel/Predictions_train.shp"
    # save_predictions_to_shapefile(train_preds, "trainingsamples.shp", train_output_path, train_valid_indices)

    # Generate predictions for the test dataset
    test_preds, test_true_labels, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader)
    print(f"Length of test_preds: {len(test_preds)}")
    print(f"Length of test_true_labels: {len(test_true_labels)}")
    confusion_mat_test, overall_accuracy_test, kappa_coeff_test = generate_accuracy_assessment(test_true_labels, test_preds)

    # Print accuracy assessment results for test data
    print("Test Data - Confusion Matrix:")
    print(confusion_mat_test)
    print(f"Test Data - Overall Accuracy: {overall_accuracy_test:.4f}")
    print(f"Test Data - Kappa Coefficient: {kappa_coeff_test:.4f}")
    # Print precision, recall, and F1-score for test data
    print(f"Test Data - Precision: {test_precision:.4f}")
    print(f"Test Data - Recall: {test_recall:.4f}")
    print(f"Test Data - F1-Score: {test_f1:.4f}")

    # Save test predictions to shapefile
    # test_output_path = r"C:/Users/Eric/Desktop/thesis/0612model/cnn3dmodel/Predictions_test.shp"
    # save_predictions_to_shapefile(test_preds, "validationsamples.shp", test_output_path, test_valid_indices)