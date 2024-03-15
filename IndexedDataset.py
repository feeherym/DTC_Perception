import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

#custom dataloader so you don't have to use ImageFolder
#just put in the class name and indeces

class IndexedDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
            img_dir (string): directory with all the images
            transform: transform to be applied
        """
        self.img_dir = img_dir
        self.transform = transform
        self.indexes_to_classes = {}
        self.index_class_pairs = []

        # Load image paths
        self.img_paths = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])

    def add_class_indexes(self, class_name, indexes):
        """
        Add class indexes to the dataset.
            class_name (str): Class name
            indexes (list): List of image indexes
        """
        if class_name in self.indexes_to_classes:
            print(f"Warning: Class '{class_name}' already exists. Updating indexes.")
        self.indexes_to_classes[class_name] = indexes
        self.index_class_pairs.extend([(idx, class_name) for idx in indexes])
        self.index_class_pairs.sort()  # Ensure the ordering is consistent

    def __len__(self):
        return len(self.index_class_pairs)

    def __getitem__(self, idx):
        idx, class_name = self.index_class_pairs[idx]
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, class_name
    
#example usage
# dataset = IndexedDataset(img_dir=img_dir, transform=transform)

# class1_indexes = [0, 1, 2]
# class2_indexes = [3, 4, 5]

# dataset.add_class_indexes('class1', class1_indexes)
# dataset.add_class_indexes('class2', class2_indexes)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
#how to get the indexes from the data
# file_path = r"C:\Users\mgfee\OneDrive - Villanova University\Desktop\Sping 2024\Research\D01_G1_S1_crops\D01_G1_S1_vitals.npy"
# loaded_data = np.load(file_path, allow_pickle=True)

# #C3 (blue shirt manikin) is index 1
# print(loaded_data[1]['CasualtyID'])

# print(loaded_data[1]['Info'])
# # print(loaded_data[0]['Data'][0])

# amp_r_leg_col = loaded_data[0]['Info'].index(' Amp R Leg')
# elapsed_time_col = loaded_data[0]['Info'].index('Elapsed Time (ms)')

# matching_values = []
# for row in loaded_data[1]['Data']:
#     if row[amp_r_leg_col] != 0:
#         matching_values.append((row[amp_r_leg_col], row[elapsed_time_col]))

# image_directory = r"C:\Users\mgfee\OneDrive - Villanova University\Desktop\Sping 2024\Research\D01_G1_S1_crops"

# image_indexes = []
# for index, filename in enumerate(os.listdir(image_directory)):
#     if filename.endswith('.jpg'):
#         image_index = int(filename.split('.')[0])
#         if image_index in matching_values:
#             image_indexes.append(index)