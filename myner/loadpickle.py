# Load the dictionary back from the pickle file.
import pickle
import os
# favorite_color = pickle.load( open( "dataset.pickle", "rb" ) )
# print(favorite_color)

# objects = []
# with (open("dataset.pickle", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

# print(objects)

#from prepare_pretraind_model.py
model_folder='G:/project2/thamkhao/NeuroNER-master (1)/NeuroNER_Vietnamese/myner'
dataset_filepath = os.path.join(model_folder, 'dataset.pickle')
dataset = pickle.load(open(dataset_filepath, 'rb'))
# print(dataset.__dict__)
print(list(dataset.__dict__.keys()))