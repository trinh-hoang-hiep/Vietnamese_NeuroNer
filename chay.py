import os
os.system('neuroner --output_folder=./outputvietnam --token_pretrained_embedding_filepath=./data/word_vectors/glove.6B.100d.txt --parameters_filepath=./neuroner/trained_models/event/parameters.ini --train_model=False --use_pretrained_model=True --dataset_text_folder=./neuroner/data/du_doan --pretrained_model_folder=./neuroner/trained_models/event')
