* This folder exists because there are potential use cases for these files
* But they don't have any use for this competition and failed to show any potential

# Brief Description of files
## TrainClip
    Trains Clip Model based on the similarity predicted between the both by the model itself

    - In on sentence, it identifies how similar a given text, or image are
    - Uses Cosine Loss
    - ViT-L/14 Model

    Eg: Given an image and text, it uses .encode_image and .encode_text
    These functions give the 786 embeddings representations of Image and Text (the closer these both are to
    each other, the better the model will be able to predict the text of an image or vice versa)

## TrainGPT
    Takes the image features from the Clip model, gives it to the GPT2 and outputs a 384 prediction embeddings
    
    - Uses Cosine Loss
    - ViT-L/14 Model, GPT2Model