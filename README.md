# An Open-Source General Purpose Machine Learning Framework for Individual Animal Re-Identification Using Few-Shot Learning
Code used to implement the triplet-loss learning architecture as first presented in: https://doi.org/10.1111/2041-210X.14278

# Install
 1. Install [Anaconda](https://www.anaconda.com/download/success).
 2. Open the Anaconda prompt.
 3. (recommended) Create a new environment for your program.

        conda create -n reid
 4. Use anaconda to install python 3.10 and pip

        conda install python==3.10 pip

 6. Download this project onto your computer.
 7. Navigate to that folder in the Anaconda prompt.
 8. Install all of the required programs.

        pip install -r requirements.txt

# Run
 1. Place all images of the same species (the dataset) in a folder with the following structure:
     - animal_1
       - photo_1_1.jpg
       - photo_1_2.jpg
       - ...
       - photo_1_N.jpg
     - animal_2
       - photo_2_1.jpg
       - photo_2_2.jpg
       - ...
       - photo_2_N.jpg
     - ...
     - animal_N
       - photo_N_1.jpg
       - photo_N_2.jpg
       - ...
       - photo_N_N.jpg
 2. Run the python program, providing as arguments
     - the folder with the images and
     - the desired output destination.

    If you downloaded the sea star image dataset and placed the unzipped folder with the images in the same directory as the program, this might look like this:
    ````
    python experiment.py --dataset=sea-star-re-id --output=matchOutput.zip
    ````
    
## Data Availability
The sea star image datasets are available to download at https://lila.science/sea-star-re-id-2023/.
