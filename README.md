# Development of pre-processing pipeline &amp; Feature engineering for dexterity assessment

## Description
This project develops a pre-processing pipeline of 3D body points as such : <a href=https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose><img src="https://user-images.githubusercontent.com/13503330/245036409-2417e4f7-2203-468f-bad0-e7a6a6bf8251.jpg"  width="150" height="300">

These 3D body points are generated from participants during their ratatouille cooking session.

Feature engineering is also applied on these 3D body points, for dexterity assessment between forearm amputees and able-bodied people.

The following primary features are generated : 
- 3D body points transformed into body coordinates
- Angle of shoulder flexion and rotation
- Naive angles of elbows, shoulders (naive meaning it is simply the angle between body segments, without projection on a body plane)
- Angle of trunk forward/lateral flexion
- Centroids of each arm, all arms together and the trunk
- Instantaneous velocities and accelerations of all body points and centroids (separated into the 3 components x, y, z, and not separated)
- Instantaneous angular velocities and accelerations of all joint (same as above)
- Filtered velocities and accelerations of all above

More details on these primary features in the file `src/feature.ipynb`.

From these primary features, secondary features are generated :
- Range of Motion 
- Mean
- Standard deviation

On all previous primary features, for each of these events during each ratatouille cooking session : 

- Chop zucchini
- Chop eggplant
- Chop mushrooms
- Chop peppers
- Chop tomatoes
- Reach to things
- Transfer things to containers
- Stirring
- Seasoning
- Washing hands
- Doing dishes

Then, some statistical comparison is done between able-bodied and amputees people.

In the future, one should try to better define the joint angles (limitations are presented in `src/feature.ipynb`). 
One could also expand the feature engineering, such as adding some tertiary features (e.g. number of cuts)

## Project organization
This project is organized as follows :

- the repository **data** that includes :
    - **data_test_31.07.npy**, small a dataset used for primary features validation
    - the repository **ESK_data** that contains :
        - **body_kpts_{id}.npy** the generated 3D body points of each participants
        - **{id}.npy** The labelled events throughout the session of each participant
    - the repository **features** that contains :
        - **primary_features_amputees.pkl** that contains the primary features of amputees participants
        - **primary_features_healthy.pkl** that contains the primary features of able_bodies participants
        - **primary_features_{id}.pkl** that contains the primary features of a participant
        - **secondary_features_amputees.pkl** that contains the secondary features of amputees participants
        - **secondary_features_healthy.pkl** that contains the secondary features of healthy participants
        - **secondary_features_{id}.pkl** that contains the secondary features of a participant
    - the repository **figures** that contains a few figures for statistical comparison between amputees and able-bodied people, and also : 
        - the repository **mean** that contains the plot of the means of each primary features of each participant
        - the repository **ROM** that contains the plot of the ROMs of each primary features of each participant
        - the repository **std** that contains the plot of the standard deviation of each primary features of each participant
- the repository **src** that includes : 
    - **feature.ipynb** that pre-processes the dataset **data_test_31.07.npy** and validate the primary features
    - **create_features.ipynb** that produces the datasets of the repository **features** 
    - **stats.ipynb** that produces some plots for statistical comparison, that can be found in repository **figures**
    - **utils.py** that contains the pipeline, feature engineering and helper functions
    - the repository **videos** that contains : 
        - **Boutput0.mp4** which is the video that corresponds to the **data_test_31.07.npy**
        - And other videos of the skeleton produced for validation by **feature.ipynb**

## How to use the project

Just make sure to have the libraries mentioned below installed on your environment before running the cells in the jupyter notebook.
To pull the data files, please use lfs by typing `git lfs pull`.

## Libraries
In this project the following libraries were used : 
- matplotlib
- pandas
- numpy
- scipy
- seaborn
- cv2
- skspatial
- [lfs](https://git-lfs.com/)