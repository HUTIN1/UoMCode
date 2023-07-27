# Description of the main roal of each file in this folder

**training** is to launch the training allowing the neural network to learn. \
**net** contains :
 - neural network used during the training
 - all the steps doing during the training (training step, validation, test)
 - rendering \
MonaiUNetHRes have been used to do the training for the final method.

**dataset** contains 3 dataset classes:
- TeethDatasetLm have been used to train a neural network to detect landmarks on palate
- TeethDatasetLmCoss have been used to use cosine similarity loss
- TeethDatasetPatch have been used to train to detect the region of reference on palate 
TeethDatasetPatch have been used to do the training for the final method.

**callback** allows display images in web browser.
To open the tab with the progress and images of the training write this command in terminal : 
> tensorboard --logdir=lightning_logs/ 

[link](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)


**display** contains a script to display the rendering happen in training, and allows to make picture for poster and paper.

**prediction_seg** predicts the region of reference on palate. 

**prediction_cosine** prediction the position of the landmark on the palate from training using loss CosineSimilarity

**post_process** contains functions for the post processing the creation of the patch on the palate.
The post process is necessare to fill the hole in segmentaiton of the patch.

**utils** contains little functions util for different step of the training or the prediction. 

**ManageClass** contains transformation usefull for data augmentation.

**icp** contains funcitons to orient surfaces.