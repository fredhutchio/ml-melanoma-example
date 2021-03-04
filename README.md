# This is the tutorial focused on solving research questions regarding the melenoma lesions dataset.

## Research questions are split into three categories based on level of interest and skill.

The research questions are isolated but build upon each other in progressively "harder" modules. For instance, the first questions "path" deals with two variables: lesion appearance and diagnosis. Hence, the beginner version of Q1 correlates color and malignancy in basic plotting terms, while the intermediate version applies regression and testing to compare the same variables. At the expert level, appearance is fed into a neural network to predict the diagnosis. The same pattern applies to the remaining questions. 

### Beginner (basic stats and visualization):
1. Does coloration (as simply as can be extracted) correlate with malignancy somehow? 
2. Whether malignant lesions occur more along some gender or age line? 
3. What is the essential frequency distribution of lesion diagnosis? Is it skewed (overall and within categories)? 
Potential techniques to apply: scatter plots, bar charts, etc.

### Intermediate (conditional probability and A/B Testing): 
1. Some sort of A/B testing or difference-between-groups testing after categorizing by coloration of image
2. Risk prediction based on demographic (coming from the joint distribution)?
3. Clustering/Categorization with only part of the data

### Expert: 
1. Is a sample Benign or malignant (image classification)?
Note: we may add preprocessing steps like spatial transformer/zoom operation/color processing
2. Can you predict the gender/age of a person from a sample (multiple linear regression)?
3. Clustering/categorization that hopefully lines up with their lesion diagnosis or the lesion diagnosis frequency chart from the beginner section. 

# If you want to run these notebooks yourself follow these steps:

## Step 1.
clone this repo

## Step 2. 
clone [this](https://github.com/GalAvineri/ISIC-Archive-Downloader) repo in the same directory (i.e., as a sibling directory to this repository). 

## Step 3.

__Warning!__ this command downloads the entire set and will take a very, very long time. In the notebooks, examples work on a tiny sample thereof (instructions are provided in each notebook). You can add the `--num_images=X` flag in the below command to limit the downloaded data points to however many/few you want. 

Go into the ISIC folder and run this command 
`python download_archive.py --images-dir ../sample_imgs --descs-dir ../sample_dscs -s --seg-dir ../sample_segs --seg-skill expert`

## Step 4.
Ensure your environment has the following package dependencies installed (you can use `pip install` or `conda`):

```
opencv
pandas
numpy
scikit-learn
tensorflow
pillow
matplotlib
seaborn
statsmodels
fire
```

## Step 5.

If you are using Fred Hutch computing resources, log into `rhino` and do the following to access all necessary packages:

```
grabnode
ml Python/3.7.4-foss-2019b-fh1
pip install -r requirements.txt
```

## Step 5. 

To run Jupyter notebooks in `rhino` specifically, make sure you are logged in to the Fred Hutch VPN, and paste the link resulting from this command:
Note: for any tasks requiring GPUs, make sure you specify "y" when asked for GPUs in the `grabnode` command. 

```
jupyterlab
```


