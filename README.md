# This is the tutorial focused on solving research questions regarding the melenoma lesions dataset.

## Research questions are split into three categories based on level of interest and skill.

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
clone [this](https://github.com/GalAvineri/ISIC-Archive-Downloader) repo in the same directory

## Step 3.
Go into the ISIC folder and run this command `python download_archive.py --images-dir ../sample_imgs --descs-dir ../sample_dscs -s --seg-dir ../sample_segs --seg-skill expert`

## Step 4.

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


