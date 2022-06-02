# SWIF(r): SWeep Inference Framework (controlling for correlation)

SWIF(r) Version 2.1.1
Lauren Alpert Sugden, Kaileigh Ahlquist, Joseph Paik, Ramachandran Lab, Brown University

Publication: Sugden, Lauren Alpert, Elizabeth G. Atkinson, Annie P. Fischer, Stephen Rong, Brenna M. Henn, and Sohini Ramachandran. "Localization of adaptive variants in human genomes using averaged one-dependence estimation." Nature communications 9, no. 1 (2018): 703.

Ahlquist, K.D., Lauren Sugden, and Sohini Ramachandran. "Enabling interpretable machine learning for biological data with reliability scores." https://doi.org/10.1101/2022.02.18.481082

Questions: post a git issue or contact sugdenl@duq.edu, kaileigh_ahlquist@brown.edu, joseph_paik@alumni.brown.edu.

### Updates:  
This version of SWIF(r) is an update to SWIF(r) Version 1 released in 2017. The new updates to SWIF(r) include:

(1) Both Python2 and Python3 compatibility (tested with Python versions 2.7 <= x <= 3.7). 

(2) A convenient installation process for SWIF(r) via the pip package installer for Python (Python 3 only).

### Installation:

\> pip install swifr

To use SWIF(r) for Python 2.7, updated scripts found under SWIFr/swifr_pkg/ can be run according to the README at SWIFr/SWIFr_v1/README.md. SWIF(r) version 1 (as used in the original SWIF(r) publication) is also available under SWIFr/SWIFr_v1.

### Important Note for Training Examples:

**Training examples in example_2classes/ and example_3classes/ are based on a toy dataset. The classifiers learned in these examples should not be used for real applications. Instead, use swifr_train to train the classifier on a complete set of user-supplied training examples.**

If you wish to use the provided training examples to test SWIF(r), you can locate them inside the directory where the SWIF(r) package was downloaded on your local machine: **/path/to/swifr_pkg/test_data**.

### Training SWIF(r):

#### Usage:  
To train SWIF(r), run swifr_train with the --path flag pointing to the directory containing the input files.  

###### Example:  
\>swifr_train --path example_2classes/ 

#### Required Input Files:
Note: the following files and directories must be in a single directory, the path to which will be passed to swifr_train using the flag --path. The directory must contain classes.txt, component_stats.txt, and a directory called simulations. The simulations directory must have a subdirectory with the same name as each entry in classes.txt. Each subdirectory should have a set of files to be used for training that class. Example directories can be found in example_2classes/ and example_3classes/.  

File hierarchy:  

 >     classes.txt      
 >     component_stats.txt  
 >     simulations/  
 >          neutral/  
 >               simfile1.txt              
 >               simfile2.txt  
 >               ...  
 >          sweep/  
 >               simfile1.txt              
 >               simfile2.txt  
 >               ...


1. classes.txt - this file has one line for each class you wish to train.  

	>     neutral  
	>     sweep

2. component_stats.txt - this file has one line for each summary statistic. The same statistic names must be in the header of the training files below. The order of statistics does not matter.

	>     Fst  
	>     XP-EHH  
	>     iHS  
	>     DDAF

3. simfile.txt - tab-delimited file with one line per simulated SNP. File can have any number of columns for identification, but must have one column for each of the statistics in component_stats.txt. Header line should use the names of the statistics with the same spelling and capitalization as in component_stats.txt. The statistics need not be in the same order. **Please note: the classifier will be trained on every line in the file**, so make sure that files only contain SNPs that you wish to train with. For example, if you only want to train a sweep class on the actual swept site, the file must only contain that site. **Also note: the value -998 is used to denote missing values** for SNPs where the statistic of interest is undefined (or otherwise missing).

	>     SNP     position     Fst     XP-EHH     iHS     DDAF  
	>     SNP1    1            3.4     2.2        -3.4    1.3  
	>     SNP2    2            0.1     5.0        -998    -3.2  
	>     ...    

  


#### Command Line Options:
--path \<string\>: relative path to all input files  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Required (default: '')  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e.g. --path example_2classes/

--retrain \<bool\>: use to re-train classifier after manually altering joint_component_nums or marginal_component_nums  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(see Output)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Default: false  

#### Output:
The output of swifr_train creates three directories:  AODE_params/ which contains the parameter files that swifr_test will use, component_statistic_distributions/ which contains illustrations of the learned marginal (1D) and joint (2D) distributions, and BIC_plots/ which contains Bayesian Information Criterion curves (https://en.wikipedia.org/wiki/Bayesian_information_criterion) that SWIF(r) uses to learn the number of components for the gaussian mixtures for each of the 1- and 2-dimensional component statistic distributions. The BIC balances model complexity with fit, so that the optimal number of components is the one that minimizes the BIC. SWIF(r) automatically chooses the number of components between 1 and 5 that defines this minimum. The number chosen is highlighted in pink in the pdf files in BIC_plots/.  

If you wish to change the number of Gaussian mixture components based on the BIC curves (filenames in BIC_plots/ contain the statistic or pair of statistics, and the class), you can edit the file joint_component_nums and/or marginal_component_nums in the AODE_params/ directory, then re-run swifr_train with the flag --retrain. This will regenerate the files in AODE_params/, BIC_plots/, and component_statistic_distributions/.  


### Running SWIF(r) on testing data:
#### Usage:  
\>swifr_test

#### Command Line Options:
--path2trained \<string\>: relative path to the directory containing the input and output files from swifr_train (same as --path argument for swifr_train)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Required (default: '')  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e.g. --path2trained example_2classes/  

--pi \<pi_1 pi_2 ...\>: prior probabilities on the classes. Number of arguments must match the number of classes, must be in the same order as classes.txt, and must add to 1.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Default: 0.99999 0.00001  

--interactive \<bool\>: use to calculate posterior probabilities for a single site. You will be prompted for values for each of the component statistics, and the output will be a posterior probability for each class.  

--file \<string\>: use to calculate SWIF(r) on multiple sites, where argument is the filename (including relative path). File must be in the same tab-delimited format as simfile.txt above; any number of columns are allowed, as long as there is a column for each component statistic, and a header line that uses the names of the statistics exactly as they appear in component_stats.txt. Again, use -998 to denote any missing values. swifr_test will output a file with all of the columns from the original, plus n columns that record the prior value given upon input for each class, and n columns for the posterior probabilities for each class.  

--stats2use \<stat1 stat2 ...\>: use if you want to run SWIF(r) with a subset of the statistics that were used to train. Make sure that statistic names listed match those used for training. Can be combined with either --interactive or --file. 

--outfile \<string\>: specify file for output (including path). By default, output will be written to the same directory as the original file, with name \<filename\>_classified. Only use with --file.

**Note: either --interactive or --file must be specified.**  

#### Examples:  

Interactive example:  
 >swifr_test --path2trained example_2classes/ --interactive  
 >Value for Fst: 2.3  
 >Value for XP-EHH: 3.3  
 >Value for iHS: -2.5  
 >Value for DDAF: 3.6  
 >Probability of neutral: 0.860299223365  
 >Probability of sweep: 0.139700776635  

application_example/ in this repository has classified files generated from the following commands:  

\> swifr_test --path2trained example_3classes/ --pi 0.99998 0.00001 0.00001 --file application_example/test_block_3classes  

\> swifr_test --path2trained example_2classes/ --pi 0.99999 0.00001 --file application_example/test_block_2classes  


### Calibrating Probabilities:
#### Usage:  
\> calibration

#### Command Line Options:
--frac1 \<float\>: for calibration purposes, fraction of data points that come from class 1 (sweep).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Required (default: 0.1)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e.g. --frac1 0.001  

--train \<bool\>: use to learn calibration function from training data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Requires --input_train  

--input_train \<string\>: file containing training data for calibration. Two tab-delimited columns, no header, each line is a datapoint. First column contains labels (e.g. 0 for neutral, 1 for sweep) and second column contains raw (un-calibrated) probabilities.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Required for --train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e.g. see calibration/example/test_for_calibration.txt  

--apply \<bool\>: use to apply calibration learned with --train to a new set of probabilities.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Requires --input_apply  

--input_apply \<string\>: file containing probabilities to be calibrated. Can have any number of tab-delimited columns, with a header line. One column name must be "uncalibrated" and that column must contain the probabilities to be calibrated. Other columns may have rsids, chromosome numbers, etc...    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Required for --apply  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e.g. see calibration/example/test_for_application  

**Note: Either --train or --apply must be specified. Currently, calibration functionality only supports binary classification probabilities.**  

#### Examples:  

\>calibration --frac1 0.1 --train --input_train calibration/example/test_for_calibration.txt  

\>calibration --frac1 0.1 --apply --input_apply calibration/example/test_for_application  

#### Output:  
calibration will create a directory called calibration/ if one does not already exist. With the --train flag, calibration will draw two reliability plots (before and after calibration) and will save them in pdf format in the calibration/ directory. It will also save .p files that contain the values necessary for applying smoothed isotonic regression calibration to a new set of datapoints. With the --apply flag, calibration will create a new file in the calibration/ directory with the same name as the file provided with --input_apply, appended with "_calibrated". This file will contain all of the columns in the original file, plus a final column with column name "calibrated" that provides the new calibrated probabilities for each raw probability. For examples of all of these, see files in calibration/example/.
