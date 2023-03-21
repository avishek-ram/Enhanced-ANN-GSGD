# Enhanced-ANN-GSGD
#use branch: pytorch_version

Read Me for ANN-GSGD
Paper: A Guided Neural Network Approach to Predict Early Readmission of Diabetic Patients
Programmer: Avishek Anishkar Ram
Supervisor: Dr. Anuraganand Sharma

The ANN-GSGD code has been written and tested with Python 3.10.6

Steps to setup and run  ANN-GSGD 
1.	Install Conda Package Manager
2.	Create a new conda environment using the file “environment.yml” available in the directory, the following command can be used for this “conda env create -f environment.yml” 
3.	The main program is written in main.py. Run this program.
4.	A popup opens where you have to select the data file, which is available in the repository directory inside the “data” folder.
5.	Now, the program will start running, It will train both the guided and canonical models and present the results on the performance of both the models based on the validation dataset.

Different Datasets can also be evaluated with this code as long as they comply with the existing data file formats. The model parameters can be updated for different datasets at lines 31-45 in main.py.

Actual ANN-GSGD implementation starts from lines 106 to 233.

The following repositories were referenced to aid the development of the ANN-GSGD algorithm:
https://github.com/anuraganands/GSGD
https://github.com/anuraganands/GSGD-CNN
