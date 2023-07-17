# Defect detection and classification with ZSL and FSL
This Repository contains
-	Zero shot learning image classification training and testing details
-	Few shot learning image classification training and testing details
-	And general detection code for zero shot learning and few shot learning.

Zero Shot Learning:
Here I have used MvTec Ads Carpet defected images. And check “zsl_reports” document for detailed information

Few Shot Learning:
Here I have used MvTec Ads Transistor defected images. And check “fsl_reports” document for detailed information

The General Detection 
General detection code will give the results for, find the defect in carpet images with zsl model and find the defect in transistor with fsl model.

To detect the carpet defects use command-
“python detection.py –input_image image_path –model zsl”

To detect the transistor defects use command-
“python detection.py –input_image image_path –model fsl”


