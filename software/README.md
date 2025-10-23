## introduce
This project aims to obtain road sensor data from the digitraffic API, identify the dryness and wetness of the road, and save the results to a local file. By analyzing this sensor data, traffic management departments can better understand road conditions, thereby improving traffic safety and efficiency.
Currently in the testing phase.

## Directory structure and description
- [code](code) The directory contains the main code files for the project.
  - [get_pic_from_digitraffic_to_local](code%2Fget_pic_from_digitraffic_to_local) This directory contains scripts for obtaining road sensor data from the digitraffic API and saving it locally.
      - [local_dry_wet_file.py](code%2Fget_pic_from_digitraffic_to_local%2Flocal_dry_wet_file.py)
        Run directly to get data and save it locally
  - [model_test](code%2Fmodel_test) The directory contains scripts for testing and verifying models. You can select different models for testing according to the model switching prompts in the script.
    - [resnetV2_50_test.py](code%2Fmodel_test%2FresnetV2_50_test.py) The ResNetV2-50 model is used to evaluate the model metric.
    - [resnetV2_50_test_single_img.py](code%2Fmodel_test%2FresnetV2_50_test_single_img.py) Use the ResNetV2-50 model to make predictions for a single image.
  - [dataset](dataset) The directory contains the datasets for training and testing.
  - [models](models) The directory contains the trained model files, rar files need to be extracted before use.
  - [rejected_images](rejected_images) The directory contains image files that were rejected by the model.
  - [single_image_test](single_image_test) Directory containing image files for a single test.
  - Currently only supported image format is JPG