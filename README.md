# AED
Source codes of our paper 'Automatic Emesis Detection (AED): a deep learning-based system for automatic detection of emesis with high accuracy in Suncus murinus'


# Automatic Emesis Detection Pipeline
1. Download and install miniconda [here](https://docs.anaconda.com/miniconda/).



2. Install required python packages by
   
    ```
       conda create -n aed python=3.9
       conda activate aed
       pip install -f requirements.txt
    ```
   
3. Prepare your own videos
   
    Please place your videos in the `place_your_video_in_this_folder` folder. You can also change the video path in the `detection_pipeline.py` script.

4. Download a trained model from this [link](https://drive.google.com/drive/folders/1dh1JiKJpmIjO-pvLNpcLN5GHUi6tg_Or?usp=sharing) and put it in the  `models` folder


5. Emesis detection automatically

    Run detection_pipeline.py script directly by

    ```python detection_pipeline.py```

    Before that, you could also change the configs/default.py file to change different model parameters.

6. Get the detected results. The prediction results will be stored in the `results` folder by default.

    