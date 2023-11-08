# CS330_course_project

## Getting started

* We use `Meta-World` as the simulator. The official one is under development. In case official version changes and potential custom modification, install it using our forked repo:

  ```bash
  conda create -n MetaWorld python=3.8
  conda activate MetaWorld
  cd 'where you want to put the Meta-World'
  git clone git@github.com:Jianhao-zheng/Metaworld.git
  cd Metaworld
  pip install -e .
  ```

To run the basic script `random_sample.py`, you need to install opencv lib:

  ```bash
  pip install opencv-python
  ```

* Code structure

  ```bash
  ├── data_collection                  # Code to collect data autonomously
  │   └── random_sample.py             # Collect data randomly
  ├── train                            # Code to train the model
  └── eval                             # Code to evaluate the model
  ```

### Run random data collector

* You can parse "--visual" to visualize the current situation of the robot:

  ```bash
  python ./data_collection/random_sample.py --visual
  ```

In default, the data won't be saved. If you want to save the data, please take a look at the arguments in `random_sample.py`