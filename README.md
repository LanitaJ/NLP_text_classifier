# Task

You need to predict contact information in the ad.

For training, you have the following fields:
* `title`,
* `description`,
* `subcategory`,
* `category`,
* `price`,
* `region`,
* `city`,
* `datetime_submitted`.

Target: `is_bad`

You have 2 datasets `train.csv` and `val.csv`: `train.csv` contains more data, but data markup is worse; in `val.csv` there is much less data, but data markup is better.
Test dataset for checking your solution will look like `val.csv`.
Pay attention that dataset can contain incorrect marks.

All data here: https://drive.google.com/drive/folders/1anZ1bxi5WhPmBlCBnYBYzo4foSgGSee5?usp=sharing. 
Your solution file with test data will be placed at `/task-for-hiring-data/test_data.csv` during launching

The file with the result of the model should be csv-file with columns:
* `index`: `int`, the number of the entry in file;
* `prediction`: `float` from 0 to 1, probability that ad contains contact information.

|index  |prediction|
|-------|----------|
|0|0.12|
|1|0.95|
|...|...|
|N|0.68|

After working out the script `run.py` a csv file should be written `/task-for-hearing-data/target_prediction.csv`.

As a metric of the quality of your model's performance, we will use the average 'ROC-AUC' for each ad category.

There is also a bonus task: to predict the beginning and end of contact information in the ad description. For example:
* for string `Call +7-888-888-88-88, incorrect number in ad `: (5, 21),
* for string `Call +7-888aaaaa888aaaa88a88, incorrect number in ad`: (5, 28),
* for string `my telegram: @ivanicki_i I don't answer calls`: (13, 24),
* for string `my telegram: ivanicki_i I don't answer calls`: (13, 24),
* if there is no contact in description: (None, None)
* if there is more then one contact in description(`Call 89990000000 or 89991111111`): (None, None).

The file with the result of the model should be a ' csv` with columns:
* `index`: `int`, position of the record in file;
* `start`: `int` or `None`, start contact mask;
* `end`: `int` or `None`, end contact mask.\
(`start` < `end`)

|index  |start|end|
|-------|----------|-----|
|0|None|None|
|1|0|23
|2|31|123
|...|...|
|N|None|None

After working out the script `run.py` a csv file should be written `/task-for-hearing-date/mask_prediction.csv`.

For a bonus task, the metric will be the average IoU (`Intersection over Union') for the ad texts.

A container have no access to the internet
You can add necessary modules in `requirements.txt` or directly in `Dockerfile`.

Make a fork of this repository, and as a solution, send a link to your branch

Good luck :)

# Launching the solution
1. The docker image will be builded by the command:\
```docker build -t task_for_hiring -f Dockerfile .```
2. Next, the container will be launched:\
```docker run -it -v ~/main/task-for-hiring-data:/task-for-hiring-data task_for_hiring python lib/run.py```
3. The files with the received prediction should be located in the same path as in the test version of the script `lib/run.py`

Pay attention container uses python3:

```docker run -it task_for_hiring python -c "import sys; print(sys.version)"```
> 3.7.3 (default, Mar 27 2019, 22:11:17)

Container resources:
* 4 GB RAM
* 2 core CPU
* no GPU

Limitation of work duration:
* 100 000 objects should be processed no more than 90 minutes for the previously saved model, 270 minutes for training and inference.

**It is important that everything you need to run run.py, was in the repository. Often, the decision makers offer to manually download the archive with the weights of the model before launching, in this case, it is necessary that the weights are downloaded and unpacked during the assembly of the container or the training takes place in the pipeline.**

# Baseline

Current baseline, you should break - 0.9.
