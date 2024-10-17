# Introduction to Deep Learning (IN2346)

# Technical University of Munich - WS 24/25

Welcome to the exercise session of Introduction to Deep Learning. In this file, we are going to guide you in setting up the environment and running the first exercise. You can run this exercise (and the following exercises) on your PC (see 1.1) or in Google Colab (See 1.2).

## 1.1 Anaconda setup
The environment that we are going to use throughout this course is Anaconda.
Download and install conda from https://www.anaconda.com/.

Open a terminal (open the Anaconda Prompt or Anaconda Powershell Prompt on Windows) and create an environment using the command:

`conda create --name i2dl python=3.11 -y`

Next, activate the environment using the command:

`conda activate i2dl`

You will notice that the name of the current activated name is shown in the command line, like:

`(i2dl) current/working/directory>`

Make sure to check the name of the environment every time you run a command in a terminal.

Within the terminal, direct yourself to exercise_01's folder and once there, navigate to either the Windows or macOS/Linux requirements folder to continue with the installation of the requirements. You can use the change directory command like this:

`cd some_directory`

You can list all folders in the current directory using the list command:

`ls`

Once you are in the correct path, execute this:

`pip install -r requirements.txt`

Now you are ready to go! You can directly start jupyter notebook, i.e.

`jupyter notebook`

Or just use your preferred IDE to start with the exercise!

## 1.2 Google Colab
Deep learning is an expensive practice. It only burst about 10 years ago into our lives because GPUs became strong enough to allow the magic it is.
As most of us do not possess a computer with a GPU, Google offers a free platform that allows you to use their cloud GPUs. Weak as they might be, they are still powerful enough to ease our training processes and make them 10x faster. This will be crucial for the later exercises of the course. Therefore, we recommend that you become familiar with it early.
However, exercises 1-5 do not require such capabilities.

In order to use the platform, open a folder in your Google Drive main page, under the name `i2dl` for consistency with the exercises.
In there, paste the exercises folders. Then, you could simply open the notebooks with the colab-notebook. There, you should follow the instructions we've assembled for you in each notebook.

Pay attention to the fact that files require a few seconds to save to the Colab cloud disk. Therefore, run the zipping cell in the notebook after you've waited a few seconds, letting the previous cell
save your models to the disk. Otherwise, you will have trouble submitting your code without your models.

Download your zipped exercise from the drive and submit it to the submission platform.

NOTE: Pytorch does NOT support MacBooks with the M1 or M2 cpus. Therefore, in order to utilize a GPU --> use Colab.

## 2. Exercise Download
The exercises will be uploaded to the [course website](https://cvg.cit.tum.de/teaching/ws2024/i2dl). You can download the exercises directly from there or from the [Resources](https://piazza.com/tum.de/fall2024/in2346ws2425/resources) section on Piazza.

Each exercise contains at least one jupyter notebook, which could be opened by the jupyter-notebook platform (In the terminal, go to the relevant folder and type `jupyter notebook`), or several IDEs that support it, such as Microsoft's VScode or JetBrains' PyCharm.

The rest of the code resides in .py files. Access those files in any IDE of your choice (VScode, Pycharm, Spyder). You could also work directly on the jupyter platform, but we do not recommend it.
IDE is a powerful tool that allows you to navigate easily through the projects, debug, and even show you your errors.

### The directory layout for the exercises
The exercises are organized to work with the file structure shown below. By unzipping the first exercise, you automatically get some of the folders. For the remaining exercises, you need to download and unzip the exercise folder and place it in the `i2dl/` folder.

    i2dl
    ├── datasets       # The datasets required for all exercises will be placed here
    ├── exercise_1                    
    ├── exercise_2     # To be added later
    ├── exercise_3     # To be added later
    ├── exercise_4     # To be added later
    ├── exercise_5     # To be added later
    ├── exercise_6     # To be added later
    ├── exercise_7     # To be added later  
    ├── exercise_8     # To be added later
    ├── exercise_9     # To be added later
    ├── exercise_10    # To be added later
    ├── exercise_11    # To be added later
    └── output         # Stores files to be uploaded to the submission system.

You are now ready for the first exercise! Open `1_introduction.ipynb` in Jupyter, your IDE, or in Google Colab, and follow the instructions to finish the first exercise.
We also recommend you to read the rest of this file to have a better understanding of how the exercise works.

## 3. Dataset Download
Datasets will generally be downloaded automatically by exercise notebooks and stored in a common datasets directory shared among all exercises. A sample directory structure for cifar10 dataset is shown below:

    i2dl
        ├── datasets                   # The datasets required for all exercises will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files


## 4. Exercise Submission
Your trained models will be automatically evaluated on a test set on our server. To this end, log in or register for an account at:

https://i2dl.cvg.cit.tum.de

Note that only students who have registered for this class in TUM Online can register for an account. This account provides you with temporary credentials to log in to the machines at our chair.

At the end of each exercise, a script zips all of the relevant files. All your trained models should be inside the `models` directory in the exercise folder. Make sure they are there, especially while working with Google Colab.

Then, submit it to the submission server (should not include any datasets).

You can log in to the above website and upload your zip submission. Our system will evaluate your submission.

You will receive an email notification with the results upon completion of the evaluation. To make the exercises more fun, you will be able to see a leaderboard of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the current evaluation result, the entry in the leaderboard always represents the best score for the respective exercise.

## 5. Acknowledgments
We want to thank the **Stanford Vision Lab** and **PyTorch** for allowing us to build these exercises on material they had previously developed. We also thank the **TU Munich Chair of Computer Graphics and Visualization** for helping create course content.