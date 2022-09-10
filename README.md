## README

Hallo und Willkommen!

### Initial SetUp

Unzip both the COSC4368-RL-Project and RL_v0 files and keep them in their current locations as independent directories

Please download and install the following packages:

numpy:
pip install numpy

.NET SDK x64:
https://dotnet.microsoft.com/en-us/download

Download your OS's version of Unity hub:
https://unity.com/download 

Run Unity hub
Click on "Installs" from the left side pannel
Choose Unity version: 2020.3.32f1
Proceed to install Unity locally on your machines

Once the installation is complete, click on "Projects"
Select the "Open" drop down menu on the upper right
Select "Add project from disk"
Select the RL_v folder 
Select "Add project"


### Running from Unity
#### For optimal performance, quit all running applications except for Unity and keep the GUI in focus during runtime

Click on "RL_v0" in Unity Hub to open the project

To connect VS CODE:
Select the Unity drop down menu on the upper left and select "Preferences"
Find: External Tools -> External Script Editor -> Visual Studio Code (or your prefered editor, however, we highly recommend following this setup for optimal performance ease)
Close "Preferences"

In the main GUI
Find: Assets -> scripts -> RunShell.cs and double click on the file to open the script in VS CODE
Find all instances of paths, replace the current path with your absolute path to each of the noted files and save

Open controller.sh and replace the current path with your absolute path to COSC4368-RL-Project/Q_learning/run_model.py 
Select which experiement you'd like to run from the following options:

Seed 123, hivemind=False
'exp_1a_v0', 'exp_1b_v0', exp_1c_v0', 'exp_2_v0', 'exp_4_v0'

Seed 326, hivemind=False
'exp_1a_v1', 'exp_1b_v1', exp_1c_v1', 'exp_2_v1', 'exp_4_v1'

Seed 577, hivemind=True
'exp_1a_v2', 'exp_1b_v2', exp_1c_v2', 'exp_2_v2'

Seed 440, hivemind=True
'exp_4_v2', 'exp_1a_v3', 'exp_1b_v3', exp_1c_v3', 'exp_2_v3', 'exp_4_v3'

Save the file
Return to the GUI

Upon first glance, you will see the game board, along with all possible pickUp and dropOff locations, filled with their respective maximum number of tokens
The way this software was developed implements a "trick of the eye" when each agent picks up and drops off a token. Once the game begins, you will see the appropriate setup appear and will follow along with playerM (male agent, orange android) and playerF (female agent, fuschia android) as they strengthen their minds. Now, to execute the game...

From the MENU bar, select the "Window" drop down menu, find "General" and select "Console" to assure you have a console present in your layout
On the upper center of the GUI, find and click the play button - this will execute your chosen experiement
In the console, you will see the experiement you are running as confirmation
Should you want an output of the instructions generated, goto RunShell.cs and comment out RemoveFile() as such: //RemoveFile()
Should you want to pause at any time, press the pause button next to the play button
Should you want to execute a full stop, press the play button again

Please note that you can adjust the windows in the Unity editor and i nparticular, for game play. This can cause some resolution shifts/potential aritifacts in the game when doing so. This is machine dependent. Should this game be built and exported, such artifacts will cease to exist. 

Repeat the same process for running subsequent experiements

### Running from Terminal
Open a new terminal session
Copy the python executable line from controller.sh along with your chosen experiment as an argument
Paste in the command line and hit enter

To run all models consecutively and to generate a final q-table: 
run python official_runs.py, located here: COSC4368-RL-Project/Q_learning_python/official_runs.py
Open official_runs.py to choose whether to run "hive = True" or "hive = False"

For all q-tables, please see the COSC4368-RL-Project directory.

#### Enjoy the show!

- Team Mimosa