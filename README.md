# The Binding of Isaac - AI Mod
Purpose: Build a working game-to-Python interface and a basic reinforcement-learning loop that turns room state into actions every frame. This is an integration milestone, not a fully trained bot.

## Requirements
- The Binding of Isaac: Repentence (Steam Version)
- Repentogon (Mod Extension)

## Set-up Instructions 
1. Install The Binding of Isaac: Repentence through Steam
2. Install Repentogon 
- Click [here](https://youtu.be/hF4ngfDn364?si=qrt4d8w2WkdSY-hs) for a step-by-step walkthrough
- Add '--luadebug' to your Steam launch options
- Launch the game through Steam 
3. Copy the mod into the game
- Locate the game files (usually located in the Program Files x86 folder of your Local Disk)
```powershell
C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\mods
```
- Copy and paste the **PythonBridge** folder located in the mod files folder of this repository into the 'mods' folder 

## Running the Mod
1. Locate the "isaac-dqn" folder using the terminal and install dependencies 
```powershell
cd "c:\Users\eulun\Desktop\310 Project\AI-Isaac\isaac-dqn"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```
2. Run the mod in the terminal. 
```powershell
cd "c:\Users\eulun\Desktop\310 Project\AI-Isaac\isaac-dqn"
py -m src.train_server
```
3. Launch The Binding of Isaac 
4. Start a game
5. Once loaded in the game, "Link established!" appears in Python and live data updates begin.


