import pyautogui as pg
import pyperclip
import subprocess


def write_text(text):
    for line in text.split("\n"):
        pg.hotkey("home")
        pyperclip.copy(line)
        pg.hotkey("ctrl", "v")
        pg.press("enter")

def press_shortcut(shortcut):
    pg.hotkey(*shortcut)
    
    
def update_files_list():
    directories = [
        "~/Desktop",
        "~/Downloads",
        "~/Documents",
        "~/Videos",
        "~/Pictures",
        "~/Music",
    ]
    skip_files = [
        "*/.git/*",
        "*/.vscode/*",
        "*/__pycache__/*",
        "*/node_modules/*",
        "*/venv/*",
    ]
    subprocess.run([
        "find",
        *directories, 
        "-type", "f",
        *(x for file in skip_files for x in ("-not", "-path", f"\"{file}\"")), 
        ">", "files.txt"
    ], shell=True)
