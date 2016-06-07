from arm import Arm
from eye import Eyes

def main():
    eyes = Eyes(origin=0)
    while True:
        eyes.random_eye_pos()
        eyes.redraw()

if __name__ == '__main__':
    main()