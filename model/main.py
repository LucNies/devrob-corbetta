from arm import Arm
from eye import Eyes

def main():
    eyes = Eyes(origin=0, visualize=False)
    #eyes.create_dataset(train_file='eye_train_data.p', val_file='eye_val_data.p', test_file='eye_test_data.p', n_datapoints=100)
    while True:
        eyes.random_eye_pos()
        eyes.redraw()

if __name__ == '__main__':
    main()