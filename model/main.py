from arm import Arm
from eye import Eyes
from IPython import embed

def main():
    eyes = Eyes(origin=0, visualize=True)
    #eyes.create_dataset(train_file='eye_train_data.p', val_file='eye_val_data.p', test_file='eye_test_data.p', n_datapoints=100)
    prototypes = eyes.create_prototypes(shape = (10,10))
    embed()
    wait = raw_input("Press enter when done...")

if __name__ == '__main__':
    main()