from create_dataset import create_dataset
from data_processor import split_data_into_training_and_test_sets
from process_test_data import transform_test_data
from data_preparation import prepare_test_data
from sklearn.preprocessing import LabelEncoder
from model import create_model
from disambiguer import Disambiguer
from results_evaluation import Evaluation

one_hot = LabelEncoder()

print('----------------------------------------------------------------------------------------------------------')
print("STAGE 1: CREATING DATASET")
print('----------------------------------------------------------------------------------------------------------')

create_dataset(one_hot)

print('----------------------------------------------------------------------------------------------------------')
print("STAGE 2: SPLIT DATA INTO TRAINING AND TEST SETS")
print('----------------------------------------------------------------------------------------------------------')

split_data_into_training_and_test_sets()

print('----------------------------------------------------------------------------------------------------------')
print("STAGE 3: CREATE MODEL")
print('----------------------------------------------------------------------------------------------------------')

create_model()

print('----------------------------------------------------------------------------------------------------------')
print("STAGE 4: PREPARE TEST DATA")
print('----------------------------------------------------------------------------------------------------------')

data_path = "./data/test-analyzed.xml"
prepare_test_data(data_path, one_hot)

print('----------------------------------------------------------------------------------------------------------')
print("STAGE 5: PROCESS TEST DATA WITH MODEL")
print('----------------------------------------------------------------------------------------------------------')

transform_test_data(data_path)

disamb = Disambiguer(data_path)
print('----------------------------------------------------------------------------------------------------------')
print("STAGE 6: TAGG LEXEMS")
print('----------------------------------------------------------------------------------------------------------')

disamb.tag_lexems()

print("STAGE 7: EVALUATE MODEL")


