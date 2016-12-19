"""Create the adapted training set"""
import pandas as pd

# Load input (Context, Utterance, Label)
input = pd.read_csv('train.csv', sep=',', header=0)
print("Open training data: %d lines" % len(input))

# Split data according to label
input_true = input[input['Label'] == 1]
print('%d true context' % len(input_true))
input_false = input[input['Label'] == 0]
print('%d false context' % len(input_false))

# Create output (Context, Ground Truth Utterance, Distractor_0, ... , Distractor_8)
output = pd.DataFrame()
output['Context'] = input_true['Context'].values
output['Ground Truth Utterance'] = input_true['Utterance'].values
for i in range(9):
    name_col = ('Distractor_%d' % i)
    random_distractors = input['Utterance'].sample(n=len(input_true))
    output[name_col] = random_distractors.values

# Remove inconsistent rows
output = output[(output['Ground Truth Utterance'] != output['Distractor_0'])
                & (output['Ground Truth Utterance'] != output['Distractor_1'])
                & (output['Ground Truth Utterance'] != output['Distractor_2'])
                & (output['Ground Truth Utterance'] != output['Distractor_3'])
                & (output['Ground Truth Utterance'] != output['Distractor_4'])
                & (output['Ground Truth Utterance'] != output['Distractor_5'])
                & (output['Ground Truth Utterance'] != output['Distractor_6'])
                & (output['Ground Truth Utterance'] != output['Distractor_7'])
                & (output['Ground Truth Utterance'] != output['Distractor_8'])]

# Save output
output.to_csv('train_mem.csv', index=False)



# NB: On average, 0.036% chance of getting two identical distractors for the same context