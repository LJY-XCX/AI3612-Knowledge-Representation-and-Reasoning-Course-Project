## Codes

All prompting and post-processing methods can be found in ./src/, including randomly selecting method, selecting according to similarity method, self-consistency method, GPT-NER method, and two post-processing scripts.

## Illustration on k-nearest part

The checkpoint folder already contains the PCA transformation matrix, training data embedding matrix, and test data embedding matrix. During normal operation, the program can directly read and use these data to quickly obtain similar prompts. If these files are accidentally deleted, the source code can still generate these files.

## Outputs

Model outputs can be found in ./ckpts/chatgpt_api/.

If you have any problem with our project, please feel free to contact us in any way.
