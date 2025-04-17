# Batch Processing

We recommend using the terminal in batch processing, if you wanted to play with the functions, you can check the `batch-processing-example.ipynb`

To do batch processing, the following assumes that you have your research papers in PDF format (in our case we have climate related [PRWP documents](https://www.worldbank.org/en/research/brief/world-bank-policy-research-working-papers), as well as Adaptation-One-Earth-Policy documents) on the `input` directory.

You also need to set up your `config.yaml` and `.env` file and put your `OPENAI_API_KEY` there. Also, make sure to change the necessary configurations such as `MAX_REQUESTS_PER_BATCH` if you have large scale pdf files you can set it to Max. of 50,000 which is the api limit for batch processing.

## Workflow

We have a 3-step process in this data labeling process:

1. **`Zero-shot Extraction`**

Using 4o-mini, we will extract potential dataset mentions and its corresponding metadata (if available).
2. **`LLM-as-a-Judge Validation`**

Using the zero-shot extraction outputs, we will then use a validation layer where we will tag each of the dataset mentions `valid:true` if the model thinks it is a dataset mention and set it to false if not, together with its corresponding `invalid_reason`.
3. **`Autonomous Reasoning`**

Using the output of the LLM-as-a-Judge validation, we will then make the final layer where we incorporate a `Devil's Advocate` mechanism to challenge its own classification by considering alternative interpretations. It also re-evaluates ambiguous cases and overrides the previous judgements of the earlier layers.

The process are named `"extraction"`, `"judge"` and `"reasoning"`.

## Zero-Shot Extraction

```bash
# once the prerequisites and dependencies are sufficed run the following in a terminal

python run_batch.py --process extraction
```

The script above will process the `input directory` and handles the processing of the desired openai batches format and submits it. It will set up the directories needed for the process to run, saves the list of the `batch_ids` to a text file to track its status.

The helper code below lets you check the status of your batch run.

```python
def list_batches(client):
    """
    Lists all submitted batches along with their statuses.
    """
    try:
        batches = client.batches.list()
        print("All Batch Jobs:")
        for batch in batches:
            print(f"Batch ID: {batch.id}, Status: {batch.status}, Created At: {batch.created_at}")
    except Exception as e:
        print(f"Error listing batches: {e}")

# or use the text file under `extraction_outputs` to filter the outputs
api_key = "YOUR_API_KEY" # or get from config using load_config
client = OpenAI(api_key=api_key)
file_path = "extraction_outputs/extraction_batches.txt"

with open(file_path, "r") as f:
    batches_res = f.readlines()

batch_ids = [batch.strip() for batch in batches_res]
batches = client.batches.list()
for batch in batches:
    if batch.id in batch_ids:
        print(f"{batch.id} : {batch.status}")

```

**Note: It will take a while for the batches to be `completed`.**

Once the status of all the batches are completed. We need to retrieve its results, just run the following code.

```bash

python retrieve_results.py --process extraction

```

It will automatically place the result of each batch run under the `extraction_outputs/extraction` to its corresponding output file.

## LLM-as-a-Judge

Once the outputs are saved under `extraction_outputs/extraction` we can now process the LLM-as-a-Judge pipeline where the model will validate the zero-shot extracted dataset mentions.

```bash

python run_batch.py --process judge

```

batch ids for this process will be saved under `extraction_putputs/judge_batches.txt`, you can track again the batch run until it is `completed`.

Again, once completed we can run the file to retrieve its results.

```bash

python retrieve_results.py --process judge

```

It will automatically place the result of each batch run under the `extraction_outputs/judge` to its corresponding output file.

## Autonomous Reasoning Agent

Once the information is validated by the LLM, we will use the autonomous reasoning agent to further refine and validate the extracted data. The reasoning agent will follow a structured prompt to ensure the accuracy and relevance of the dataset mentions.

```bash

python run_batch.py --process reasoning

```

batch ids for this process will be saved under `extraction_putputs/reasoning.txt`, you can track again the batch run until it is `completed`.

Once completed we can run the file to retrieve its results.

```bash

python retrieve_results.py --process reasoning

```

It will automatically place the result of each batch run under the `extraction_outputs/reasoning` to its corresponding output file.

## Next Steps

Now that you have your validated results from the pipeline, you can now make a `fine-tuning dataset.`
After you have your reasoning outputs from the task earlier, you just need to run the code below.

```bash

python generate_finetune_data.py

```

## [OPTIONAL] MANUALLY LABELLED DATA

You can also label manually annotated data to finetune your model.

## Finetuning Your Model

After generating your finetuning data, you can now finetune it. We have provided a notebook where you can finetune your model using Unsloth. You can find this notebook in the `examples` folder. Follow the instructions in the notebook to load your finetuning data and start the finetuning process.
