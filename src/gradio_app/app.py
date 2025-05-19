import boto3
import os
import json
import re
import gradio as gr
from typing import List, Dict, Tuple, Optional, Union, Any

# ── S3 CONFIG ─────────────────────────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name           = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
)
BUCKET       = "doccano-processed"
INIT_KEY     = "gradio/initial_data_train.json"
VALID_PREFIX = "validated_records/"

# ── Helpers to load & save from S3 ──────────────────────────────────────────────
def load_initial_data() -> List[Dict]:
    obj = s3.get_object(Bucket=BUCKET, Key=INIT_KEY)
    return json.loads(obj['Body'].read())

def load_all_validations() -> Dict[int, Dict]:
    records = {}
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=VALID_PREFIX
    )
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            idx = int(os.path.splitext(os.path.basename(key))[0])
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            records[idx] = json.loads(data)
    return records

def save_single_validation(idx: int, record: Dict):
    key = f"{VALID_PREFIX}{idx}.json"
    s3.put_object(
        Bucket      = BUCKET,
        Key         = key,
        Body        = json.dumps(record, indent=2).encode('utf-8'),
        ContentType = 'application/json'
    )

class DynamicDataset:
    def __init__(self, data: List[Dict]):
        self.data    = data
        self.len     = len(data)
        self.current = 0
        for ex in self.data:
            ex.setdefault("validated", False)

    def example(self, idx: int) -> Dict:
        self.current = max(0, min(self.len - 1, idx))
        return self.data[self.current]

    def next(self) -> Dict:
        if self.current < self.len - 1:
            self.current += 1
        return self.data[self.current]

    def prev(self) -> Dict:
        if self.current > 0:
            self.current -= 1
        return self.data[self.current]

    def jump_next_unvalidated(self) -> Dict:
        for i in range(self.current + 1, self.len):
            if not self.data[i]["validated"]:
                self.current = i
                break
        return self.data[self.current]

    def jump_prev_unvalidated(self) -> Dict:
        for i in range(self.current - 1, -1, -1):
            if not self.data[i]["validated"]:
                self.current = i
                break
        return self.data[self.current]

    def validate(self):
        self.data[self.current]["validated"] = True

def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+(?:[-_]\w+)*|[^\s\w]", text)

def prepare_for_highlight(data: Dict) -> List[Tuple[str, Optional[str]]]:
    tokens = data["tokenized_text"]
    ner    = data["ner"]
    highlighted, curr_ent, ent_buf, norm_buf = [], None, [], []
    for idx, tok in enumerate(tokens):
        if curr_ent is None or idx > curr_ent[1]:
            if ent_buf:
                highlighted.append((" ".join(ent_buf), curr_ent[2]))
                ent_buf = []
            curr_ent = next((e for e in ner if e[0] == idx), None)
        if curr_ent and curr_ent[0] <= idx <= curr_ent[1]:
            if norm_buf:
                highlighted.append((" ".join(norm_buf), None))
                norm_buf = []
            ent_buf.append(tok)
        else:
            if ent_buf:
                highlighted.append((" ".join(ent_buf), curr_ent[2]))
                ent_buf = []
            norm_buf.append(tok)
    if ent_buf:
        highlighted.append((" ".join(ent_buf), curr_ent[2]))
    if norm_buf:
        highlighted.append((" ".join(norm_buf), None))
    return [(re.sub(r"\s(?=[,\.!?…:;])", "", txt), lbl) for txt, lbl in highlighted]


def extract_tokens_and_labels(highlighted: List[Dict[str, Union[str, None]]]
                            ) -> Tuple[List[str], List[Tuple[int,int,str]]]:
    tokens, ner = [], []
    token_idx = 0

    for entry in highlighted:
        text  = entry['token']
        label = entry.get('class_or_confidence') or entry.get('class') or entry.get('label')
        # split into real tokens
        toks = tokenize_text(text)
        start = token_idx
        end   = token_idx + len(toks) - 1

        tokens.extend(toks)
        if label:
            ner.append((start, end, label))

        token_idx = end + 1

    return tokens, ner


# ── App factory ────────────────────────────────────────────────────────────────
def create_demo() -> gr.Blocks:
    data             = load_initial_data()
    validated_store  = load_all_validations()

    for idx in validated_store:
        if 0 <= idx < len(data):
            data[idx]["validated"] = True
    dynamic_dataset  = DynamicDataset(data)
    with gr.Blocks() as demo:
        prog      = gr.Slider(0, dynamic_dataset.len-1, value=0, step=1, label="Example #", interactive=False)
        inp_box   = gr.HighlightedText(label="Sentence", interactive=True)
        status    = gr.Checkbox(label="Validated?", value=False, interactive=False)
        gr.Markdown(
            "[📖 Entity Tag Guide](https://huggingface.co/spaces/rafmacalaba/datause-annotation/blob/main/guidelines.md)"
        )
        
        with gr.Row():
            prev_btn  = gr.Button("◀️ Previous")
            apply_btn = gr.Button("📝 Apply Changes")
            next_btn  = gr.Button("Next ▶️")
        with gr.Row():
            skip_prev = gr.Button("⏮️ Prev Unvalidated")
            validate_btn = gr.Button("✅ Validate")
            skip_next = gr.Button("⏭️ Next Unvalidated")

        def load_example(idx):
            rec  = validated_store.get(idx, dynamic_dataset.example(idx))
            segs = prepare_for_highlight(rec)
            return segs, rec.get("validated", False), idx

        def update_example(highlighted, idx: int):
            # grab the record
            rec = dynamic_dataset.data[idx]

            # re‐tokenize from the raw text (same as do_validate)
            orig_tokens = tokenize_text(rec["text"])

            # realign the user's highlights back to those tokens
            new_ner = align_spans_to_tokens(highlighted, orig_tokens)

            # overwrite both token list and span list (and mark un‐validated)
            rec["tokenized_text"] = orig_tokens
            rec["ner"]            = new_ner
            rec["validated"]      = False

            # re‐render
            return prepare_for_highlight(rec)

        def align_spans_to_tokens(
            highlighted: List[Dict[str, Union[str, None]]],
            tokens: List[str]
        ) -> List[Tuple[int,int,str]]:
            """
            Align each highlighted chunk to the next matching tokens in the list,
            advancing a pointer so repeated tokens map in the order you clicked them.
            """
            spans = []
            search_start = 0

            for entry in highlighted:
                text  = entry["token"]
                label = entry.get("class_or_confidence") or entry.get("label") or entry.get("class")
                if not label:
                    continue

                chunk_toks = tokenize_text(text)
                # scan only from the end of the last match
                for i in range(search_start, len(tokens) - len(chunk_toks) + 1):
                    if tokens[i:i+len(chunk_toks)] == chunk_toks:
                        spans.append((i, i + len(chunk_toks) - 1, label))
                        search_start = i + len(chunk_toks)
                        break
                else:
                    print(f"⚠️ Couldn’t align chunk: {text!r}")

            return spans

        def do_validate(highlighted, idx: int):
            # mark validated in memory
            dynamic_dataset.validate()

            # grab the record
            rec = dynamic_dataset.data[idx]

            # re-tokenize from the original text
            orig_tokens = tokenize_text(rec["text"])

            # realign the user's highlighted segments to those tokens
            new_ner = align_spans_to_tokens(highlighted, orig_tokens)

            # overwrite both token list and span list
            rec["tokenized_text"] = orig_tokens
            rec["ner"]            = new_ner

            # persist
            save_single_validation(idx, rec)

            # re-render and show checkbox checked
            return prepare_for_highlight(rec), True


        def nav(fn):
            rec  = fn()
            segs = prepare_for_highlight(rec)
            return segs, rec.get("validated", False), dynamic_dataset.current

        demo.load(load_example, inputs=prog, outputs=[inp_box, status, prog])
        apply_btn.click(
            fn=update_example,
            inputs=[inp_box, prog],     # pass both the highlights *and* the example idx
            outputs=inp_box
        )
        #apply_btn.click(update_spans, inputs=inp_box, outputs=inp_box)
        prev_btn.click(lambda: nav(dynamic_dataset.prev), inputs=None, outputs=[inp_box, status, prog])
        validate_btn.click(do_validate, inputs=[inp_box, prog], outputs=[inp_box, status])
        next_btn.click(lambda: nav(dynamic_dataset.next), inputs=None, outputs=[inp_box, status, prog])
        skip_prev.click(lambda: nav(dynamic_dataset.jump_prev_unvalidated), inputs=None, outputs=[inp_box, status, prog])
        skip_next.click(lambda: nav(dynamic_dataset.jump_next_unvalidated), inputs=None, outputs=[inp_box, status, prog])

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True, inline=True, debug=True)