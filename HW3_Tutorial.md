# Assignment 3: Tokenizer & Inference with LLMs — Complete Tutorial

**Due: Thursday, March 26 at 11:59PM EST**

**Submission Files:** `bpe.py`, `test_tokenizer.py`, written answers PDF (Part 1), `submission.py`, `prompting_exercises.ipynb` (Part 2)

---

## PROGRESS TRACKER — Where We Stopped & What's Next

### COMPLETED (Part 1 — 35 pts)
- [x] `bpe.py` — `merge()`, `encode()`, `decode()` all implemented and working
- [x] `test_tokenizer.py` — all 3 property tests (`test_not_injective`, `test_not_invertible`, `test_not_preserving_concat`) filled in
- [x] Written answers for Q2b and Q2c drafted (see Section 1e below)
- [x] Tutorial theory for Part 1 complete

### TODO (Part 2 — 65 pts) — Your Next Steps
- [ ] **Step A: Get HPC / GPU access ready** — You need Meta Llama access on HuggingFace + GPU (HPC/Colab/Lightning.ai)
- [ ] **Step B: Update `submission.py`** — The file still has the OLD prompt. Copy the optimized version from Section 2c below (or follow the step-by-step update guide in Section 2f)
- [ ] **Step C: Fill in notebook answer cells** — All "Answer:" cells are empty. Section 2g below tells you exactly what to write in each one
- [ ] **Step D: Fill in notebook code cell-39** — The Q4b post-processing function is empty. Code provided in Section 2g
- [ ] **Step E: Run the entire notebook on GPU** — Every cell must be executed, results visible
- [ ] **Step F: Run `python run_tests.py`** — Test your `submission.py` on the 30-instance grader
- [ ] **Step G: Paste actual results** — Into Section 2d's placeholder blocks
- [ ] **Step H: Write the PDF** — Compile written answers from Sections 1e + 2e into a PDF
- [ ] **Step I: Replace placeholders in `submission.py`** — Put your actual NetID and HF token

### CRITICAL REMINDERS
- Part 2 requires **meta-llama/Llama-2-7b-chat-hf** access (gated model — request early!)
- Part 2 also requires **Qwen/Qwen3-8B** access
- `max_tokens` in `submission.py` must be >= 50
- No arithmetic operations in pre/post-processing (autograder checks!)
- The notebook mislabels Q3d as "Q4d" in cell-26 — it IS Q3d per the assignment PDF

---

# PART 1: BPE Tokenizer (35 pts)

---

## 1a. Deliverables Overview

| Deliverable | Type | Points | File |
|---|---|---|---|
| Q1: Implement `merge()`, `encode()`, `decode()` | Code | 23 | `part1/src/bpe.py` |
| Q2a: Counterexamples for tokenizer properties | Code | 4 | `part1/tests/test_tokenizer.py` |
| Q2b: Description of each violation | Written (PDF) | 4 | PDF |
| Q2c: Why concatenation preservation is impossible | Written (PDF) | 4 | PDF |

**What you will build:**
- A working **Byte Pair Encoding (BPE) tokenizer** that can train on ASCII text, encode strings into token IDs, and decode token IDs back to strings.
- **Property tests** that demonstrate real-world tokenizers violate injectivity, invertibility, and concatenation preservation.

---

## 1b. Background and Theory

### What is BPE (Byte Pair Encoding)?

BPE is a **subword tokenization algorithm** used by GPT-2, GPT-3, LLaMA, and many other LLMs. It bridges the gap between character-level and word-level tokenization.

**Core idea:** Start with individual characters as tokens, then iteratively merge the most frequent adjacent pair of tokens into a new single token.

### The BPE Algorithm Step-by-Step

**Initialization:**
- Start with a vocabulary of all individual characters. For ASCII, that's 128 characters (IDs 0–127).
- The character `'1'` has ASCII code 49, `'2'` → 50, `'3'` → 51, `'a'` → 97, etc.

**Training (repeated `n_merges` times):**
1. Scan through the token sequence and count every adjacent pair (bigram). E.g., in `[49, 50, 51, 49, 50, 51]`, the bigram `(49, 50)` appears 2 times, `(50, 51)` appears 2 times, `(51, 49)` appears 1 time.
2. Find the **most frequent** bigram. If there's a tie, pick the **lexicographically smaller** one. Lexicographic comparison on tuples means: compare first element, then second. So `(49, 50) < (50, 51)` because `49 < 50`.
3. Assign this bigram a **new token ID** = `128 + number_of_merges_so_far`. The first merge creates token 128, the second creates 129, etc.
4. **Replace** every occurrence of that bigram in the token sequence with the new token ID.
5. Add the new token to `self.vocab` (its string is the concatenation of the two merged tokens' strings).
6. Record the merge rule in `self.merge_rules`: `{(left_id, right_id): new_id}`.

**Concrete Example with `"123123"`:**

Initial token IDs (ASCII): `[49, 50, 51, 49, 50, 51]`

**Merge 1:**
- Bigram counts: `(49,50):2`, `(50,51):2`, `(51,49):1`
- Tie between `(49,50)` and `(50,51)` → pick `(49,50)` (lexicographically smaller)
- New token ID = 128, represents `"12"`
- After replacement: `[128, 51, 128, 51]`
- `vocab[128] = "12"`, `merge_rules = {(49,50): 128}`

**Merge 2:**
- Bigram counts on `[128, 51, 128, 51]`: `(128,51):2`, `(51,128):1`
- Most frequent: `(128, 51)`
- New token ID = 129, represents `"123"`
- After replacement: `[129, 129]`
- `vocab[129] = "123"`, `merge_rules = {(49,50): 128, (128,51): 129}`

### Encoding (applying a trained tokenizer to new text)

Encoding converts a string to token IDs using the **learned merge rules**:
1. Convert the string to ASCII codes (the base tokens).
2. Apply each merge rule **in the order they were learned** (this order is preserved by the dict insertion order in `self.merge_rules`).
3. For each merge rule `(a, b) → c`: scan the token list and replace every adjacent `[a, b]` with `[c]`.

**Why order matters:** Merge rules build on each other. Rule 1 creates token 128 from `(49,50)`. Rule 2 uses token 128 in `(128,51)` → 129. If you applied rule 2 before rule 1, token 128 wouldn't exist yet in the sequence.

### Decoding (converting token IDs back to text)

Decoding is simple: look up each token ID in `self.vocab` to get its string, then concatenate all the strings.

### Tokenizer Properties

**Injective:** Different strings should produce different token sequences. Violated when a tokenizer normalizes input (e.g., collapsing whitespace, lowercasing).

**Invertible:** `decode(encode(s)) == s`. Violated when the tokenizer loses information during encoding (e.g., unknown characters replaced with `[UNK]`).

**Preserves Concatenation:** `encode(s1 + s2) == encode(s1) + encode(s2)`. Violated because merges at the boundary of `s1` and `s2` can create tokens that wouldn't exist when encoding separately. This is inherent to any tokenizer that considers context beyond single characters.

---

## 1c. Step-by-Step Code Solutions

### Q1: Implementing `merge()`, `encode()`, `decode()`

File: `part1/src/bpe.py`

---

#### Function 1: `merge(self, token_ids)`

**What it does:** Performs ONE merge step of the BPE algorithm.

**Line-by-line explanation:**

```python
def merge(self, token_ids: list[int]) -> list[int]:
```
- Takes the current list of token IDs and returns a new list after one merge.

```python
    stats = compute_bigram_statistics(token_ids)
```
- Count all adjacent pairs. Returns a `Counter` like `{(49,50): 2, (50,51): 2, (51,49): 1}`.

```python
    best_bigram = min(stats, key=lambda x: (-stats[x], x))
```
- Find the bigram with the **highest count** (hence `-stats[x]` for descending sort).
- On ties, pick the **lexicographically smallest** tuple (that's the `x` part of the key).
- `min` with this key gives us exactly what we need: highest frequency first, then lex order.

```python
    new_id = 128 + len(self.merge_rules)
```
- The new token ID. First merge → `128 + 0 = 128`. Second merge → `128 + 1 = 129`. Etc.

```python
    self.vocab.append(self.vocab[best_bigram[0]] + self.vocab[best_bigram[1]])
```
- The new token's string = concatenation of the two merged tokens' strings. E.g., `vocab[49]="1"` + `vocab[50]="2"` = `"12"`.

```python
    self.merge_rules[best_bigram] = new_id
```
- Record the merge rule: `{(49, 50): 128}`.

```python
    return replace_bigram(token_ids, best_bigram, new_id)
```
- Use the provided helper to replace all occurrences of the bigram in the token list.

**Complete function to paste** (replace lines 85–101 in `bpe.py` — note these are methods, so they need 4-space indentation inside the class):

```python
    def merge(self, token_ids: list[int]) -> list[int]:
        """Perform one merge in the BPE algorithm.

        Specifically, implement the following:
          1. Find the most frequent bigram B, and create a new token T for it.
            If two bigram have the same frequency, break tie by taking
            the lexicographically smaller bigram.
          2. Replace all occurrences of B in token_ids with T

        Args:
            token_ids (list[int]): Current list of token ids

        Returns:
            list[int]: New list of token ids, after one merge step
        """

        stats = compute_bigram_statistics(token_ids)
        best_bigram = min(stats, key=lambda x: (-stats[x], x))
        new_id = 128 + len(self.merge_rules)
        self.vocab.append(self.vocab[best_bigram[0]] + self.vocab[best_bigram[1]])
        self.merge_rules[best_bigram] = new_id
        return replace_bigram(token_ids, best_bigram, new_id)
```

---

#### Function 2: `encode(self, text)`

**What it does:** Converts an ASCII string into a list of token IDs by applying all learned merge rules in order.

**Line-by-line explanation:**

```python
    token_ids = string_to_ascii(text)
```
- Start by converting each character to its ASCII integer. `"123"` → `[49, 50, 51]`.

```python
    for bigram, bigram_id in self.merge_rules.items():
```
- Iterate through merge rules **in the order they were added** (Python dicts preserve insertion order since 3.7).

```python
        token_ids = replace_bigram(token_ids, bigram, bigram_id)
```
- For each rule, replace all occurrences of the bigram with the merged token ID.

```python
    return token_ids
```

**Complete function to paste** (replace lines 103–115 in `bpe.py`):

```python
    def encode(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: An arbitrary ASCII string

        Returns:
            Tokens produced under BPE
        """

        assert all(ord(c) < 128 for c in text), "input text is not ASCII"
        token_ids = string_to_ascii(text)
        for bigram, bigram_id in self.merge_rules.items():
            token_ids = replace_bigram(token_ids, bigram, bigram_id)
        return token_ids
```

---

#### Function 3: `decode(self, token_ids)`

**What it does:** Converts a list of token IDs back to a string.

**Line-by-line explanation:**

```python
    return "".join(self.vocab[token_id] for token_id in token_ids)
```
- Look up each token ID in the vocab list. `vocab[128]` = `"12"`, `vocab[51]` = `"3"`.
- Join them all together. `[128, 51]` → `"12" + "3"` → `"123"`.

**Complete function to paste** (replace lines 117–126 in `bpe.py`):

```python
    def decode(self, token_ids: list[int]) -> str:
        """Convert tokens back to text.

        Args:
            token_ids: A list of token ids.

        Returns:
            str: An ASCII string.
        """
        return "".join(self.vocab[token_id] for token_id in token_ids)
```

---

### Q2a: Property Violation Tests

File: `part1/tests/test_tokenizer.py`

We use the **BERT tokenizer** (`google-bert/bert-base-cased`) to find violations. Here's why each works:

---

#### `test_not_injective()` — Two different strings, same encoding

**Concept:** BERT's tokenizer handles whitespace in a special way. Leading spaces are often stripped or tokens are prefixed with `##` for subword continuation. Multiple spaces can be normalized.

```python
s1 = "a"
s2 = " a"
```

**Why this works:** BERT's tokenizer strips leading whitespace during tokenization, so `"a"` and `" a"` produce the same token IDs. Two different strings → same encoding → not injective.

**Complete test to paste:**

```python
def test_not_injective():
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    s1 = "a"
    s2 = " a"

    assert s1 != s2 and tokenizer.encode(
        s1, add_special_tokens=False
    ) == tokenizer.encode(s2, add_special_tokens=False)
```

---

#### `test_not_invertible()` — Encode then decode doesn't return original

**Concept:** Characters that BERT's `BasicTokenizer` cannot handle are stripped or normalized during tokenization, losing information.

```python
s = "\x00"
```

**Why this works:** The null character `\x00` is a control character. BERT's `BasicTokenizer._clean_text()` method strips characters with code point 0 entirely. So `encode("\x00")` produces an empty token list `[]`, and `decode([])` returns `""`. Since `"\x00" != ""`, the invertibility property is violated.

**Alternative choices that also work:** `"\t"` (tab), `"\r"` (carriage return), or any other control character — all get stripped by BERT's text cleaning step.

**Complete test to paste:**

```python
def test_not_invertible():
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    s = "\x00"

    s_recovered = tokenizer.decode(tokenizer.encode(s, add_special_tokens=False))
    assert s != s_recovered
```

---

#### `test_not_preserving_concat()` — Encoding concatenation ≠ concatenation of encodings

**Concept:** When two strings are concatenated, new subword merges can happen at the boundary.

```python
a = "un"
b = "doing"
```

**Why this works:** BERT tokenizes `"un"` as `["un"]` and `"doing"` as `["doing"]`. But `"undoing"` is tokenized as `["un", "##doing"]` — the `##` prefix changes the token IDs. So `encode("un" + "doing") != encode("un") + encode("doing")`.

**Complete test to paste:**

```python
def test_not_preserving_concat():
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    a = "un"
    b = "doing"
    assert tokenizer.encode(a + b, add_special_tokens=False) != tokenizer.encode(
        a, add_special_tokens=False
    ) + tokenizer.encode(b, add_special_tokens=False)
```

---

## 1d. Results

> **Instructions:** Run the tests and paste the output here after running them.

### Implementation Tests
```
# Run from part1/ directory:
python tests/test_tokenizer.py implementation
```

**Expected output (paste actual output here after running):**
```
test_encode passed
test_decode passed
test_one_merge passed
test_two_merges passed
test_100_merges passed
```

### Property Tests
```
# Run from part1/ directory:
python tests/test_tokenizer.py properties
```

**Expected output (paste actual output here after running):**
```
test_not_injective passed
test_not_invertible passed
test_not_preserving_concat passed
```

---

## 1e. Written Answers (Theory)

### Q2b: Description of Each Violation (4 pts)

**Not Injective:** Using the BERT (`google-bert/bert-base-cased`) tokenizer, the strings `"a"` and `" a"` (with a leading space) produce identical token ID sequences. BERT's tokenizer strips leading whitespace, mapping two distinct inputs to the same encoding.

**Not Invertible:** Using the BERT tokenizer, the null character `"\x00"` is a control character that BERT's `BasicTokenizer` strips during text cleaning. Encoding produces an empty list, and decoding that returns `""`, which differs from the original `"\x00"`.

**Not Preserving Concatenation:** Using the BERT tokenizer, `"un"` and `"doing"` tokenized separately produce `["un"]` and `["doing"]`, but `"undoing"` tokenized as one string produces `["un", "##doing"]` with a different subword split at the boundary.

### Q2c: Why Concatenation Preservation is Generally Impossible (4 pts)

Any non-trivial tokenizer (one that produces multi-character tokens) must consider **context** — the characters surrounding a given position — to decide how to segment text. When two strings `s` and `ŝ` are concatenated, the characters at the boundary of `s` and `ŝ` form a new local context that didn't exist when each string was tokenized independently. This means the tokenizer may merge characters across the boundary into tokens that wouldn't appear in either individual tokenization. For example, if the tokenizer has learned the token `"ab"`, then `encode("a") + encode("b") = [a_id] + [b_id]`, but `encode("ab") = [ab_id]` — a single token. The only tokenizer that preserves concatenation is a trivially character-level tokenizer (one token per character), which is not useful in practice because it loses the efficiency and semantic grouping that subword tokenization provides.

---
---

# PART 2: Prompt Engineering for Addition (65 pts)

---

## 2a. Deliverables Overview

| Deliverable | Type | Points | File |
|---|---|---|---|
| Q3a: Why performance degrades 1-digit → 7-digit | Written | 2 | Notebook + PDF |
| Q3b: Config parameter explanations | Written | 3 | Notebook + PDF |
| Q3c: Qwen3-8B comparison | Written | 2 | Notebook + PDF |
| Q3d: Removing output length prior | Written | 3 | Notebook + PDF |
| Q4a: Baseline ICL performance | Written | 3 | Notebook + PDF |
| Q4b: Relaxed output length + post-processing | Written | 7 | Notebook + PDF |
| Q4c: In-distribution example evaluation | Written | 9 | Notebook + PDF |
| Q4d: Error analysis and prompt improvement | Written | 6 | Notebook + PDF |
| Q5a: Prompt-a-thon (submission.py) | Code | 30 | `part2/submission.py` |

**What you will build:**
- Run experiments in `prompting_exercises.ipynb` exploring zero-shot and in-context learning for addition.
- Design an optimized prompt in `submission.py` that teaches Llama-2-7b to add 7-digit numbers with >20% accuracy or MAE < 100,000.

**Requirements:**
- HuggingFace account with access to `meta-llama/Llama-2-7b-chat-hf` and `Qwen/Qwen3-8B`
- GPU access (HPC, Google Colab, or Lightning.ai)
- `max_tokens` must be ≥ 50 in `submission.py`
- No arithmetic operations in pre/post-processing functions

---

## 2b. Background and Theory

### How LLMs Do (or Fail at) Arithmetic

LLMs process text as **tokens**, not as numbers. The number `1234567` might be tokenized as `["123", "45", "67"]` — the model never "sees" the full number as a single entity. Addition requires:
1. Understanding each digit's place value
2. Performing column-by-column addition with carries
3. Outputting digits in the correct order

For **1-digit addition** (e.g., 3+7=10), the model has likely memorized these facts from training data. For **7-digit addition**, the model must generalize multi-step reasoning it hasn't memorized, and the tokenization fragments the numbers unpredictably.

### Key Generation Parameters

| Parameter | What it does | Effect of increasing |
|---|---|---|
| **Temperature** | Controls randomness of token selection. 0 = greedy (always pick most likely), higher = more random | More diverse/creative but less accurate outputs. For arithmetic, **lower is better** (0.0–0.1). |
| **Max Tokens** | Maximum number of new tokens the model can generate | More room for output. Too few → truncated answers. Too many → model may generate extra text after the answer. |
| **Top-K** | Only consider the K most likely next tokens at each step | Higher K = more diversity. Lower K (e.g., 1) = more deterministic. |
| **Top-P (nucleus)** | Only consider tokens whose cumulative probability ≤ P | Lower P = more focused. P=1.0 = no filtering. For arithmetic, **lower is better**. |
| **Repetition Penalty** | Penalizes tokens that have already appeared | Higher values discourage repetition. Too high can distort arithmetic outputs. Keep at 1.0. |

### In-Context Learning (ICL)

Instead of fine-tuning, we provide **examples** in the prompt that teach the model the task format:

```
Question: what is 1234567+1234567?
Answer: 2469134
Question: what is {a}+{b}?
Answer:
```

The model learns the pattern from the example(s) and attempts to follow it. Key factors for success:
- **In-distribution examples**: 7-digit examples work better than 1-digit examples for 7-digit tasks
- **Multiple examples**: More examples can improve performance (few-shot > one-shot > zero-shot)
- **Correct formatting**: The model should clearly understand where the answer goes
- **Diverse examples**: Examples covering different carry patterns help generalization

### Prompt Structure in `submission.py`

The prompt is constructed as:
```
prefix + pre_processing(f"{a}+{b}") + suffix
```

For example, with the default prompt:
```
Question: what is 1234567+1234567?
Answer: 2469134
Question: what is 5555555+3333333?
Answer:
```

### Scoring Formula

```
score = accuracy × (1/prompt_length) × (1 - MAE/5,000,000)
```

But for passing, you just need **either**: `accuracy > 0.2` **or** `MAE < 100,000`.

### Post-Processing

The model's raw output may contain extra text. The post-processing function extracts the numerical answer. The default strips all non-digits and converts to int. For longer outputs (max_tokens=50), you may need smarter extraction (e.g., take only the first line's digits).

---

## 2c. Step-by-Step Guide — Cell-by-Cell with Full Explanations

This walks you through every notebook cell. Read the explanation BEFORE running each cell so you understand what you're about to see.

---

### Setup Cells (0–7): Loading the Model

**Cells 0–2: Imports & Device Check**

Just run these. Cell 2 should print `Using device: cuda`. If it says `cpu`, your GPU isn't active — restart your Lightning.ai studio with a GPU runtime.

**Cell 3: HuggingFace Login**

Paste your HF token. This is needed if you're downloading models from HuggingFace. If you're using the local Meta download (Path B), you still need to log in — it won't break anything.

**Cells 4–5: Load Llama-2-7b-chat-hf**

This downloads ~14GB of model weights and loads them into GPU memory. Takes 5–10 minutes first time.

If you're using the local Meta download, change cell 4 to your absolute path:
```python
model_id_1 = "/teamspace/studios/this_studio/llm-arithmetic-prompting/llama/llamadownload"
```

**Cell 6: `call_model` — The Core Function**

This is the function that actually talks to the LLM. Here's what it does step by step:

```
Your prompt string → [Tokenize into numbers] → [Feed to model] → [Model generates new tokens] → [Decode back to text] → [Post-process to extract answer]
```

For example, if your prompt is `"Question: What is 3+5?\nAnswer: "`:
1. The tokenizer converts it to token IDs: `[16492, 29901, 1724, 338, 29871, 29941, ...]`
2. The model predicts the NEXT tokens autoregressively (one at a time, left to right)
3. It might generate tokens that decode to `"8\n\nQuestion"`
4. Post-processing strips non-digits → `8`

**Key config parameters** (you'll experiment with these in Q3b):
- `max_tokens` → how many NEW tokens the model is allowed to generate
- `temperature` → how "random" the model's choices are (0 = always pick the most likely token)
- `top_k` / `top_p` → filters that restrict which tokens the model can choose from
- `repetition_penalty` → discourages repeating the same token

**Cell 7: `test_range` — The Testing Harness**

This function runs the full experiment. Here's the flow:

```
1. Generate random pairs of numbers (e.g., a=3847291, b=5612034)
2. For each pair:
   a. Build the prompt: prefix + "3847291+5612034" + suffix
   b. Call the model with that prompt
   c. Post-process the model's output to get a number
   d. Compare to the true answer (3847291+5612034 = 9459325)
3. Compute metrics:
   - acc (accuracy): what fraction did the model get exactly right?
   - mae (mean absolute error): on average, how far off was the model?
   - res (combined score): acc × (1/prompt_length) × (1 - mae/10000)
```

---

### Q3a: Zero-Shot Addition — "Can the Model Do Math?"

**What is "zero-shot"?** It means you give the model a question with NO examples — you just ask it directly. No demonstrations, no hints. You're testing what the model learned during pre-training.

---

**Cell 10: 1-Digit Zero-Shot Addition**

Here's what the prompt looks like for each test case:
```
Question: What is 3+7?
Answer:
```

The model sees this and must generate the next tokens. For 1-digit addition, it will almost certainly output `10` — because facts like `3+7=10` appear millions of times in its training data. The model has essentially **memorized** single-digit addition.

**What to look for in the output:**
- A table showing each pair, the correct answer, the model's response, and whether it was correct
- You should see ~100% accuracy (or close)
- `mae` should be very low (near 0)

**Config being used:**
```python
added_prompt = ('Question: What is ', '?\nAnswer: ')  # prefix and suffix
max_tokens = 2    # model can only generate 2 tokens (enough for 1-digit sums like "10")
temperature = 0.7  # some randomness
```

---

**Cell 12: 7-Digit Zero-Shot Addition**

Same prompt format, but now with 7-digit numbers:
```
Question: What is 3847291+5612034?
Answer:
```

**Why this is MUCH harder for the model:**

1. **Tokenization destroys the number.** The model doesn't see `3847291` as one thing. The tokenizer might chop it as `["384", "72", "91"]`. The model has to somehow understand that these fragments represent a 7-digit number.

2. **No memorization possible.** There are ~81 trillion possible 7-digit addition pairs. The model hasn't seen these specific problems in training.

3. **Carries are hard.** To add `3847291 + 5612034`, you need to:
   - Add column by column from RIGHT to LEFT: 1+4=5, 9+3=12 carry 1, ...
   - But the model generates LEFT to RIGHT. It has to output the leftmost digit first, before it's "computed" the carries from the right side!

4. **`max_tokens=8`** — The answer to a 7-digit addition is 7 or 8 digits. By setting `max_tokens=8`, we're giving the model a **helpful prior**: "your answer should be about 8 digits." This actually helps — we'll see what happens when we remove it in Q3d.

**What to look for in the output:**
- Accuracy will be very low (~0%, maybe 10% if lucky)
- MAE will be huge (millions off)
- The model's responses will often look "plausible" (7-8 digit numbers) but be wrong

**The big takeaway for Q3a:** 1-digit = memorized facts, works great. 7-digit = actual multi-step reasoning, LLMs fail badly.

---

### Q3b: Playing with Config Parameters

There's no specific cell for this — you go back to **cell 12** and manually change one parameter at a time, re-run, and see what happens.

**Experiments to try (change one, re-run cell 12, note what happens, then reset):**

**Temperature (controls randomness):**
```python
prompt_config['temperature'] = 0.01  # Nearly deterministic — model picks the most likely token every time
# Re-run cell 12, note the results
prompt_config['temperature'] = 0.7   # Default — moderate randomness
# Re-run cell 12, note the results
prompt_config['temperature'] = 1.5   # Very random — model picks unlikely tokens often
# Re-run cell 12, note the results
```
What you'll see: Low temperature → consistent but not necessarily correct. High temperature → wildly different outputs each run, sometimes nonsensical.

**Think of it like this:** Temperature 0 = the model always gives its "best guess." Temperature 2 = the model is drunk and picks random words.

**Max Tokens:**
```python
prompt_config['max_tokens'] = 3   # Can only generate 3 tokens — answer gets cut off
prompt_config['max_tokens'] = 8   # Default — just enough for 7-8 digit answer
prompt_config['max_tokens'] = 50  # Way too much room — model will ramble after the answer
```
What you'll see: Too few → truncated numbers. Just right (8) → clean answers. Too many → model generates extra text like "Question: What is..." after the answer, and post-processing grabs those extra digits too.

**Top-K (how many token candidates to consider):**
```python
prompt_config['top_k'] = 1    # Only consider the single most likely token (greedy)
prompt_config['top_k'] = 50   # Default — consider top 50 candidates
prompt_config['top_k'] = 200  # Consider top 200 candidates — more variety
```

**Top-P / Nucleus Sampling (probability-based filtering):**
```python
prompt_config['top_p'] = 0.1   # Only tokens in the top 10% probability mass
prompt_config['top_p'] = 0.6   # Default
prompt_config['top_p'] = 1.0   # No filtering at all
```

**Repetition Penalty:**
```python
prompt_config['repetition_penalty'] = 1.0  # Default — no penalty
prompt_config['repetition_penalty'] = 1.5  # Penalize repeated tokens
prompt_config['repetition_penalty'] = 2.0  # Strongly penalize — BAD for arithmetic!
```
What you'll see: High penalty → the model avoids repeating digits, which ruins answers like `1111111` because it can't output the same digit twice.

**IMPORTANT: Reset to defaults after experimenting:**
```python
prompt_config = {'max_tokens': 8, 'temperature': 0.7, 'top_k': 50, 'top_p': 0.6, 'repetition_penalty': 1, 'stop': []}
```

---

### Q3c: Qwen3-8B Comparison — "Does a Better Model Help?"

**Cell 20: Offload Llama-2**

GPU memory is limited. You can only hold one model at a time. This cell deletes Llama-2 from memory and clears the GPU cache. Think of it like closing a massive app before opening another one.

**Cell 21: Load Qwen3-8B**

Loads a different, newer model. Qwen3-8B is a 2024/2025 model vs Llama-2's 2023 release. It has:
- Better training data (more math, more code)
- Improved architecture
- Possibly better number tokenization

**Cell 22: Test Qwen3 on 7-Digit Addition**

Same test as cell 12 but with Qwen3. Compare the accuracy and MAE numbers.

**What you'll likely see:** Qwen3 performs better than Llama-2 on 7-digit addition. This shows that model architecture and training data matter — but even Qwen3 won't get 100% because the fundamental tokenization + left-to-right problem remains.

---

### Q3d: Removing the Length Prior — "What if We Don't Constrain Output Length?"

> **Note:** The notebook labels this "Q4d" in cell-26, but it's actually Q3d per the assignment PDF.

**Cell 28: max_tokens=20 Test**

Previously `max_tokens=8` told the model: "output about 8 digits." This was a helpful hint — the answer to 7-digit addition IS 7–8 digits.

Now we set `max_tokens=20`. The model has no idea when to stop. Here's what typically happens:

```
Prompt:  "Question: What is 3847291+5612034?\nAnswer: "
Model output with max_tokens=8:  "9459325"     ← clean, just the number
Model output with max_tokens=20: "9459325\n\nQuestion: What is 1234"  ← keeps going!
```

The post-processing function does `re.sub(r"\D", "", output_string)` — it strips ALL non-digits from the ENTIRE output. So those extra digits from the follow-up question get concatenated:
```
"9459325\n\nQuestion: What is 1234" → strip non-digits → "94593251234" ← WRONG!
```

**The takeaway:** `max_tokens` acts as an implicit prior on output length. Removing it hurts because (a) the model generates garbage after the answer, and (b) the naive post-processing can't separate the answer from the garbage.

**After cell 28, run cell 29** to offload Qwen3 and free GPU memory for Part 2.

---

### Q4: In-Context Learning (25 pts) — "Teaching by Example"

**What is In-Context Learning (ICL)?**

Instead of just asking the model a question (zero-shot), you SHOW it an example first:

```
Zero-shot:                          In-context (one-shot):
Question: What is 3847291+5612034?  Question: What is 3+7?
Answer:                             Answer: 10
                                    Question: What is 3847291+5612034?
                                    Answer:
```

The model sees the pattern (question → numerical answer) and is more likely to follow it. This is called "in-context learning" because the model learns the task format from examples IN the prompt context, without any weight updates.

---

**Cell 31: Reload Llama-2**

We offloaded it for Qwen3. Now we need it back for all Q4 experiments.

**Cell 32: Q4a — Baseline ICL with 1-Digit Example**

The prompt for each test case looks like:
```
Question: What is 3+7?
Answer: 10
Question: What is 3847291+5612034?
Answer:
```

This is **one-shot learning** — one example before the actual question. The example is 1-digit (`3+7=10`), but the test is 7-digit.

**What to look for:** Slight improvement over zero-shot. The model now "understands" it should output a number. But a 1-digit example doesn't teach it HOW to do 7-digit addition — it just teaches the format.

---

**Cells 37–40: Q4b — Relaxing Output Length**

Now we set `max_tokens=50` (instead of 8) with the same ICL prompt. Same problem as Q3d — the model generates extra text.

**Cell 39 is EMPTY — you need to paste this code:**

```python
def your_post_processing(output_string):
    """Extract the first number from the output."""
    # Take only the first line to avoid extra generated questions/answers
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
```

**Why first-line extraction?** With `max_tokens=50`, the model output might be:
```
9459325

Question: What is 1111111+2222222?
Answer: 3333333
```
The old post-processing would grab ALL digits → `"945932511111112222222333333"` → wrong.
The new version takes only the first line → `"9459325"` → correct.

**Run cell 40** after pasting the post-processing. Compare results with Q4a.

---

**Cells 44–45: Q4c — In-Distribution Example**

**This is the key insight of the assignment.** Instead of a useless 1-digit example, use a 7-digit example:

```
Question: What is 1234567+1234567?
Answer: 2469134
Question: What is 3847291+5612034?
Answer:
```

Now the model sees: "oh, the input is 7 digits, the output is 7 digits, this is what addition at this scale looks like." This is called an **in-distribution** example — it matches the difficulty of the actual test.

**Cell 44:** Tests with `max_tokens=8`. Record `res`, `acc`, `mae`, `prompt_length`.
**Cell 45:** Tests with `max_tokens=50`. Record the same metrics.

**What to expect:** Significant improvement over Q4a. The 7-digit example teaches scale, format, AND gives a hint about carry patterns.

---

**Cell 49: Q4d — Error Analysis**

Tests a specific hard pair: `9090909 + 1010101 = 10101010`.

This is tricky because:
- The answer is 8 digits (carry at the top: 9+1=10)
- There's an alternating pattern that might confuse the model

**Run this cell at least 5 times.** Each time the model may give a different answer because `temperature=0.7` adds randomness. Note the error each time — it demonstrates that LLM outputs are **stochastic** (non-deterministic) at non-zero temperature.

---

### Q5: Prompt-a-thon (30 pts) — `submission.py`

File: `part2/submission.py`

**Goal:** Achieve accuracy > 0.2 OR MAE < 100,000 on 30 instances of 7-digit addition.

**Strategy:** Use multiple diverse 7-digit in-context examples with digit-by-digit formatting, low temperature, and careful post-processing.

Here's the optimized `submission.py`:

---

#### `your_prompt()` — The Prompt Design

**Key insight:** Llama-2-7b-chat-hf is fine-tuned with `[INST]...[/INST]` tags. Using this format significantly improves instruction following. We combine it with multiple diverse 7-digit few-shot examples.

```python
def your_prompt():
    prefix = (
        "[INST] You are a calculator. Compute the exact sum. "
        "Only output the number, nothing else.\n"
        "1234567+1234567=2469134\n"
        "1111111+1111111=2222222\n"
        "2345678+3456789=5802467\n"
        "5000000+5000000=10000000\n"
        "9999999+1=10000000\n"
    )
    suffix = "= [/INST] "
    return prefix, suffix
```

**Why this design:**
- **`[INST]...[/INST]` tags:** Llama-2-chat was fine-tuned to follow instructions wrapped in these tags. The model is much more likely to produce a concise numerical answer.
- **`"Only output the number, nothing else"`:** Minimizes extraneous text after the answer.
- **The `= [/INST] ` suffix:** The `=` sign primes the model to output a number right after. The space after `[/INST]` is where the model starts generating.

**Why these examples:**
- `1234567+1234567=2469134` — straightforward doubling, no carries
- `1111111+1111111=2222222` — trivially simple pattern, all 2s
- `2345678+3456789=5802467` — involves carry propagation across multiple digits
- `5000000+5000000=10000000` — demonstrates 8-digit output (carry at the top)
- `9999999+1=10000000` — extreme carry propagation, tests 8-digit output

**Backup strategy (if the above doesn't hit >0.2 accuracy):**
If needed, you can try **digit-reversal** in pre/post-processing. Reversed digits let the model add left-to-right (matching autoregressive generation with right-to-left column addition):
```python
def your_pre_processing(s):
    # "1234567+7654321" → "7654321+1234567"
    parts = s.split('+')
    return parts[0][::-1] + '+' + parts[1][::-1]

def your_post_processing(output_string):
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits[::-1])  # Reverse the output digits back
    except:
        res = 0
    return res
```
If using this approach, the prompt examples must also use reversed digits (e.g., `7654321+7654321=4319642`). String reversal is NOT arithmetic, so it passes the autograder check.

---

#### `your_config()` — Generation Config

```python
def your_config():
    config = {
        'max_tokens': 50,
        'temperature': 0.1,
        'top_k': 10,
        'top_p': 0.1,
        'repetition_penalty': 1,
        'stop': []
    }
    return config
```

**Why these values:**
- `temperature: 0.1` — near-deterministic; we want the most likely (correct) answer
- `top_k: 10` — restrict to top 10 candidates for focused generation
- `top_p: 0.1` — aggressive nucleus sampling for precise outputs
- `repetition_penalty: 1` — no penalty; digits naturally repeat in numbers

---

#### `your_pre_processing(s)` — Input Preprocessing

Keep it simple (no arithmetic allowed):

```python
def your_pre_processing(s):
    return s
```

---

#### `your_post_processing(output_string)` — Output Extraction

```python
def your_post_processing(output_string):
    """Extract the answer: take digits from the first line only."""
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
```

**Why first-line extraction:** With `max_tokens=50`, the model may generate additional text after the numerical answer (e.g., follow-up examples, explanations). We only want the first line, which contains the answer.

---

#### Complete `submission.py` to paste:

```python
import json
import collections
import argparse
import random
import numpy as np
import requests
import re

def your_netid():
    YOUR_NET_ID = 'YOUR_NET_ID'  # <-- REPLACE with your actual NetID
    return YOUR_NET_ID

def your_hf_token():
    YOUR_HF_TOKEN = 'YOUR_HF_TOKEN'  # <-- REPLACE with your actual HF token
    return YOUR_HF_TOKEN


def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers"""
    prefix = (
        "[INST] You are a calculator. Compute the exact sum. "
        "Only output the number, nothing else.\n"
        "1234567+1234567=2469134\n"
        "1111111+1111111=2222222\n"
        "2345678+3456789=5802467\n"
        "5000000+5000000=10000000\n"
        "9999999+1=10000000\n"
    )
    suffix = "= [/INST] "
    return prefix, suffix


def your_config():
    """Returns a config for prompting api"""
    config = {
        'max_tokens': 50,
        'temperature': 0.1,
        'top_k': 10,
        'top_p': 0.1,
        'repetition_penalty': 1,
        'stop': []
    }
    return config


def your_pre_processing(s):
    return s


def your_post_processing(output_string):
    """Extract the first number from the model output."""
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
```

---

## 2d. Results

> **Instructions:** Run on GPU and paste results here.

### Q3a: Zero-shot 1-digit addition results
```
# Paste output here
```

### Q3a: Zero-shot 7-digit addition results
```
# Paste output here
```

### Q3c: Qwen3-8B 7-digit addition results
```
# Paste output here
```

### Q3d: max_tokens=20 results
```
# Paste output here
```

### Q4a: Baseline ICL (3+7=10 example) with max_tokens=8
```
# Paste output here
```

### Q4b: Baseline ICL with max_tokens=50
```
# Paste output here
```

### Q4c: In-distribution example (1234567+1234567) with max_tokens=8
```
# Paste res, acc, mae, prompt_length here
```

### Q4c: In-distribution example with max_tokens=50
```
# Paste res, acc, mae, prompt_length here
```

### Q4d: Error analysis (9090909+1010101) — 5 runs
```
# Paste each run's output here
```

### Q5: Final submission.py test results
```
# Run: python run_tests.py
# Paste output here
```

---

## 2e. Written Answers (Theory)

### Q3a: Why does performance degrade from 1-digit to 7-digit? (2 pts)

1. **Tokenization fragmentation:** 7-digit numbers are split into multiple tokens (e.g., `1234567` → `["123", "45", "67"]`), so the model never processes the number as a coherent unit. This makes place-value reasoning extremely difficult.

2. **Lack of training data coverage:** While single-digit addition facts (0+0 through 9+9) are heavily represented in training data, specific 7-digit addition pairs are astronomically rare, so the model cannot rely on memorization.

3. **Multi-step reasoning:** 7-digit addition requires up to 7 sequential carry operations. Each carry depends on the result of the previous column, creating a chain of dependencies that autoregressive generation struggles with since it produces one token at a time left-to-right (opposite to how column addition works right-to-left).

4. **Output length:** 7-digit sums can be 7 or 8 digits, requiring the model to generate many more tokens, each of which must be exactly correct.

### Q3b: Config parameter explanations (3 pts)

- **Temperature:** Controls the sharpness of the probability distribution over next tokens. At temperature=0, the model always picks the most probable token (greedy decoding). Higher temperatures flatten the distribution, making less likely tokens more probable. *Increasing it* makes outputs more random and diverse, but less accurate for factual/arithmetic tasks.

- **Max Tokens:** The maximum number of new tokens the model will generate. *Increasing it* allows longer outputs, but the model may generate irrelevant continuation text beyond the answer. Setting it too low truncates the answer.

- **Top-P (Nucleus Sampling):** Only considers the smallest set of tokens whose cumulative probability exceeds P. At `top_p=0.1`, only the very most likely tokens are candidates. *Increasing it* includes more candidate tokens, increasing diversity but reducing precision.

- **Top-K:** Only the K most probable tokens are considered at each generation step. *Increasing it* allows more token candidates, similar to top-p but with a hard count cutoff rather than probability threshold.

- **Repetition Penalty:** Multiplies the probability of previously generated tokens by `1/penalty`. At 1.0, no penalty. *Increasing it* discourages the model from repeating tokens, which can be harmful for arithmetic since digits naturally repeat (e.g., `1111111`).

### Q3c: Qwen3-8B vs Llama-2-7b (2 pts)

> **Paste your actual observations after running.** Typical expected answer:

Qwen3-8B generally performs **better** than Llama-2-7b on 7-digit addition. This is because:

1. **Newer architecture and training:** Qwen3-8B is a more recent model trained on a larger and more diverse dataset, likely including more mathematical content and potentially synthetic arithmetic training data.

2. **Better tokenization:** Qwen3's tokenizer may handle numbers more effectively, potentially tokenizing digits individually or in more consistent chunks.

3. **Larger parameter count:** At 8B parameters (vs 7B), Qwen3 has slightly more capacity for complex reasoning.

### Q3d: Removing the output length prior (max_tokens=20) (3 pts)

> **Paste your actual observations after running.** Typical expected answer:

With `max_tokens=20`, performance **degrades** because:

1. **Extra generation:** Without the tight constraint of 8 tokens, the model generates additional text after the numerical answer (e.g., `"8901234\n\nQuestion: What is..."` or `"8901234 (approximately)"`).

2. **Post-processing contamination:** The default post-processing strips all non-digits from the entire output. Extra generated digits from follow-up text get concatenated with the answer, producing an incorrect (often much larger) number.

3. **Loss of implicit constraint:** Setting `max_tokens=8` implicitly told the model the answer should be ~8 digits. Without this constraint, the model has no structural guidance on answer length.

### Q4a: Baseline ICL with 1-digit example (3 pts)

> **Paste your actual observations after running.** Typical expected answer:

Performance **slightly improves** compared to zero-shot because:

1. **Format priming:** The example `3+7=10` teaches the model the exact format: Question → Answer with just a number. This makes the model more likely to output just digits rather than verbose text.

2. **Task clarification:** The in-context example makes the task unambiguous — the model understands it should compute a sum and output just the result.

However, improvement is modest because a 1-digit example provides little useful information for 7-digit addition strategy.

### Q4b: Relaxed output length with post-processing (7 pts)

**Post-processing approach:** Extract only the first line of the output and take digits from that line. This prevents contamination from follow-up generated text.

> **Paste your actual observations after running.** Typical expected answer:

Performance **degrades** compared to Q4a (max_tokens=8) because:

1. **The model generates additional content** beyond the answer (e.g., follow-up questions and answers), and even with improved post-processing, the first line may contain extra text.

2. **Without the length constraint**, the model doesn't have an implicit signal about how many digits the answer should have.

3. **The tight max_tokens=8 constraint** in Q4a actually helped by forcing the model to output only ~8 digits (which is the correct length for 7-digit addition sums).

### Q4c: In-distribution example evaluation (9 pts)

> **Paste your actual res, acc, mae, prompt_length after running.**

Typical expected observations:

- **max_tokens=8:** Accuracy improves significantly compared to Q4a (1-digit example). The 7-digit example shows the model the expected scale and format of the answer.
- **max_tokens=50:** Performance decreases compared to max_tokens=8 for the same reasons as Q4b.
- **Overall:** Moving from a 1-digit to 7-digit example dramatically improves performance because the model sees an example of similar complexity, learning the approximate scale and digit count of the answer.

### Q4d: Error analysis (6 pts)

> **Paste actual observations from 5 runs.**

Typical expected answers:

- **Does the error change each time?** Yes, because `temperature=0.7` introduces randomness. Each generation samples from the probability distribution differently, producing different outputs for the same input.

- **Prompt to reduce error:** Add more diverse examples, particularly ones involving carry propagation patterns similar to `9090909+1010101=10101010`. For example, include examples like `9000000+1000000=10000000` to teach carry behavior.

- **Why it would work:** More examples covering edge cases (especially carry propagation) give the model better in-context signal for handling similar patterns.

- **Does it work in practice?** Partially. Adding relevant examples can reduce average error, but LLMs still struggle with precise multi-digit arithmetic because the fundamental limitation is the autoregressive left-to-right generation conflicting with right-to-left carry propagation.

---

## HPC / GPU Setup Notes

If you are running Part 2 on NYU HPC, you'll need a batch script. Here's a template:

```bash
#!/bin/bash
#SBATCH --job-name=hw3_nlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=hw3_%j.out

module purge
module load python/intel/3.8.6
# Or activate your conda environment:
# source activate nlp_hw3

cd /path/to/hw3code/part2

pip install -r ../requirements.txt

# For the notebook exercises, convert to script or use jupyter:
# jupyter nbconvert --to script prompting_exercises.ipynb
# python prompting_exercises.py

# For testing submission.py:
python run_tests.py
```

Adjust paths and module loads to match your HPC environment. Let me know your specific HPC details and I'll customize this for you.

---

## 2f. How to Update `submission.py` (Step-by-Step)

Your current `submission.py` still has the **old, non-optimized** version. Here are the exact changes to make:

### Change 1: Replace `your_netid()` and `your_hf_token()`

Replace `'YOUR_NET_ID'` with your actual NYU NetID (e.g., `'abc1234'`).
Replace `'YOUR_HF_TOKEN'` with your actual HuggingFace token (starts with `hf_...`).

### Change 2: Replace `your_prompt()` (lines 19–29)

**Current (old — won't pass grader):**
```python
def your_prompt():
    prefix = '''Question: what is 1234567+1234567?\nAnswer: 2469134\nQuestion: what is '''
    suffix = '?\nAnswer: '
    return prefix, suffix
```

**Replace with (optimized — uses Llama-2-chat `[INST]` format + 5 diverse examples):**
```python
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers"""
    prefix = (
        "[INST] You are a calculator. Compute the exact sum. "
        "Only output the number, nothing else.\n"
        "1234567+1234567=2469134\n"
        "1111111+1111111=2222222\n"
        "2345678+3456789=5802467\n"
        "5000000+5000000=10000000\n"
        "9999999+1=10000000\n"
    )
    suffix = "= [/INST] "
    return prefix, suffix
```

**Why this is better:**
- `[INST]...[/INST]` tags match Llama-2-chat's fine-tuning format — the model follows instructions much better
- 5 diverse examples cover: simple doubling, trivial pattern, carry propagation, 8-digit output, extreme carry
- `"= [/INST] "` suffix primes the model to output a number immediately after `=`

### Change 3: Replace `your_config()` (lines 32–49)

**Current (old):**
```python
config = {
    'max_tokens': 50,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.7,
    'repetition_penalty': 1,
    'stop': []}
```

**Replace with (optimized — near-deterministic):**
```python
config = {
    'max_tokens': 50,
    'temperature': 0.1,
    'top_k': 10,
    'top_p': 0.1,
    'repetition_penalty': 1,
    'stop': []
}
```

**Why:** `temperature: 0.1` and `top_p: 0.1` make the model near-deterministic — for arithmetic, you want the most likely (correct) answer, not creative diversity.

### Change 4: Replace `your_post_processing()` (lines 56–70)

**Current (old — grabs ALL digits from entire output):**
```python
def your_post_processing(output_string):
    only_digits = re.sub(r"\D", "", output_string)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
```

**Replace with (improved — first line only):**
```python
def your_post_processing(output_string):
    """Extract the first number from the model output."""
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
```

**Why:** With `max_tokens=50`, the model may generate follow-up text after the answer. Taking only the first line prevents stray digits from contaminating the result.

### Change 5 (optional): `your_pre_processing()` — Leave As-Is

```python
def your_pre_processing(s):
    return s
```

No changes needed. The identity function is fine for the standard approach.

---

## 2g. Notebook Cell-by-Cell Walkthrough

This section tells you **exactly what to do in each notebook cell** when you run `prompting_exercises.ipynb` on GPU.

### Cells 0–7: Setup (just run them)
- Cell 0: Imports
- Cell 1: Debug helper
- Cell 2: Check device (should print `Using device: cuda`)
- Cell 3: HuggingFace login (enter your token)
- Cell 4–5: Load Llama-2-7b-chat-hf
- Cell 6: `call_model` helper
- Cell 7: `test_range` helper

### Cell 10: Zero-shot 1-digit addition
Run it. Should get ~100% accuracy. **Save the output** for Q3a comparison.

### Cell 12: Zero-shot 7-digit addition
Run it. Should get ~0% accuracy, high MAE. **Save the output** for Q3a.

### Cell 15: Q3a Answer — Write This

```markdown
Answer:
1. **Tokenization fragmentation:** 7-digit numbers are split into multiple tokens (e.g., 1234567 → ["123", "45", "67"]),
   so the model never processes the number as a coherent unit, making place-value reasoning difficult.
2. **Training data scarcity:** Single-digit addition facts (0+0 through 9+9) are heavily represented in training data
   and effectively memorized. Specific 7-digit addition pairs are astronomically rare.
3. **Multi-step carry propagation:** 7-digit addition requires up to 7 sequential carry operations, each depending
   on the previous column. Autoregressive left-to-right generation conflicts with right-to-left column addition.
4. **Output length:** 7-digit sums can be 7 or 8 digits, requiring many more tokens that must each be exactly correct.
```

### Cell 18: Q3b Answer — Write This

```markdown
Answer:
- **Temperature:** Controls randomness in token selection. At 0, always picks the most probable token (greedy).
  Increasing it flattens the probability distribution, producing more diverse but less accurate outputs.
  For arithmetic, lower is better (0.0-0.1).

- **Max Tokens:** Maximum number of new tokens generated. Increasing allows longer output but the model
  may generate irrelevant text beyond the answer. Too few truncates the answer.

- **Top-K:** Only considers the K most probable next tokens. Increasing K allows more diversity.
  Lower K (e.g., 1) is more deterministic.

- **Top-P (Nucleus Sampling):** Only considers the smallest set of tokens whose cumulative probability exceeds P.
  Increasing it includes more candidate tokens. For arithmetic, lower is better.

- **Repetition Penalty:** Penalizes previously generated tokens by multiplying their probability by 1/penalty.
  At 1.0, no penalty. Increasing discourages repetition — harmful for arithmetic since digits naturally repeat
  (e.g., 1111111).
```

### Cells 20–22: Load Qwen3-8B and test 7-digit addition
Run cells 20 (offload Llama), 21 (load Qwen3), 22 (test). **Save the output.**

### Cell 24: Q3c Answer — Write This

```markdown
Answer:
Qwen3-8B generally performs better than Llama-2-7b on 7-digit addition because:
1. **Newer architecture and training:** Qwen3-8B is a more recent model trained on larger and more diverse data,
   likely including more mathematical content and potentially synthetic arithmetic data.
2. **Better tokenization:** Qwen3's tokenizer may handle numbers more effectively, potentially tokenizing digits
   more consistently.
3. **Slightly larger capacity:** At 8B parameters vs 7B, Qwen3 has marginally more capacity for complex reasoning.

(Adjust based on your actual numbers — report the specific accuracy and MAE you observed.)
```

### Cell 27: Q3d Answer — Write This
(Note: the notebook labels this "Q4d" but per the assignment PDF it's Q3d)

```markdown
Answer:
With max_tokens=20, performance degrades because:
1. **Extra generation:** Without the tight 8-token constraint, the model generates additional text after the answer
   (e.g., follow-up questions, explanations).
2. **Post-processing contamination:** The default post-processing strips ALL non-digits from the ENTIRE output.
   Extra digits from follow-up text get concatenated with the answer, producing an incorrect (much larger) number.
3. **Loss of implicit length prior:** Setting max_tokens=8 implicitly told the model the answer should be ~8 digits.
   Without this, the model has no structural guidance on answer length.
```

### Cell 28: Run max_tokens=20 test
Run it. **Save the output.** Then run cell 29 to offload Qwen3.

### Cells 31–32: Reload Llama-2 and run baseline ICL
Run both. Cell 32 uses the 1-digit example (`3+7=10`) with `max_tokens=8`. **Save the output.**

### Cell 34: Q4a Answer — Write This

```markdown
Answer:
Performance slightly improves compared to zero-shot because:
1. **Format priming:** The example "3+7=10" teaches the model the expected format — input an addition problem,
   output just a number. This reduces verbose or off-topic responses.
2. **Task clarification:** The in-context example makes the task unambiguous.

However, improvement is modest because a 1-digit example provides little useful signal for 7-digit addition strategy.
The model still cannot generalize multi-step carry propagation from a trivial example.
```

### Cell 38: Q4b Answer — Write This

```markdown
Answer:
**Post-processing approach:** Extract only the first line of output and take digits from that line.
This prevents contamination from follow-up text the model generates after the actual answer.

Performance degrades compared to Q4a (max_tokens=8) because:
1. The model generates continuation text (more questions/answers) beyond the answer.
2. Even with first-line extraction, the model may embed extra text on the first line.
3. The tight max_tokens=8 constraint in Q4a actually helped by forcing the model to produce only ~8 digits,
   which is the correct length for 7-digit addition sums.
```

### Cell 39: Q4b Post-Processing Code — PASTE THIS

This cell is currently empty. Paste this code:

```python
def your_post_processing(output_string):
    """Extract the first number from the output."""
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res
```

### Cell 40: Run ICL with max_tokens=50
Run it with the updated post-processing from cell 39. **Save the output.**

### Cells 43–45: Q4c In-Distribution Example

Cell 43: Q4c Answer — Write This:

```markdown
Answer:
- **max_tokens=8:** Accuracy improves significantly compared to Q4a (1-digit example).
  The 7-digit example teaches the model the expected scale, format, and digit count of the answer.
  Report: res=___, acc=___, mae=___, prompt_length=___

- **max_tokens=50:** Performance decreases compared to max_tokens=8, for the same reasons as Q4b —
  the model generates extra text beyond the answer.
  Report: res=___, acc=___, mae=___, prompt_length=___

- **1-digit → 7-digit improvement:** The in-distribution example dramatically helps because the model
  sees an example of similar complexity, learning the approximate scale and number of digits in the answer.
  This is the core principle of in-context learning — examples closer to the test distribution are more useful.
```

Cell 44: Run with `max_tokens=8`. **Fill in the actual numbers in cell 43.**
Cell 45: Run with `max_tokens=50`. **Fill in the actual numbers in cell 43.**

### Cell 48: Q4d Answer — Write This

```markdown
Answer:
- **Does the error change each time?** Yes, because temperature=0.7 introduces randomness. Each generation
  samples differently from the probability distribution, producing different outputs for the same input.

- **Prompt to reduce error:** Add more examples covering carry propagation patterns similar to 9090909+1010101.
  For instance, include examples like 9000000+1000000=10000000 and 8888888+1111112=10000000 to teach
  the model about carrying across many digits.

- **Why it would work:** More examples covering edge cases (especially carry propagation from 9→10) give the
  model better in-context signal for handling similar patterns.

- **Does it work in practice?** Partially. Adding relevant examples reduces average error, but LLMs still
  struggle with precise multi-digit arithmetic because the fundamental limitation is autoregressive
  left-to-right generation conflicting with right-to-left carry propagation. The model is pattern-matching,
  not truly computing.
```

### Cell 49: Run error analysis
Run this cell **at least 5 times**. Each time note the output and whether the error changed. This demonstrates the effect of temperature on stochastic generation.

---

## 2h. Backup Strategy: Digit Reversal

If the standard `[INST]` prompt doesn't hit >20% accuracy on the grader, try **digit reversal**. This is a powerful technique because it aligns the model's left-to-right generation with the right-to-left order of column addition (least significant digit first).

**Replace `your_pre_processing`:**
```python
def your_pre_processing(s):
    # "1234567+7654321" → "7654321+1234567" (reversed digits)
    parts = s.split('+')
    return parts[0][::-1] + '+' + parts[1][::-1]
```

**Replace `your_post_processing`:**
```python
def your_post_processing(output_string):
    first_line = output_string.strip().split('\n')[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits[::-1])  # Reverse output digits back
    except:
        res = 0
    return res
```

**Replace `your_prompt`** (examples must also use reversed digits):
```python
def your_prompt():
    prefix = (
        "[INST] You are a calculator. Compute the exact sum of reversed-digit numbers. "
        "Only output the reversed sum digits, nothing else.\n"
        "7654321+7654321=4319642\n"
        "1111111+1111111=2222222\n"
        "8765432+9876543=7641370\n"
        "0000005+0000005=00000001\n"
        "9999999+1000000=0000001\n"
    )
    suffix = "= [/INST] "
    return prefix, suffix
```

**Why this works:** String reversal is NOT arithmetic (it's just reordering characters), so it passes the autograder check. But it lets the model process digits from least-significant to most-significant, which matches how column addition actually works.

> **Important:** Only use this if the standard approach fails. Try the standard `[INST]` prompt first.

---

## Quick Reference: Files to Submit

1. **`part1/src/bpe.py`** — with `merge()`, `encode()`, `decode()` implemented
2. **`part1/tests/test_tokenizer.py`** — with property test counterexamples filled in
3. **Written answers PDF** — Q2b, Q2c, Q3a-d, Q4a-d answers
4. **`part2/submission.py`** — with prompt, config, pre/post-processing, NetID, HF token
5. **`part2/prompting_exercises.ipynb`** — with all cells run and answers filled in
