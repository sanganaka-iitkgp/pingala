
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig
from skrutable.scansion import Scanner
from skrutable.meter_patterns import anuzwuB_pAda
import re
import argparse
import pandas as pd
import gc
import os
from datetime import datetime
from functools import lru_cache, partial
from typing import List, Dict


def apply_phi4_chat_template(user_message):
    return f"<|im_start|>user<|im_sep|>{user_message}<|im_end|>\n<|im_start|>assistant<|im_sep|>"


SLP1_VOWELS = set('aAiIuUfFxXeEoO')
SLP1_CONSONANTS = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh')
SLP1_SPECIAL = set('MH~')
SLP1_VALID_CHARS = SLP1_VOWELS | SLP1_CONSONANTS | SLP1_SPECIAL

def is_slp1_token(token):
    return all(char in SLP1_VALID_CHARS or not char.strip() for char in token)


def has_excessive_conjuncts(text, max_half_akshars=3):
    consecutive_consonants = 0
    for char in text:
        if char in SLP1_CONSONANTS:
            consecutive_consonants += 1
            if consecutive_consonants > max_half_akshars:
                return True
        elif char in SLP1_VOWELS or char in SLP1_SPECIAL:
            consecutive_consonants = 0
        else:
            consecutive_consonants = 0
    return False


class AnushtupConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length, k=100, no_vipula=True, only_567_constraint=False):
        self.tokenizer = tokenizer
        self.s = Scanner()
        self.k = k
        self.prompt_length = prompt_length
        self.rgx_prefix_helper = re.compile(r'{\d}')
        self.make_partial_regexes(no_vipula, only_567_constraint)

        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        self.scan = lru_cache(maxsize=1000)(partial(self.s.scan, from_scheme='SLP'))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            current_ids = input_ids[b].tolist()
            generated_ids = current_ids[self.prompt_length:]

            if not generated_ids:
                continue

            current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            best_scores, best_options = scores[b].topk(min(self.k, len(scores[b])))
            max_best_score = best_scores.max()

            options_text = self.tokenizer.batch_decode(best_options, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            found_any_anushtup, scores[b] = self.call_onesample_topk(
                self.k, current_text, options_text, scores[b], best_options, max_best_score, generated_ids
            )

            if not found_any_anushtup and len(current_text) > 4:
                best_scores_i, best_options_i = scores[b].topk(min(10*self.k, len(scores[b])))
                options_text = self.tokenizer.batch_decode(best_options_i, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                found_any_anushtup, scores[b] = self.call_onesample_topk(
                    10*self.k, current_text, options_text, scores[b], best_options_i, max_best_score, generated_ids
                )

        return scores

    def call_onesample_topk(self, k: int, x: str, options_text: List[str], scores: torch.FloatTensor, best_options: torch.Tensor, max_best_score, generated_ids):
        found_any_anushtup = False
        best = None

        try:
            syllables_old = self.scan(x)
            l = len(syllables_old.syllable_weights.replace('\n', ''))
        except:
            l = 0

        for j in range(min(k, len(options_text))):
            try:
                syllables = self.scan(x + options_text[j])
            except Exception:
                scores[best_options[j]] = -torch.inf
                continue

            if syllables.syllable_weights is None:
                scores[best_options[j]] = -torch.inf
                continue

            is_anushtup = self.decide(syllables.syllable_weights)
            l_new = len(syllables.syllable_weights.replace('\n',''))

            if l_new > 32:
                is_anushtup = False

            if l_new - l > 1 and l//8 != l_new//8:
                is_anushtup = False

            if options_text[j].strip() and not is_slp1_token(options_text[j]):
                is_anushtup = False

            if is_anushtup and has_excessive_conjuncts(x + options_text[j]):
                is_anushtup = False

            if not is_anushtup:
                scores[best_options[j]] = -torch.inf
            else:
                found_any_anushtup = True
                if best is None:
                    best = j

        if l < 32 and self.eos_id in best_options:
            scores[self.eos_id] = -torch.inf
        elif l >= 32:
            scores[self.eos_id] = max_best_score + 1

        return found_any_anushtup, scores

    def decide(self, syllables: str) -> bool:
        halves = syllables.split('\n')
        padas = []
        padas += [halves[0][:8], halves[0][8:16]] if len(halves[0]) > 8 else [halves[0]]

        if len(halves) == 1:
            if len(halves[0]) > 16:
                padas += [halves[0][16:24]]
            if len(halves[0]) > 24:
                padas += [halves[0][24:]]
        elif len(halves) > 1:
            padas += [halves[1][:8], halves[1][8:]] if len(halves[1]) > 8 else [halves[1]]

        l = len(padas)
        pada = padas[-1]
        pada_len = len(pada)

        if pada_len == 0:
            return True
        if pada_len > 8:
            return False

        if l % 2 == 1:
            return any([odd_rgx[pada_len-1].match(pada) is not None for odd_rgx in self.odd_regexes.values()])
        else:
            return self.even_regex[pada_len-1].match(pada) is not None

    def make_partial_regexes(self, no_vipula: bool, only_567_constraint: bool):
        def prefix_helper(regex: str):
            if regex.startswith('^') and regex.endswith('$'):
                regex = regex[1:-1]
            generated_partials = []
            if (match := re.search(self.rgx_prefix_helper, regex)) is not None:
                generated_partials = [f'^{regex[:match.start()]}{{{x+1}}}$' for x in range(int(match.group()[1]))]
            return generated_partials

        even = anuzwuB_pAda['even']
        if only_567_constraint:
            even = '^.{4}lgl.$'
        even_regex = prefix_helper(even)
        even_regex += [even[:-x] + '$' for x in range(8-len(even_regex),1,-1)]
        even_regex += [even]
        self.even_regex = [re.compile(rgx) for rgx in even_regex]

        odd_regexes: Dict[str, List[re.Pattern[str]]] = {}
        for odd_rgx, name in anuzwuB_pAda['odd'].items():
            if no_vipula and name != 'pathyā':
                continue
            if only_567_constraint:
                odd_rgx = '^.{4}lgg.$'
            odd_rgxs = prefix_helper(odd_rgx)
            odd_rgxs += [odd_rgx[:-x] + '$' for x in range(8-len(odd_rgxs),1,-1)]
            odd_rgxs += [odd_rgx]
            odd_regexes[name] = [re.compile(rgx) for rgx in odd_rgxs]
        self.odd_regexes = odd_regexes


class AnushtupChecker:
    def __init__(self):
        self.scanner = Scanner()
        self.cache = {}
        self.slp1_pattern = re.compile(r'[a-zA-Z~]+')

    def clear_cache(self):
        self.cache.clear()
        gc.collect()

    def clean_text(self, text):
        text = text.replace('.', ' ').replace('|', ' ').replace('/', ' ').replace('\n', ' ')
        matches = self.slp1_pattern.findall(text)
        return ' '.join(matches)

    def has_valid_sanskrit(self, text):
        clean = self.clean_text(text)
        if not clean:
            return False
        slp1_chars = len([c for c in text if c.isalpha()])
        total_chars = len([c for c in text if c.strip()])
        return total_chars > 0 and (slp1_chars / total_chars) > 0.7

    def check(self, text):
        clean = self.clean_text(text)
        if not clean:
            return (True, "Empty", 0, [])

        if clean in self.cache:
            return self.cache[clean]

        try:
            syl = self.scanner.scan(clean, from_scheme='SLP')
            if not syl.syllable_weights:
                result = (True, "Empty", 0, [])
                self.cache[clean] = result
                return result
            pat = syl.syllable_weights.replace('\n', '').replace(' ', '')
        except:
            result = (True, "Scan failed", 0, [])
            self.cache[clean] = result
            return result

        total = len(pat)
        if total > 32:
            result = (False, "Too long", total, [])
            self.cache[clean] = result
            return result

        num_padas = total // 8
        pada_status = []

        for i in range(num_padas):
            pada_pat = pat[i*8:(i+1)*8]
            ending = pada_pat[4:7]
            expected = 'lgg' if i % 2 == 0 else 'lgl'
            is_valid = (ending == expected)
            pada_status.append({'pada': i+1, 'pattern': pada_pat, 'ending': ending, 'expected': expected, 'valid': is_valid})
            if not is_valid:
                result = (False, f"P{i+1}:{ending}!={expected}", total, pada_status)
                self.cache[clean] = result
                return result

        remaining = total % 8
        if remaining >= 7 and num_padas < 4:
            partial_pat = pat[num_padas*8:]
            expected = 'lgg' if num_padas % 2 == 0 else 'lgl'
            if partial_pat[4:7] != expected:
                result = (False, "Partial bad", total, pada_status)
                self.cache[clean] = result
                return result

        if total == 32 and num_padas == 4:
            result = (True, "Perfect", total, pada_status)
        else:
            result = (True, f"Valid ({num_padas}p)", total, pada_status)

        self.cache[clean] = result
        return result


ANUSHTUP_INSTRUCTION = """### INSTRUCTION:
The goal is to generate Sanskrit verse that follows the anushtup meter rules for the given input translation.
RULES:
Verse Rules:
The verse contains 32 syllables/akshara and 4 padas in total.
The verse is divided in 2 lines, each containing 16 syllables.
Each line is divided into 2 padas (quartets),each containing exactly 8 syllables.
The fifth syllable of every pada must be LAGHU or short.
The sixth syllable of every pada must be GURU or long.
The seventh syllable of the second and the fourth pada must be HRASVA
The seventh letter of the first and third paada must be DEERGHA

Syllable Rules:
LAGHU vowels: a, i, u, f, x
GURU vowels: A, I, U, F, X, e, E, o, O
HRASVA vowels: a, i, u, f, x
DEERGHA vowels: A, I, U, F, X, e, E, o, O

Syllable is marked laghu, guru and hrasva, deergha based on the vowel it contains.
Syllable containing anusvAra("M") or visarga("H") is always marked as guru.
Syllable that is followed by a conjunct consonant (saMyuktAkzara) is always marked guru."""


def create_simple_prompt(english_text):
    user_message = f"{ANUSHTUP_INSTRUCTION}\n\nEnglish: {english_text}\n\nSanskrit Anushtubh:"
    return apply_phi4_chat_template(user_message)


def create_anushtup_prompt(english_text):
    user_message = f"{ANUSHTUP_INSTRUCTION}\n\nEnglish: {english_text}\n\nSanskrit Anushtubh:"
    return apply_phi4_chat_template(user_message)


def generate_simple_translation(model, tokenizer, text, device="cuda"):
    prompt = create_simple_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_length = len(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=80,
            num_beams=5,
            num_return_sequences=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    translations = []
    for out in outputs:
        generated_ids = out[prompt_length:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if decoded:
            translations.append(decoded)

    del outputs, inputs
    torch.cuda.empty_cache()

    return translations[0] if translations else ""


def generate_constrained_anushtup(model, tokenizer, text, checker, device="cuda"):
    prompt = create_anushtup_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_length = len(inputs["input_ids"][0])

    logits_processor = LogitsProcessorList([
        AnushtupConstrainedLogitsProcessor(tokenizer, prompt_length, k=100, no_vipula=True, only_567_constraint=False)
    ])

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=25,
            num_return_sequences=8,
            max_new_tokens=100,
            logits_processor=logits_processor,
            early_stopping=False,
            no_repeat_ngram_size=2,
            length_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in outputs:
        generated_ids = out[prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not generated_text or not checker.has_valid_sanskrit(generated_text):
            continue

        if has_excessive_conjuncts(generated_text):
            continue

        is_valid, reason, count, pada_status = checker.check(generated_text)
        if is_valid and count == 32:
            results.append(generated_text)

    del outputs, inputs
    torch.cuda.empty_cache()

    return results[0] if results else ""


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Anushtup Generator (Phi-4, SLP1)")
    parser.add_argument("--input_csv", type=str, default="input.csv", help="Input CSV with 'English' column")
    parser.add_argument("--output_csv", type=str, default="translations_output.csv", help="Output CSV path")
    parser.add_argument("--model_name", type=str, default="manojbalaji1/phi4-slp1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded: {args.model_name}")

    checker = AnushtupChecker()

    df_in = pd.read_csv(args.input_csv)
    if 'English' not in df_in.columns:
        raise ValueError(f"CSV must have 'English' column. Found: {df_in.columns.tolist()}")
    all_inputs = df_in['English'].tolist()

    total_samples = len(all_inputs)
    output_file = args.output_csv

    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        start_idx = len(df_existing)
        translations = df_existing['Sanskrit'].tolist()
        print(f"Resuming from sample {start_idx + 1}")
    else:
        start_idx = 0
        translations = []

    start_time = datetime.now()
    perfect_count = 0
    fallback_count = 0

    for idx in range(start_idx, total_samples):
        text = all_inputs[idx]

        try:
            anushtup_verse = generate_constrained_anushtup(model, tokenizer, text, checker, device)

            if anushtup_verse:
                translations.append(anushtup_verse)
                perfect_count += 1
            else:
                simple_trans = generate_simple_translation(model, tokenizer, text, device)
                translations.append(simple_trans)
                fallback_count += 1

            df_out = pd.DataFrame({
                'English': all_inputs[:len(translations)],
                'Sanskrit': translations
            })
            df_out.to_csv(output_file, index=False)

            if (idx + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / (idx - start_idx + 1)
                remaining = (total_samples - idx - 1) * avg_time
                print(f"Sample {idx+1}/{total_samples} | Perfect: {perfect_count} | Fallback: {fallback_count} | {remaining/60:.1f} min remaining")
                checker.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR at sample {idx + 1}: {str(e)}")
            df_out = pd.DataFrame({
                'English': all_inputs[:len(translations)],
                'Sanskrit': translations
            })
            df_out.to_csv(output_file, index=False)
            raise

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Complete. Output: {output_file} | {len(translations)} samples | Perfect: {perfect_count} | Fallback: {fallback_count} | {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
