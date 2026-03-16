import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessor, LogitsProcessorList
from skrutable.scansion import Scanner
from skrutable.meter_patterns import anuzwuB_pAda
from skrutable.meter_identification import VerseTester
import re
import argparse
import pandas as pd
import gc
import os
from datetime import datetime
from functools import lru_cache, partial
from typing import List, Dict


devanagari_range = range(0x0900, 0x097F + 1)

def is_sanskrit_token(token):
    return all(ord(char) in devanagari_range for char in token)


class AnushtupConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, k=100, no_vipula=True, only_567_constraint=False):
        self.tokenizer = tokenizer
        self.s = Scanner()
        self.k = k
        self.rgx_prefix_helper = re.compile(r'{\d}')
        self.make_partial_regexes(no_vipula, only_567_constraint)

        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.space_token_id = 248059
        self.newverse_token_id = 615
        self.newverse_token = tokenizer.decode(self.newverse_token_id)
        self.endverse_token_id = 8622
        self.endverse_token = tokenizer.decode(self.endverse_token_id)
        self.prevent_token_ids = torch.tensor([255375, 249912, 248225])
        self.nonincr_token_deny = [self.newverse_token_id, self.endverse_token_id, self.space_token_id, *self.prevent_token_ids.tolist()]

        self.scan = lru_cache(maxsize=1000)(partial(self.s.scan, from_scheme='DEV'))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_text: List[str] = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        best_scores, best_options = scores.topk(self.k)
        max_best_score = best_scores.max()

        for i, x in enumerate(input_text):
            options_text = self.tokenizer.batch_decode(best_options[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            found_any_anushtup, scores[i] = self.call_onesample_topk(self.k, x, options_text, scores[i], best_options[i], max_best_score)

            if not found_any_anushtup and len(x) > 4:
                best_scores_i, best_options_i = scores[i].topk(10*self.k)
                options_text = self.tokenizer.batch_decode(best_options_i, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                found_any_anushtup, scores[i] = self.call_onesample_topk(10*self.k, x, options_text, scores[i], best_options_i, max_best_score)

        return scores

    def call_onesample_topk(self, k: int, x: str, options_text: List[str], scores: torch.FloatTensor, best_options: torch.Tensor, max_best_score):
        found_any_anushtup = False
        best = None
        syllables_old = self.scan(x)
        l = len(syllables_old.syllable_weights.replace('\n', ''))

        for j in range(k):
            try:
                syllables = self.scan(x + options_text[j])
            except Exception:
                scores[best_options[j]] = -torch.inf
                continue

            assert syllables.syllable_weights is not None
            is_anushtup = self.decide(syllables.syllable_weights)
            l_new = len(syllables.syllable_weights.replace('\n',''))

            if l_new - l > 1 and l//8 != l_new//8:
                is_anushtup = False

            if l in [8,24] and l_new == l and x[-1] == ' ':
                is_anushtup = False

            if not is_sanskrit_token(options_text[j]) and options_text[j] != 'san_Deva':
                is_anushtup = False

            if not is_anushtup:
                scores[best_options[j]] = -torch.inf
            else:
                found_any_anushtup = True
                if l_new in [8, 16, 24, 32]:
                    scores[best_options[j]] += 5.0
                if best is None:
                    best = j

        try:
            if best is not None and best_options[best] not in self.nonincr_token_deny:
                l_best = len(self.scan(x + options_text[best]).syllable_weights.replace('\n', ''))
            else:
                l_best = -1

            scores[self.newverse_token_id] = -torch.inf
            scores[self.endverse_token_id] = -torch.inf
            scores[self.prevent_token_ids] = -torch.inf

            if l < 32 and self.eos_id in best_options:
                scores[self.eos_id] = -torch.inf
            elif l >= 32:
                if self.endverse_token not in x:
                    scores[self.endverse_token_id] = max_best_score + 1
                    found_any_anushtup = True
                elif not (x.endswith(self.tokenizer.eos_token) or x.endswith(self.tokenizer.pad_token)):
                    scores[self.eos_id] = max_best_score + 1

            if l == 16 and self.newverse_token not in x[-4:] and l_best != l:
                scores[self.newverse_token_id] = max_best_score + 1

            if scores[self.space_token_id] == scores.max():
                scores[self.space_token_id] = -torch.inf

            if (l == 8 or l == 24) and x[-1] != ' ' and l_best != l:
                scores[self.space_token_id] = max_best_score + 1

        except IndexError:
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
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')

    def clear_cache(self):
        self.cache.clear()
        gc.collect()

    def clean_text(self, text):
        text = text.replace('।', ' ').replace('॥', ' ').replace('\n', ' ')
        matches = self.devanagari_pattern.findall(text)
        return ' '.join(matches)

    def has_valid_sanskrit(self, text):
        clean = self.clean_text(text)
        if not clean:
            return False
        devanagari_chars = len([c for c in text if '\u0900' <= c <= '\u097F'])
        total_chars = len([c for c in text if c.strip()])
        return total_chars > 0 and (devanagari_chars / total_chars) > 0.8

    def check(self, text):
        clean = self.clean_text(text)
        if not clean:
            return (True, "Empty", 0, [])

        if clean in self.cache:
            return self.cache[clean]

        try:
            syl = self.scanner.scan(clean, from_scheme='DEV')
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
            partial = pat[num_padas*8:]
            expected = 'lgg' if num_padas % 2 == 0 else 'lgl'
            if partial[4:7] != expected:
                result = (False, "Partial bad", total, pada_status)
                self.cache[clean] = result
                return result

        if total == 32 and num_padas == 4:
            result = (True, "Perfect", total, pada_status)
        else:
            result = (True, f"Valid ({num_padas}p)", total, pada_status)

        self.cache[clean] = result
        return result


def generate_anushtup_batch(
    model, tokenizer, texts_batch, checker,
    num_beams=25, num_return=25, max_length=100, device="cuda"
):
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        texts_batch,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    logits_processor = LogitsProcessorList([
        AnushtupConstrainedLogitsProcessor(tokenizer, k=100, no_vipula=True, only_567_constraint=False)
    ])

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("san_Deva"),
            num_beams=num_beams,
            num_return_sequences=num_return,
            max_length=max_length,
            logits_processor=logits_processor,
            early_stopping=False,
            no_repeat_ngram_size=2,
            length_penalty=1.2,
        )

    batch_size = len(texts_batch)
    all_best = []

    for i in range(batch_size):
        start = i * num_return
        end = (i + 1) * num_return
        cand_outputs = outputs[start:end]

        results = []
        for out in cand_outputs:
            text = tokenizer.decode(out, skip_special_tokens=True)
            if not checker.has_valid_sanskrit(text):
                continue
            is_valid, reason, count, pada_status = checker.check(text)
            results.append({
                'text': text,
                'valid': is_valid,
                'syllables': count,
                'reason': reason,
                'pada_status': pada_status
            })

        perfect = [r for r in results if r['valid'] and r['syllables'] == 32]

        if perfect:
            best_verse = perfect[0]['text']
        else:
            if results:
                results.sort(key=lambda x: (not x['valid'], abs(x['syllables'] - 32), -x['syllables']))
                best_verse = results[0]['text']
            else:
                best_verse = tokenizer.decode(cand_outputs[0], skip_special_tokens=True) if len(cand_outputs) > 0 else ""

        all_best.append(best_verse)

    del outputs, inputs
    torch.cuda.empty_cache()

    return all_best


def main():
    parser = argparse.ArgumentParser(description="NeuroSymbolic Anushtup Generator (NLLB)")
    parser.add_argument("--input_csv", type=str, default="input.csv", help="Input CSV with 'English' column")
    parser.add_argument("--output_csv", type=str, default="translations_output.csv", help="Output CSV path")
    parser.add_argument("--model_name", type=str, default="sanganaka/nllb-200-distilled-1.3B-FT_anushtubh")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang="eng_Latn", tgt_lang="san_Deva")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()
    print(f"Model loaded: {args.model_name} on {device}")

    checker = AnushtupChecker()

    df_in = pd.read_csv(args.input_csv)
    if 'English' not in df_in.columns:
        raise ValueError(f"CSV must have 'English' column. Found: {df_in.columns.tolist()}")
    all_inputs = df_in['English'].tolist()

    batch_size = args.batch_size
    total_batches = (len(all_inputs) + batch_size - 1) // batch_size

    output_file = args.output_csv
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        start_idx = len(df_existing)
        translations = df_existing['Sanskrit'].tolist()
        print(f"Resuming from row {start_idx}")
    else:
        start_idx = 0
        translations = []

    start_time = datetime.now()

    for batch_num, start in enumerate(range(start_idx, len(all_inputs), batch_size), start=start_idx//batch_size + 1):
        batch_texts = all_inputs[start:start+batch_size]

        try:
            batch_best = generate_anushtup_batch(
                model, tokenizer, batch_texts, checker,
                num_beams=args.num_beams,
                num_return=args.num_beams,
                max_length=100,
                device=device
            )
            translations.extend(batch_best)

            df_out = pd.DataFrame({
                'English': all_inputs[:len(translations)],
                'Sanskrit': translations
            })
            df_out.to_csv(output_file, index=False)

            if batch_num % 10 == 0:
                checker.clear_cache()
                gc.collect()
                torch.cuda.empty_cache()

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Batch {batch_num}/{total_batches} | {len(translations)}/{len(all_inputs)} ({100*len(translations)/len(all_inputs):.1f}%) | {elapsed/60:.1f} min elapsed")

        except Exception as e:
            print(f"ERROR in batch {batch_num}: {str(e)}")
            df_out = pd.DataFrame({
                'English': all_inputs[:len(translations)],
                'Sanskrit': translations
            })
            df_out.to_csv(output_file, index=False)
            raise

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Complete. Output: {output_file} | {len(translations)} rows | {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
