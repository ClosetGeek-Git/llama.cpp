# DistilBERT Support in llama.cpp

This document describes the changes made to add proper DistilBERT architecture support to llama.cpp, specifically fixing the classification head activation function mismatch.

## Problem

When running DistilBertForSequenceClassification models through llama.cpp, classification outputs differed from HuggingFace reference by ~0.1-0.3 in logit values, resulting in only 85.6% agreement on top-label predictions.

**Root cause**: llama.cpp applied `tanh()` activation to all BERT-like classification heads, but DistilBERT uses `ReLU()`.

| Model | Activation | Range |
|-------|------------|-------|
| BERT | `tanh` | [-1, 1] |
| DistilBERT | `ReLU` | [0, ∞) |
| RoBERTa | `tanh` | [-1, 1] |
| ModernBERT | `GELU` | ~[-0.17, ∞) |

## Solution

Added `LLM_ARCH_DISTILBERT` as a distinct architecture with `ggml_relu()` activation in the classification head.

## Files Modified

### 1. gguf-py/gguf/constants.py

**Line ~363** - Added enum value:
```python
MODEL_ARCH = IntEnum('MODEL_ARCH', [
    ...
    'BERT',
    'DISTILBERT',  # NEW
    ...
])
```

**Line ~784** - Added name mapping:
```python
MODEL_ARCH_NAMES = {
    ...
    MODEL_ARCH.DISTILBERT: "distilbert",
    ...
}
```

**Lines ~1451-1468** - Added tensor list (same as BERT):
```python
MODEL_ARCH.DISTILBERT: [
    MODEL_TENSOR.TOKEN_EMBD,
    MODEL_TENSOR.TOKEN_EMBD_NORM,
    MODEL_TENSOR.TOKEN_TYPES,
    MODEL_TENSOR.POS_EMBD,
    MODEL_TENSOR.OUTPUT_NORM,
    MODEL_TENSOR.ATTN_OUT_NORM,
    MODEL_TENSOR.ATTN_QKV,
    MODEL_TENSOR.ATTN_Q,
    MODEL_TENSOR.ATTN_K,
    MODEL_TENSOR.ATTN_V,
    MODEL_TENSOR.ATTN_OUT,
    MODEL_TENSOR.FFN_DOWN,
    MODEL_TENSOR.FFN_UP,
    MODEL_TENSOR.LAYER_OUT_NORM,
    MODEL_TENSOR.CLS,
    MODEL_TENSOR.CLS_OUT,
],
```

### 2. convert_hf_to_gguf.py

**Lines ~5531-5549** - New converter class:
```python
@ModelBase.register("DistilBertModel", "DistilBertForMaskedLM", "DistilBertForSequenceClassification")
class DistilBertModel(BertModel):
    model_arch = gguf.MODEL_ARCH.DISTILBERT

    def set_gguf_parameters(self):
        self.gguf_writer.add_layer_norm_eps(1e-12)
        super().set_gguf_parameters()

    def modify_tensors(self, data_torch, name, bid):
        if name.startswith("distilbert."):
            name = name[11:]  # Strip prefix
        if name.startswith("vocab_"):
            return []  # Skip MLM head
        return super().modify_tensors(data_torch, name, bid)
```

### 3. src/llama-arch.h

**Line ~27** - Added enum:
```cpp
enum llm_arch {
    ...
    LLM_ARCH_BERT,
    LLM_ARCH_DISTILBERT,  // NEW
    ...
};
```

### 4. src/llama-arch.cpp

**Line ~23** - Added name mapping:
```cpp
{ LLM_ARCH_DISTILBERT, "distilbert" },
```

**Lines ~750-767** - Added tensor list:
```cpp
case LLM_ARCH_DISTILBERT:
    return {
        LLM_TENSOR_TOKEN_EMBD,
        LLM_TENSOR_TOKEN_EMBD_NORM,
        LLM_TENSOR_TOKEN_TYPES,
        LLM_TENSOR_POS_EMBD,
        LLM_TENSOR_ATTN_OUT_NORM,
        LLM_TENSOR_ATTN_QKV,
        LLM_TENSOR_ATTN_Q,
        LLM_TENSOR_ATTN_K,
        LLM_TENSOR_ATTN_V,
        LLM_TENSOR_ATTN_OUT,
        LLM_TENSOR_LAYER_OUT_NORM,
        LLM_TENSOR_FFN_DOWN,
        LLM_TENSOR_FFN_UP,
        LLM_TENSOR_CLS,
        LLM_TENSOR_CLS_OUT,
    };
```

### 5. src/llama-model.cpp

**Line ~873** - Shared hparams loading:
```cpp
case LLM_ARCH_BERT:
case LLM_ARCH_DISTILBERT:
    // ... existing BERT hparams code
```

**Line ~3242** - Shared tensor loading:
```cpp
case LLM_ARCH_BERT:
case LLM_ARCH_DISTILBERT:
    // ... existing BERT tensor loading
```

### 6. src/models/bert.cpp

**Line ~27** - Position embeddings:
```cpp
if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_DISTILBERT) {
    inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
}
```

**Line ~136** - FFN activation:
```cpp
} else if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_DISTILBERT ||
           model.arch == LLM_ARCH_NOMIC_BERT_MOE || model.arch == LLM_ARCH_JINA_BERT_V3) {
```

### 7. src/llama-graph.cpp — THE KEY FIX

**Lines ~2133-2139** - Classification head activation:
```cpp
// classification head
if (cls) {
    cur = ggml_mul_mat(ctx0, cls, cur);
    if (cls_b) {
        cur = ggml_add(ctx0, cur, cls_b);
    }
    // DistilBERT uses ReLU, other BERT variants use tanh
    if (arch == LLM_ARCH_DISTILBERT) {
        cur = ggml_relu(ctx0, cur);
    } else {
        cur = ggml_tanh(ctx0, cur);
    }
}
```

## Build & Test

### 1. Build llama.cpp

```bash
cd /path/to/llama.cpp/build
cmake --build . -j6
sudo make install  # or use LD_LIBRARY_PATH
```

### 2. Convert Model

```bash
python convert_hf_to_gguf.py ./peer_support_bert \
    --outfile ./peer_support_bert/distilbert-f32.gguf \
    --outtype f32
```

### 3. Rebuild PHP Extension

```bash
cd /path/to/llama.cpp/tools/server-coro
./make.sh
```

### 4. Run Tests

```bash
# Python reference
python test_classify_compare.py > /tmp/python_output.jsonl

# PHP/llama.cpp
php test_classify_compare.php > /tmp/php_output.jsonl

# Compare
python compare_outputs.py
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| Agreement rate | 85.6% | **100%** |
| Avg logit diff | ~0.15 | **<0.001** |
| Max logit diff | ~0.35 | **<0.004** |

---

## Test Scripts

### Python Reference (test_classify_compare.py)

```python
#!/usr/bin/env python3
"""
Compare Python HuggingFace inference output with PHP/llama.cpp output.
Uses the exact same test phrases as test_inference.py.
Outputs logits (not sigmoid) to match PHP output format.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# Load model and tokenizer
model_path = "./peer_support_bert"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", file=__import__('sys').stderr)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Label order (must match training)
label_names = ["crisis", "general", "substance"]

def predict_logits(text):
    """Return raw logits (not sigmoid) to compare with PHP output."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0].tolist()

# 90 test phrases
test_phrases = [
    # Crisis (high-lethality & indirect)
    "I used again and I'm going to kill myself with a knife by cutting my throat. I'll be dead, I'm going to make myself that way",
    "I have a rope tied to the ceiling and I'm standing on the chair. I'm done.",
    "I filled the tub with water and I'm holding the razor. I can't stop now.",
    "I'm going to overdose on my meds tonight. I've already taken the first handful.",
    "I'm standing on the bridge right now. I'm jumping in 5 minutes.",
    "I already said goodbye to my dog. I don't think I'll be here much longer.",
    "I keep thinking everyone would be better off without me.",
    "I don't want to be here anymore. I just want it to stop.",
    "I can't take this pain anymore. I'm done.",
    "I just want to sleep and never wake up.",
    "I don't see the point of trying anymore.",
    "I feel like I'm already dead inside.",
    "I've been saving up my pills for weeks.",
    "I don't care if I live or die right now.",
    "I keep thinking about how easy it would be to just... not be here.",
    
    # Substance-focused
    "I drank a whole bottle of wine by myself last night.",
    "I relapsed after 90 days clean. I feel like a total failure.",
    "I've been craving heroin every minute today.",
    "I used cocaine at the party last night and I hate myself for it.",
    "It's been a year since my last drink, but I dream about it every night.",
    "I lied to my sponsor about using again.",
    "I walked past the liquor store three times today. It's been hard.",
    "I used fentanyl last weekend. I don't even care anymore.",
    "I'm white-knuckling my sobriety right now.",
    "I smoked weed to cope with the anxiety, but now I feel worse.",
    "I'm so proud of my 30 days, but I'm scared I'll lose it.",
    "I used again after my partner left me. It's like I couldn't stop.",
    "I keep thinking about how good it felt to use, even though it destroyed me.",
    "I'm in NA and I shared for the first time today.",
    "I drank to numb the pain, but now I have a hangover and the pain is worse.",
    
    # General emotional distress
    "I've been feeling really down since my breakup last week.",
    "Work is overwhelming me, and I can't seem to catch a break.",
    "I feel so lonely even when I'm around people.",
    "My anxiety has been through the roof lately.",
    "I keep crying for no reason. I don't know what's wrong with me.",
    "I'm struggling to get out of bed most mornings.",
    "Everything feels heavy, like I'm walking through mud.",
    "I miss my mom so much it hurts to breathe.",
    "I feel like I'm failing at everything—parenting, work, life.",
    "I just want someone to listen without trying to fix me.",
    "Some days I feel completely numb, like I'm watching my life from outside.",
    "I'm so tired of pretending I'm okay when I'm not.",
    "It's hard to stay hopeful when everything keeps going wrong.",
    "I feel stuck in a loop of negative thoughts.",
    "I don't enjoy things like I used to.",
    
    # Ambiguous / borderline
    "I feel like I'm drowning in this pain and just want it to stop.",
    "I'm so done with all the drama in my life.",
    "I feel like I'm carrying the weight of the world on my shoulders.",
    "I just want to disappear forever.",
    "I don't see any future for myself.",
    "I feel like there's no way out except death.",
    "I can't face another day like this.",
    "I feel like I'm a burden to everyone who loves me.",
    "I don't have anything left to live for.",
    "I just want the pain to end, and I know how to do it.",
    
    # Recovery-positive
    "90 days clean today! It's not easy, but I'm fighting for my life.",
    "I'm so proud of my 1 year chip. It wasn't easy.",
    "I'm grateful for another day clean.",
    "I made it through today without using—that's a win.",
    "I asked for help today, and it felt right.",
    
    # Duplicates for timing consistency
    "I used again and I'm going to kill myself with a knife by cutting my throat. I'll be dead, I'm going to make myself that way",
    "I drank a whole bottle of wine by myself last night.",
    "I've been feeling really down since my breakup last week.",
    "I already said goodbye to my dog. I don't think I'll be here much longer.",
    "I'm white-knuckling my sobriety right now.",
    "Work is overwhelming me, and I can't seem to catch a break.",
    "I'm going to overdose on my meds tonight. I've already taken the first handful.",
    "It's been a year since my last drink, but I dream about it every night.",
    "I feel so lonely even when I'm around people.",
    "I keep thinking everyone would be better off without me.",
    "I relapsed after 90 days clean. I feel like a total failure.",
    "My anxiety has been through the roof lately.",
    "I filled the tub with water and I'm holding the razor. I can't stop now.",
    "I don't enjoy things like I used to.",
    "I'm standing on the bridge right now. I'm jumping in 5 minutes.",
    "I'm struggling to get out of bed most mornings.",
    "I used cocaine at the party last night and I hate myself for it.",
    "Everything feels heavy, like I'm walking through mud.",
    "I don't want to be here anymore. I just want it to stop.",
    "I'm so proud of my 30 days, but I'm scared I'll lose it.",
    "I can't take this pain anymore. I'm done.",
    "I feel like I'm failing at everything—parenting, work, life.",
    "I just want someone to listen without trying to fix me.",
    "I smoked weed to cope with the anxiety, but now I feel worse.",
    "I miss my mom so much it hurts to breathe.",
    "I'm so tired of pretending I'm okay when I'm not.",
    "It's hard to stay hopeful when everything keeps going wrong.",
    "I feel stuck in a loop of negative thoughts.",
    "I walked past the liquor store three times today. It's been hard.",
    "I used fentanyl last weekend. I don't even care anymore."
]

# Output JSON lines format for easy comparison
print(f"# Python HuggingFace inference - {len(test_phrases)} phrases")
print(f"# Labels: {label_names}")
print()

for i, text in enumerate(test_phrases):
    logits = predict_logits(text)
    result = {
        "index": i,
        "text": text[:60] + "..." if len(text) > 60 else text,
        "logits": {label: round(logit, 4) for label, logit in zip(label_names, logits)}
    }
    print(json.dumps(result))
```

### PHP Test (test_classify_compare.php)

```php
<?php
/**
 * Compare PHP/llama.cpp classification with Python HuggingFace output.
 * Uses the exact same test phrases as test_inference.py.
 * Outputs JSON lines format for easy comparison.
 */

use Swoole\Coroutine as Co;

$model_path = '/path/to/peer_support_bert/distilbert-f32.gguf';

// 90 test phrases - EXACT COPY from test_inference.py
$test_phrases = [
    // Crisis (high-lethality & indirect)
    "I used again and I'm going to kill myself with a knife by cutting my throat. I'll be dead, I'm going to make myself that way",
    "I have a rope tied to the ceiling and I'm standing on the chair. I'm done.",
    "I filled the tub with water and I'm holding the razor. I can't stop now.",
    "I'm going to overdose on my meds tonight. I've already taken the first handful.",
    "I'm standing on the bridge right now. I'm jumping in 5 minutes.",
    "I already said goodbye to my dog. I don't think I'll be here much longer.",
    "I keep thinking everyone would be better off without me.",
    "I don't want to be here anymore. I just want it to stop.",
    "I can't take this pain anymore. I'm done.",
    "I just want to sleep and never wake up.",
    "I don't see the point of trying anymore.",
    "I feel like I'm already dead inside.",
    "I've been saving up my pills for weeks.",
    "I don't care if I live or die right now.",
    "I keep thinking about how easy it would be to just... not be here.",
    
    // Substance-focused
    "I drank a whole bottle of wine by myself last night.",
    "I relapsed after 90 days clean. I feel like a total failure.",
    "I've been craving heroin every minute today.",
    "I used cocaine at the party last night and I hate myself for it.",
    "It's been a year since my last drink, but I dream about it every night.",
    "I lied to my sponsor about using again.",
    "I walked past the liquor store three times today. It's been hard.",
    "I used fentanyl last weekend. I don't even care anymore.",
    "I'm white-knuckling my sobriety right now.",
    "I smoked weed to cope with the anxiety, but now I feel worse.",
    "I'm so proud of my 30 days, but I'm scared I'll lose it.",
    "I used again after my partner left me. It's like I couldn't stop.",
    "I keep thinking about how good it felt to use, even though it destroyed me.",
    "I'm in NA and I shared for the first time today.",
    "I drank to numb the pain, but now I have a hangover and the pain is worse.",
    
    // General emotional distress
    "I've been feeling really down since my breakup last week.",
    "Work is overwhelming me, and I can't seem to catch a break.",
    "I feel so lonely even when I'm around people.",
    "My anxiety has been through the roof lately.",
    "I keep crying for no reason. I don't know what's wrong with me.",
    "I'm struggling to get out of bed most mornings.",
    "Everything feels heavy, like I'm walking through mud.",
    "I miss my mom so much it hurts to breathe.",
    "I feel like I'm failing at everything—parenting, work, life.",
    "I just want someone to listen without trying to fix me.",
    "Some days I feel completely numb, like I'm watching my life from outside.",
    "I'm so tired of pretending I'm okay when I'm not.",
    "It's hard to stay hopeful when everything keeps going wrong.",
    "I feel stuck in a loop of negative thoughts.",
    "I don't enjoy things like I used to.",
    
    // Ambiguous / borderline
    "I feel like I'm drowning in this pain and just want it to stop.",
    "I'm so done with all the drama in my life.",
    "I feel like I'm carrying the weight of the world on my shoulders.",
    "I just want to disappear forever.",
    "I don't see any future for myself.",
    "I feel like there's no way out except death.",
    "I can't face another day like this.",
    "I feel like I'm a burden to everyone who loves me.",
    "I don't have anything left to live for.",
    "I just want the pain to end, and I know how to do it.",
    
    // Recovery-positive
    "90 days clean today! It's not easy, but I'm fighting for my life.",
    "I'm so proud of my 1 year chip. It wasn't easy.",
    "I'm grateful for another day clean.",
    "I made it through today without using—that's a win.",
    "I asked for help today, and it felt right.",
    
    // Duplicates for timing consistency
    "I used again and I'm going to kill myself with a knife by cutting my throat. I'll be dead, I'm going to make myself that way",
    "I drank a whole bottle of wine by myself last night.",
    "I've been feeling really down since my breakup last week.",
    "I already said goodbye to my dog. I don't think I'll be here much longer.",
    "I'm white-knuckling my sobriety right now.",
    "Work is overwhelming me, and I can't seem to catch a break.",
    "I'm going to overdose on my meds tonight. I've already taken the first handful.",
    "It's been a year since my last drink, but I dream about it every night.",
    "I feel so lonely even when I'm around people.",
    "I keep thinking everyone would be better off without me.",
    "I relapsed after 90 days clean. I feel like a total failure.",
    "My anxiety has been through the roof lately.",
    "I filled the tub with water and I'm holding the razor. I can't stop now.",
    "I don't enjoy things like I used to.",
    "I'm standing on the bridge right now. I'm jumping in 5 minutes.",
    "I'm struggling to get out of bed most mornings.",
    "I used cocaine at the party last night and I hate myself for it.",
    "Everything feels heavy, like I'm walking through mud.",
    "I don't want to be here anymore. I just want it to stop.",
    "I'm so proud of my 30 days, but I'm scared I'll lose it.",
    "I can't take this pain anymore. I'm done.",
    "I feel like I'm failing at everything—parenting, work, life.",
    "I just want someone to listen without trying to fix me.",
    "I smoked weed to cope with the anxiety, but now I feel worse.",
    "I miss my mom so much it hurts to breathe.",
    "I'm so tired of pretending I'm okay when I'm not.",
    "It's hard to stay hopeful when everything keeps going wrong.",
    "I feel stuck in a loop of negative thoughts.",
    "I walked past the liquor store three times today. It's been hard.",
    "I used fentanyl last weekend. I don't even care anymore."
];

Co\run(function () use ($model_path, $test_phrases) {
    ob_start();
    
    $init_args = [
        'app',
        '-m', $model_path,
        '--reranking',
        '-c', '128',
        '--n-gpu-layers', '0',
    ];
    
    if (!swoole_llama_init($init_args)) {
        ob_end_clean();
        fwrite(STDERR, "ERROR: Failed to init model\n");
        return;
    }
    
    while (swoole_llama_ready() === 0) {
        Co::sleep(0.1);
    }
    if (swoole_llama_ready() === -1) {
        ob_end_clean();
        fwrite(STDERR, "ERROR: Model failed to load\n");
        swoole_llama_shutdown();
        return;
    }
    
    ob_end_clean();
    
    $label_names = ['crisis', 'general', 'substance'];
    
    echo "# PHP llama.cpp inference - " . count($test_phrases) . " phrases\n";
    echo "# Labels: " . json_encode($label_names) . "\n";
    echo "\n";
    
    // Extract model name from path (filename without extension)
    $model_name = pathinfo($model_path, PATHINFO_FILENAME);
    
    foreach ($test_phrases as $i => $text) {
        $req = new \Llama\Request([
            'method' => 'POST',
            'path' => '/classify',
            'body' => json_encode([
                'model' => $model_name,
                'inputs' => $text,
            ]),
            'headers' => ['Content-Type' => ['application/json']],
        ]);
        
        $status = $req->getStatusCode();
        if ($status !== 200) {
            $data = $req->getData();
            $error = $data['error']['message'] ?? 'Unknown error';
            fwrite(STDERR, "ERROR at index $i: $error\n");
            continue;
        }
        
        $data = $req->getData();
        $predictions = $data['predictions'] ?? [];
        
        $logits = [];
        foreach ($predictions as $pred) {
            $logits[$pred['label']] = round($pred['score'], 4);
        }
        
        $result = [
            'index' => $i,
            'text' => strlen($text) > 60 ? substr($text, 0, 60) . '...' : $text,
            'logits' => $logits,
        ];
        
        echo json_encode($result) . "\n";
    }
    
    swoole_llama_shutdown();
});
```

### Comparison Script (compare_outputs.py)

```python
#!/usr/bin/env python3
"""
Compare Python vs PHP classification outputs and compute differences.
"""
import json

def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results

python_results = load_jsonl('/tmp/python_output.jsonl')
php_results = load_jsonl('/tmp/php_output.jsonl')

print(f"Python: {len(python_results)} results")
print(f"PHP:    {len(php_results)} results")
print()

labels = ['crisis', 'general', 'substance']
diffs = {l: [] for l in labels}
disagree_count = 0

print("=" * 80)
print(f"{'Index':<6} {'Text':<40} {'Label':<10} {'Python':>10} {'PHP':>10} {'Diff':>10}")
print("=" * 80)

for py, php in zip(python_results, php_results):
    py_logits = py['logits']
    php_logits = php['logits']
    
    py_top = max(py_logits, key=py_logits.get)
    php_top = max(php_logits, key=php_logits.get)
    
    if py_top != php_top:
        disagree_count += 1
        print(f"{py['index']:<6} {py['text'][:38]:<40}")
        print(f"       Python top: {py_top} ({py_logits[py_top]:.4f})")
        print(f"       PHP top:    {php_top} ({php_logits[php_top]:.4f})")
        print()
    
    for label in labels:
        diff = abs(py_logits[label] - php_logits.get(label, 0))
        diffs[label].append(diff)

print("=" * 80)
print(f"\nDisagreements (different top label): {disagree_count} / {len(python_results)}")
print(f"Agreement rate: {100 * (1 - disagree_count / len(python_results)):.1f}%")
print()

print("Average absolute logit differences:")
for label in labels:
    avg = sum(diffs[label]) / len(diffs[label])
    max_d = max(diffs[label])
    print(f"  {label:<12}: avg={avg:.4f}, max={max_d:.4f}")
```

## Future Work

- **ModernBERT**: Uses GELU activation, requires similar architecture-specific handling
- **Other BERT variants**: May have different activation functions in classification heads
