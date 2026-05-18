#include "models.h"

// DistilBERT shares BERT's hparams, tensor topology, and graph. The only
// architectural difference exposed by the converter is the activation function
// used in the classifier head (ReLU instead of tanh); that dispatch lives in
// build_pooling() (src/llama-graph.cpp), keyed on model.arch.
//
// Tensor loading is inherited from llama_model_bert. bert.cpp's
// load_arch_tensors() already gates the classifier-head tensors on
// (LLM_ARCH_BERT || LLM_ARCH_DISTILBERT). Only the layer-count → model-type
// mapping is unique to DistilBERT, so that's the only override here.

void llama_model_distilbert::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 6:  type = LLM_TYPE_70M;  break; // distilbert-base (~66M params)
        default: type = LLM_TYPE_UNKNOWN;
    }
}
