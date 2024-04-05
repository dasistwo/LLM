This script is not runnable by itself. It is rather a note for running wikitext perplexity task.

``` zsh
accelerate launch -m lm-eval --model hf --model_args pretrained=google/gemma-2b --tasks wikitext
```

This script runs Nsight Compute to profile the TextGen.
``` zsh
ncu --target-processes all jupyter execute ./nb/TextGen.ipynb
```

This script runs Nsight Systems
``` zsh
nsys profile --trace=cuda,nvtx --output=./output/nsys/TextGen --force-overwrite true jupyter execute ./nb/TextGen.ipynb
```