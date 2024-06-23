# Model Conversion 2 Safetensors
---

A simple converter which converts PyTorch model files (.bin, .pt, .pth) to SafeTensors format. 

### Why SafeTensors?
- SafeTensors format decreases the loading time of large models.
- Supports in-place loading, which effectively decreases the required memory to load a model.
- Because the safetensor format decreases the loading time of large LLM models, currently supported in [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui); which also supports in-place loading, allowing for less required memory to load a LLM.

### Installation:

To use this converter you need to have Python installed, if you arleady have Python installed simply paste the following into cmd & you'll be all set to convert!

```
git clone https://github.com/MackinationsAi/convert-2-safetensors.git
```
```
cd convert-2-safetensors repo
```
```
install.bat
```

### Usage:

The script can be used to convert individual models or batch convert multiple models in a directory (including all other models in subfolder if user specified in cmd).

#### Convert Single Model or Multipl Models:

1. Run convert.bat
2. Copy the model(s) file (.bin, .pt, or .pth) directory_path you wish to convert.
3. Follow the cmd prompts:

- **Source folder for PyTorch files:** Specify the directory containing your model file(s). Leave blank to use the script's directory.
- **Destination folder for SafeTensors files:** Specify the directory where the converted files will be saved. Leave blank to use a default subdirectory.
- **Delete old PyTorch files? (Y/N):** Choose whether to delete the original PyTorch files after conversion.
- **Do you want to convert a batch of models? (Y/N):** Choose whether to convert multiple models in the specified directory and its subdirectories.

4. Once the models have been converted you will be prompted w/ the following:
- **Would you like to convert another model or models to .safetensors? (Y/N):** Choose whether you wish to start over & convert another model or multiple models in a different specified directory & its subdirectories. (If you input Y or y, this will bring you back to #2-3, & you will have to repeat the steps listed above. If you don't wish to convert any other models at that time simple input N or n & it will exit the cmd.)

#### Example:

```
run convert.bat
```

```
Source folder for PyTorch files (leave blank for script's directory): /path/to/models
Destination folder for SafeTensors files (leave blank for default): /path/to/save/safetensors
Delete old PyTorch files? (Y/N): Y
Do you want to convert a batch of models? (Y/N): Y
Would you like to convert another model or models to .safetensors? (Y/N): N
```

### Features:

- Supports converting individual PyTorch model files (.bin, .pt, .pth) to SafeTensors format.
- Supports batch conversion of multiple models in a directory and it's subdirectories.
- Converts models in-place unless the user chooses to copy in a different output folder path.

### Limitations:

- The program requires a significant amount of memory. Your idle memory should be at least twice the size of your largest model file to avoid running out of memory... that would be **slow!**

- This program **will not** re-shard (aka break down) the model, you'll need to do it yourself using some other tools.

### Notes:

- Most of the code originated from [Convert to Safetensors - a Hugging Face Space by safetensors.](https://huggingface.co/spaces/safetensors/convert).
- This github repo also used portions of code from [Silver267's - pytorch-to-safetensor-converter repo.](https://github.com/Silver267/pytorch-to-safetensor-converter)
- The script has been enhanced to handle various model file formats and support batch conversion across subdirectories.
