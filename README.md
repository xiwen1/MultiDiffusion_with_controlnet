# MultiDiffusion with controlnet

This code is based on https://github.com/omerbt/MultiDiffusion. By adding controlnet into MultiDiffusion's region based diffusion pipeline, you can generate images with both region and sketch control, writing prompts for every mask. You can get more precise control over single controlnet or single MultiDiffusion through combination of the two.

## Usage
1. install requirements
```bash
pip install -r requirements.txt
```
2. prepare sketch image and corresponding masks. Then run code like this:
```bash
python multidiffusion_controlnet.py \
--mask_paths mask/house.png mask/tree.png \
--bg_prompt 'a photo of the a green field and blue sky, {photo-realistic}' \
--fg_prompts 'a house old and made of stone' 'one Cedar tree' \
--fg_negative small multiple \
--sketch 'sketch/simp_house_sketch_converted.jpg'
```