python multidiffusion_controlnet.py \
--mask_paths mask/house.png mask/tree.png \
--bg_prompt 'a photo of the a green field and blue sky, {photo-realistic}' \
--fg_prompts 'a house old and made of stone' 'one Cedar tree' \
--fg_negative small multiple \
--sketch 'sketch/simp_house_sketch_converted.jpg'