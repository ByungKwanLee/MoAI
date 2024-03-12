from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput
from torch import nn

from .utils.utils import *
from .arch.build_mlp import build_vision_projector, build_vision_tower
from .arch.expert_module import PerceiverResampler
from .arch.modeling_internlm2 import InternLM2Model, InternLM2PreTrainedModel
from transformers.cache_utils import Cache

@dataclass
class MoAICausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None

class MoAIModel(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_length = config.max_length
        
        # Initialize weights and apply final processing
        self.post_init()
        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()
        
        # image processing variable
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,-1,1,1) * 255
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,-1,1,1) * 255

        # MoAI-Compressor
        self.moai_compressor = PerceiverResampler()

    def image_processor(self, images):
        norm_images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return norm_images

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def eval_process(
        self,
        images,
        prompts,
        processor,
        seg_model,
        seg_processor,
        od_model,
        od_processor,
        sgg_model,
        ocr_model,
        device,
        mode='moai_eval.yaml'):

        # Segmentation Inputs
        seg_inputs = seg_processor(images=[input for input in images], return_tensors="pt")

        # Segmentation Outputs
        with torch.inference_mode():
            seg_model.eval()
            seg_results = seg_processor.post_process_panoptic_segmentation(seg_model(**{k:v.to(device) for k, v in seg_inputs.items()}), 
                                                                           target_sizes=[(490, 490)]*len(images),
                                                                           threshold=0.5,
                                                                           mask_threshold=0.95,
                                                                           label_ids_to_fuse=())


        # OWOD Inputs
        from utils.constants import ADE20K_847, IMAGENET_CLASSES
        od_inputs = od_processor(text=[ADE20K_847+IMAGENET_CLASSES], images=[input for input in images], return_tensors="pt")

        # OWOD Outputs
        with torch.inference_mode():
            od_model.eval()
            od_results = od_processor.post_process_object_detection(od_model(**{k:v.to(device) for k, v in od_inputs.items()}), 
                                                                    threshold=0.1,
                                                                    target_sizes=[(490, 490)]*len(images))
            
        # SGG Outputs
        from mmdet.apis import inference_detector
        with torch.inference_mode():
            sgg_results = inference_detector(sgg_model, imgs=[input.permute(1,2,0).cpu().numpy() for input in images])


        batched_verb_prompt=[]
        batched_panoptic_map = []
        batched_moai_prompt=[]
        batched_im_mask=[]
        batched_lang_mask=[]
        for image, prompt, seg_result, od_result, sgg_result in zip(images, prompts, seg_results, od_results, sgg_results):

            # Panoptic Index and Class Index
            verbalization_seg, panoptic_map, nice_seg_info_list = make_seg_prompt(seg_result)
            
            # OWOD Detection
            verbalization_od, od_boxes, od_labels = make_od_prompt(od_result)
            
            # SGG
            verbalization_sgg = make_sgg_prompt(sgg_result)
            
            # OCR
            with torch.inference_mode():
                ocr_result = ocr_model.ocr(image.permute(1,2,0).cpu().numpy())
            verbalization_ocr = make_ocr_prompt(ocr_result[0])
            
            # Aux prompt
            verbalization_aux = make_human_string(verbalization_seg, verbalization_od, verbalization_sgg, verbalization_ocr)

            # moai prompt prefix
            moai_prompt, _, im_mask = make_system_prompt(processor, device, self.config.ignore_index)
            moai_prompt, im_mask = make_and_add_prompt_and_im_mask(moai_prompt=moai_prompt, 
                                                                    im_mask=im_mask, 
                                                                    prompt=prompt, 
                                                                    processor=processor, 
                                                                    device=device)
            # lang mask
            lang_mask = 1 - im_mask

            # making batched moai prompt
            batched_verb_prompt.append(verbalization_aux)
            batched_panoptic_map.append(torch.from_numpy(panoptic_map).permute(2, 0, 1))
            batched_moai_prompt.append(moai_prompt)
            batched_im_mask.append(im_mask.flip(dims=[0])) # padding left
            batched_lang_mask.append(lang_mask.flip(dims=[0])) # padding left
        
        '''For Final Outputs'''
        moai_inputs = processor(batched_moai_prompt, padding='longest', return_tensors="pt")

        # [1] input_ids
        input_ids = moai_inputs.input_ids.to(device)

        # [2] pixel values
        pixel_values = self.image_processor(images).to(device)

        # [3] aux embed
        verb_embeds = self.get_input_embeddings()(processor(batched_verb_prompt, padding='longest', return_tensors="pt").input_ids)
        with torch.inference_mode(): self.vit.vision_tower.eval(); map_embeds = self.vision_proj(self.vit(self.image_processor(torch.stack(batched_panoptic_map).to(torch.float16)).to(device)))
        aux_embeds = torch.cat([verb_embeds, map_embeds], dim=1)
        
        # [4] attention_mask
        attention_mask = moai_inputs.attention_mask.to(device)

        # [5] im_mask
        im_mask = torch.nn.utils.rnn.pad_sequence(batched_im_mask, batch_first=True, padding_value=0).flip(dims=[1]).bool() # padding left
        
        # [6] lang_mask
        lang_mask = torch.nn.utils.rnn.pad_sequence(batched_lang_mask, batch_first=True, padding_value=0).flip(dims=[1]).bool() # padding left

        return {"input_ids": input_ids, 
                "pixel_values": pixel_values, 
                "aux_embeds": aux_embeds, 
                "attention_mask": attention_mask, 
                "im_mask": im_mask, 
                "lang_mask": lang_mask,
                "mode": mode}

    def demo_process(
        self,
        image,
        prompt,
        processor,
        seg_model,
        seg_processor,
        od_model,
        od_processor,
        sgg_model,
        ocr_model,
        device,
        mode='moai_eval.yaml'):

        # RGB Dimension
        image = image[:3]
        
        # Segmentation Inputs
        seg_inputs = seg_processor(images=[image], return_tensors="pt")

        # Segmentation Outputs
        with torch.inference_mode():
            seg_model.eval()
            seg_results = seg_processor.post_process_panoptic_segmentation(seg_model(**{k:v.to(device) for k, v in seg_inputs.items()}), 
                                                                           target_sizes=[(490, 490)],
                                                                           threshold=0.5,
                                                                           mask_threshold=0.95,
                                                                           label_ids_to_fuse=())


        # OWOD Inputs
        from utils.constants import ADE20K_847, IMAGENET_CLASSES
        od_inputs = od_processor(text=[ADE20K_847+IMAGENET_CLASSES], images=[image], return_tensors="pt")

        # OWOD Outputs
        with torch.inference_mode():
            od_model.eval()
            od_results = od_processor.post_process_object_detection(od_model(**{k:v.to(device) for k, v in od_inputs.items()}), 
                                                                    threshold=0.1,
                                                                    target_sizes=[(490, 490)])
            
        # SGG Outputs
        from mmdet.apis import inference_detector
        with torch.inference_mode():
            sgg_results = inference_detector(sgg_model, imgs=[image.permute(1,2,0).cpu().numpy()])

        # OCR Outputs
        with torch.inference_mode():
            ocr_results = ocr_model.ocr(image.permute(1,2,0).cpu().numpy())


        # Panoptic Index and Class Index
        verbalization_seg, panoptic_map, nice_seg_info_list = make_seg_prompt(seg_results[0])
        
        # OWOD Detection
        verbalization_od, od_boxes, od_labels = make_od_prompt(od_results[0])
        
        # SGG
        verbalization_sgg = make_sgg_prompt(sgg_results[0])
        
        # OCR
        verbalization_ocr = make_ocr_prompt(ocr_results[0])
        
        # Aux prompt
        verbalization_aux = make_human_string(verbalization_seg, verbalization_od, verbalization_sgg, verbalization_ocr)

        # moai prompt prefix
        moai_prompt, _, im_mask = make_system_prompt(processor, device, self.config.ignore_index)
        moai_prompt, im_mask = demo_make_and_add_prompt_and_im_mask(moai_prompt=moai_prompt, 
                                                                    im_mask=im_mask, 
                                                                    prompt=prompt, 
                                                                    processor=processor, 
                                                                    device=device)
        # lang mask
        lang_mask = 1 - im_mask


        # making batched moai prompt
        batched_verb_prompt=[verbalization_aux]
        batched_panoptic_map = [torch.from_numpy(panoptic_map).permute(2, 0, 1)]
        batched_moai_prompt=[moai_prompt]
        batched_im_mask=[im_mask.flip(dims=[0])]
        batched_lang_mask=[lang_mask.flip(dims=[0])]
        
        '''For Final Outputs'''
        moai_inputs = processor(batched_moai_prompt, padding='longest', return_tensors="pt")

        # [1] input_ids
        input_ids = moai_inputs.input_ids.to(device)

        # [2] pixel values
        pixel_values = self.image_processor(image.unsqueeze(0)).to(device)

        # [3] aux embed
        verb_embeds = self.get_input_embeddings()(processor(batched_verb_prompt, padding='longest', return_tensors="pt").input_ids)
        with torch.inference_mode(): self.vit.vision_tower.eval(); map_embeds = self.vision_proj(self.vit(self.image_processor(torch.stack(batched_panoptic_map).to(torch.float16)).to(device)))
        aux_embeds = torch.cat([verb_embeds, map_embeds], dim=1)
        
        # [4] attention_mask
        attention_mask = moai_inputs.attention_mask.to(device)

        # [5] im_mask
        im_mask = torch.nn.utils.rnn.pad_sequence(batched_im_mask, batch_first=True, padding_value=0).flip(dims=[1]).bool() # padding left
        
        # [6] lang_mask
        lang_mask = torch.nn.utils.rnn.pad_sequence(batched_lang_mask, batch_first=True, padding_value=0).flip(dims=[1]).bool() # padding left

        return {"input_ids": input_ids, 
                "pixel_values": pixel_values, 
                "aux_embeds": aux_embeds, 
                "attention_mask": attention_mask, 
                "im_mask": im_mask, 
                "lang_mask": lang_mask,
                "mode": mode}


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            aux_embeds: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            im_mask: torch.BoolTensor = None,
            lang_mask: torch.BoolTensor = None,
            mode: str = 'moai_eval.yaml',
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, MoAICausalLMOutputWithPast]:
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # MoAI
        self.vit.vision_tower.eval()

        if inputs_embeds is None:

            # TODO: For batch generation
            input_ids[torch.where(input_ids==-100)]=2

            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vit(pixel_values)
                comp_aux_embeds = self.moai_compressor(aux_embeds).to(image_outputs.dtype)
                
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                image_features = self.vision_proj(image_outputs)
                inputs_embeds, attention_mask, _, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                    im_mask = torch.zeros(inputs_embeds.shape[:2]).bool().to(inputs_embeds.device)
                    lang_mask = torch.zeros(inputs_embeds.shape[:2]).bool().to(inputs_embeds.device)
                    comp_aux_embeds = None

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            aux_embeds=comp_aux_embeds,
            im_mask=im_mask,
            lang_mask=lang_mask,
            mode=mode,
        )
        logits = self.output(outputs[0])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Try except handling for use_cache=True
        return MoAICausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, aux_embeds=None, attention_mask=None, im_mask=None, lang_mask=None, mode='moai_eval.yaml', **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "aux_embeds": aux_embeds,
                "im_mask": im_mask,
                "lang_mask": lang_mask,
                "mode": mode
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past