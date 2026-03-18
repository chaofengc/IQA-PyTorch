#    Copyright 2023 Haotian Liu & Qinghao Ye (Modified from LLaVA)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from PIL import Image

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mplug_owl2 import MPLUGOwl2Config, MplugOwlVisionConfig, MplugOwlVisualAbstractorConfig
from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel
from .modeling_llama2 import LlamaModel, LlamaForCausalLM, replace_llama_modality_adaptive
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids if len(chunk) > 0 else [] for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def optimize_score_map_pytorch_cuda(c, seed=0, original_seed=20020, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    c = torch.tensor(c, dtype=torch.float32, device=device, requires_grad=False)
    initial_scores = torch.rand(c.shape[0], device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([initial_scores], lr=0.1)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        sum_log_diff = torch.sum(c * torch.log(torch.maximum(norm_cdf(initial_scores[:, None] - initial_scores), torch.tensor(1e-6, device=device))))
        sum_squares = torch.sum(initial_scores ** 2) / 2

        loss = -(sum_log_diff - sum_squares)
        loss.backward()
        optimizer.step()
    
    optimized_scores = initial_scores.detach().cpu().numpy()
    min_score, max_score = np.min(optimized_scores), np.max(optimized_scores)
    
    # Scale scores to 0-100
    scaled_scores = 100 * (optimized_scores - min_score) / (max_score - min_score)
    
    # Reset the seed
    np.random.seed(original_seed)
    return torch.tensor(scaled_scores[-1], device=device)

def softmax(logits):
    # exp_logits = np.exp(logits - np.max(logits))
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs
    # return exp_logits / exp_logits.sum()

def update_matrix(anchor_matrix, scores, indices):
    n = anchor_matrix.shape[0]
    new_row = np.zeros((1, n))
    new_col = np.zeros((n + 1, 1))
    new_row[0, indices] = scores
    new_col[indices, 0] = 1-scores  # Assuming symmetric preference for simplicity
    anchor_matrix = np.vstack([anchor_matrix, new_row])
    anchor_matrix = np.hstack([anchor_matrix, new_col])
    anchor_matrix[n, n] = 0.5
    return anchor_matrix
    

class MPLUGOwl2MetaModel:
    def __init__(self, config):
        super(MPLUGOwl2MetaModel, self).__init__(config)
        self.vision_model = MplugOwlVisionModel(
            MplugOwlVisionConfig(**config.visual_config["visual_model"])
        )
        self.visual_abstractor = MplugOwlVisualAbstractorModel(
            MplugOwlVisualAbstractorConfig(**config.visual_config["visual_abstractor"]), config.hidden_size
        )
    
    def get_vision_tower(self):
        vision_model = getattr(self, 'vision_model', None)
        if type(vision_model) is list:
            vision_model = vision_model[0]
        return vision_model

    def get_visual_abstractor(self):
        visual_abstractor = getattr(self, 'visual_abstractor', None)
        if type(visual_abstractor) is list:
            visual_abstractor = visual_abstractor[0]
        return visual_abstractor


class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_model(images).last_hidden_state
        image_features = self.get_model().visual_abstractor(encoder_hidden_states=image_features).last_hidden_state
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            multiway_indices = torch.zeros_like(input_ids).long().to(self.device)
            return input_ids, multiway_indices, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                
                cur_modality_indicators = torch.zeros(len(cur_input_embeds)).long().to(self.device)
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                
                # Add modality indicator
                assert image_token_start == len(cur_input_ids[:image_token_start])
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids[:image_token_start])).long())
                cur_modality_indicators.append(torch.ones(len(cur_image_features)).long())
                
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            # Modality
            cur_modality_indicators = [x.to(device=self.device) for x in cur_modality_indicators]
            cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            new_modality_indicators.append(cur_modality_indicators)
            
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            # Embedding
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            # Modality
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat((cur_modality_indicator, torch.zeros(max_len - cur_modality_indicator.shape[0], dtype=cur_modality_indicator.dtype, device=cur_modality_indicator.device)), dim=0)
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)
            
            # Label
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            # Attention Mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, new_modality_indicators, attention_mask, past_key_values, new_input_embeds, new_labels



class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, LlamaModel):
    config_class = MPLUGOwl2Config

    def __init__(self, config: MPLUGOwl2Config):
        super(MPLUGOwl2LlamaModel, self).__init__(config)


class MPLUGOwl2LlamaForCausalLM(LlamaForCausalLM, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MPLUGOwl2LlamaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained("VQA-CityU/Compare2Score_1")
        self.image_processor = CLIPImageProcessor.from_pretrained("VQA-CityU/Compare2Score_1")

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.preferential_ids_ = [id_[1] for id_ in self.tokenizer(["inferior", "worse", "similar", "better", "superior"])["input_ids"]]
        self.anchor_images = load_dataset("VQA-CityU/Anchor_images")
        
        self.weight_tensor = np.array([0., 0.25, 0.5, 0.75, 1.], dtype=np.float16)
        self.anchor_matrix = np.array(
            [[5.0000000e-01, 2.5912809e-01, 3.3130276e-04, 1.6087297e-06, 1.1803027e-09],
             [7.4087191e-01, 5.0000000e-01, 2.4985345e-01, 9.9954158e-02, 1.8675303e-08],
             [9.9966872e-01, 7.5014657e-01, 5.0000000e-01, 4.9968880e-01, 2.4852838e-01],
             [9.9999839e-01, 9.0004587e-01, 5.0031120e-01, 5.0000000e-01, 2.5400183e-01],
             [1.0000000e+00, 1.0000000e+00, 7.5147164e-01, 7.4599814e-01, 5.0000000e-01]], 
            dtype=np.float32)
        anchor_intervals = 5#16
        num_anchor_image_per_interval = 1
        num_anchor_image = anchor_intervals * num_anchor_image_per_interval
        self.anchor_indices = np.arange(0,num_anchor_image)
        # Initialize weights and apply final processing
        self.post_init()
        

    def get_model(self):
        return self.model

    def score(self, image):
        prompt = "USER: <|image|> <|image|> Compared with the first image, what is your quality rating for second image? \nASSISTANT: The quality of the second image is"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        anchor_images = [item['image'] for item in self.anchor_images['train']]
        
        probabilities = []
        for index in self.anchor_indices:
            anchor_image = anchor_images[index]
            images = [anchor_image, image]
            images = [expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in images]
            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(self.device)
            
            with torch.inference_mode():
                output_logits = self(input_ids, images=image_tensor)["logits"][:, -1, self.preferential_ids_]
                output_logits = output_logits.cpu().detach().numpy() / 100
                probabilities.append(np.dot(softmax(output_logits),  self.weight_tensor))
        updated_matrix = update_matrix(self.anchor_matrix, np.squeeze(np.array(probabilities)), self.anchor_indices)
        score = optimize_score_map_pytorch_cuda(updated_matrix, seed=0, original_seed=20020, num_iterations=100)
        return score

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        # modality_indicators: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, modality_indicators, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("mplug_owl2", MPLUGOwl2Config)
AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)

replace_llama_modality_adaptive()
 