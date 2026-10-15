# Third-party notices

Parts of pyiqa are adapted from the projects below. Their licences ask that these
notices be kept with copies of that code. This list covers only code that is
present in this repository. Model weights are distributed separately, and the
OpenAI CLIP and S-Lab notices are already in `LICENSE-OpenAI-CLIP` and
`LICENSE-S-Lab`.

## Apache License 2.0

The full licence text is at the end of this file.

### BasicSR

<https://github.com/XPixelGroup/BasicSR>

Copyright 2018-2022 BasicSR Authors

License: Apache License 2.0

Used in: `pyiqa/train.py`, `pyiqa/models/base_model.py`, `pyiqa/models/general_iqa_model.py`, `pyiqa/models/lr_scheduler.py`, `pyiqa/data/__init__.py`, `pyiqa/data/data_sampler.py`, `pyiqa/data/data_util.py`, `pyiqa/data/prefetch_dataloader.py`, `pyiqa/data/transforms.py`, `pyiqa/losses/loss_util.py`, `pyiqa/losses/losses.py`, `pyiqa/utils/dist_util.py`, `pyiqa/utils/download_util.py`, `pyiqa/utils/img_util.py`, `pyiqa/utils/logger.py`, `pyiqa/utils/misc.py`, `pyiqa/utils/options.py`, `pyiqa/archs/niqe_arch.py`

### PIQ

<https://github.com/photosynthesis-team/piq>

No copyright line is stated in the project.

License: Apache License 2.0

Used in: `pyiqa/archs/brisque_arch.py`, `pyiqa/archs/fsim_arch.py`, `pyiqa/archs/vsi_arch.py`, `pyiqa/archs/func_util.py`, `pyiqa/utils/color_util.py`, `pyiqa/matlab_utils/resize.py`

### pytorch-fid

<https://github.com/mseitzer/pytorch-fid>

No copyright line is stated in the project.

License: Apache License 2.0

Used in: `pyiqa/archs/inception.py`

### torch-fidelity

<https://github.com/toshas/torch-fidelity>

Copyright 2020 Anton Obukhov

License: Apache License 2.0

Used in: `pyiqa/archs/interpolate_compat_tensorflow.py`

### A-FINE

<https://github.com/ChrisDud0257/AFINE>

Copyright 2025 CHEN Du

License: Apache License 2.0

Used in: `pyiqa/archs/afine_arch.py`, `pyiqa/archs/afineclip_model.py`

### MANIQA

<https://github.com/IIGROUP/MANIQA>

No copyright line is stated in the project.

License: Apache License 2.0

Used in: `pyiqa/archs/maniqa_arch.py`, `pyiqa/archs/maniqa_swin.py`

### DETR

<https://github.com/facebookresearch/detr>

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

License: Apache License 2.0

Used in: `pyiqa/archs/topiq_arch.py`, `pyiqa/archs/tres_arch.py`

### IL-NIQE (PyTorch)

<https://github.com/IceClear/IL-NIQE>

No copyright line is stated in the project.

License: Apache License 2.0

Used in: `pyiqa/archs/niqe_arch.py`

### PyTorch Image Models (timm)

<https://github.com/huggingface/pytorch-image-models>

Copyright 2019 Ross Wightman

License: Apache License 2.0

Used in: `pyiqa/archs/topiq_swin.py`, `pyiqa/archs/uranker_arch.py`

### MUSIQ (google-research)

<https://github.com/google-research/google-research/tree/master/musiq>

Copyright 2026 The Google Research Authors.

License: Apache License 2.0

Used in: `pyiqa/data/multiscale_trans_util.py`

### Hugging Face Transformers

<https://github.com/huggingface/transformers>

Copyright 2023 The HuggingFace Team. All rights reserved. (`modeling_attn_mask_utils.py`)

Copyright 2025 The HuggingFace Team. All rights reserved. (`qwen3_5_src/`)

License: Apache License 2.0

Used in: `pyiqa/archs/q_align/modeling_attn_mask_utils.py`, `pyiqa/archs/qrealign/qwen3_5_src/`

## MIT License

Each project below is under the MIT License with the copyright line shown:

### DISTS

<https://github.com/dingkeyan93/DISTS>

Copyright (c) 2020 Keyan Ding

License: MIT License

Used in: `pyiqa/archs/dists_arch.py`

### DeepDC

<https://github.com/h4nwei/DeepDC>

Copyright (c) 2020 Keyan Ding

License: MIT License

Used in: `pyiqa/archs/deepdc_arch.py`

### IQA-optimization (IQA_pytorch)

<https://github.com/dingkeyan93/IQA-optimization>

Copyright (c) 2020 dingkeyan93

License: MIT License

Used in: `pyiqa/archs/mad_arch.py`, `pyiqa/archs/ssim_arch.py`, `pyiqa/archs/vif_arch.py`

### DMM

<https://github.com/Baoliang93/DMM>

Copyright (c) 2024 Baoliang CHEN（陈宝亮）

License: MIT License

Used in: `pyiqa/archs/dmm_arch.py`

### clean-fid

<https://github.com/GaParmar/clean-fid>

Copyright (c) 2021 Gaurav Parmar

License: MIT License

Used in: `pyiqa/archs/fid_arch.py`

### AHIQ

<https://github.com/IIGROUP/AHIQ>

Copyright (c) 2022 ShanShan Lao

License: MIT License

Used in: `pyiqa/archs/ahiq_arch.py`

### PyTorchSteerablePyramid

<https://github.com/tomrunia/PyTorchSteerablePyramid>

Copyright (c) 2018 Tom Runia (tomrunia@gmail.com)

License: MIT License

Used in: `pyiqa/matlab_utils/scfpyr_util.py`, `pyiqa/matlab_utils/math_util.py`

### mPLUG-Owl

<https://github.com/X-PLUG/mPLUG-Owl>

Copyright (c) 2022 mPLUG

License: MIT License

Used in: `pyiqa/archs/q_align/configuration_mplug_owl2.py`, `pyiqa/archs/q_align/modeling_mplug_owl2.py`, `pyiqa/archs/q_align/cmp_modelling_mplug_owl2.py`, `pyiqa/archs/q_align/modeling_llama2.py`, `pyiqa/archs/q_align/visual_encoder.py`

MIT License text:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD 2-Clause License

### LPIPS

<https://github.com/richzhang/PerceptualSimilarity>

Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang

License: BSD 2-Clause License

Used in: `pyiqa/archs/lpips_arch.py`, `pyiqa/archs/stlpips_arch.py`

### ST-LPIPS

<https://github.com/abhijay9/ShiftTolerant-LPIPS>

Copyright (c) 2022, Abhijay Ghildyal, Feng Liu

License: BSD 2-Clause License

Used in: `pyiqa/archs/stlpips_arch.py`

BSD 2-Clause License text (with the copyright lines above):

```
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## PieAPP: Modified BSD-2 License, for non-commercial use only

<https://github.com/prashnani/PerceptualImageError>

License: Modified BSD-2 License, for Non-Commercial Use Only

Used in: `pyiqa/archs/pieapp_arch.py`

```
Modified BSD-2 License - for Non-Commercial Use Only 

Copyright (c) 2018, The Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted for non-commercial use only provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.

For permission to use for commercial purposes, please contact UCSB’s 
Office of Technology & Industry Alliances at 805-893-5180 or info@tia.ucsb.edu.
```

## Creative Commons

### Underwater Ranker (URanker)

<https://github.com/RQ-Wu/UnderwaterRanker>

By the URanker authors. No copyright line is stated in the project.

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
<https://creativecommons.org/licenses/by-nc/4.0/>. The pyiqa file is adapted from it.

Used in: `pyiqa/archs/uranker_arch.py`

### antialiased-cnns (Adobe)

<https://github.com/adobe/antialiased-cnns>

Copyright 2019 Adobe. All rights reserved.

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),
<https://creativecommons.org/licenses/by-nc-sa/4.0/>. The pyiqa file is adapted from it,
and keeps the full notice.

Used in: `pyiqa/archs/stlpips_arch.py`

## Apache License 2.0 text

```
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
